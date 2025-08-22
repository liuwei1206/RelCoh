# author = liuwei
# date = 2024-01-05

import os
import json
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm, trange

random.seed(106524)


class LSTMDataset(Dataset):
    def __init__(self, file_name_or_data, params):
        self.max_seq_length = params["max_seq_length"]
        self.max_rel_length = params["max_rel_length"]
        self.label_list = params["label_list"]
        self.rel_list = params["rel_list"]
        self.vocab = params["vocab"]

        self.init_dataset(file_name_or_data)

    def convert_tokens_to_ids(self, tokens):
        token_ids = []
        for token in tokens:
            if token in self.word2idx:
                token_ids.append(self.word2idx[token])
            else:
                tokens.append(self.word2idx["<unk>"])  # 0 as <pad> id

        return token_ids

    def init_dataset(self, file_name_or_data):
        ## 1. init word2idx
        self.word2idx = {}
        self.rel2idx = {}
        for idx, word in enumerate(self.vocab):
            self.word2idx[word] = idx
        for idx, rel in enumerate(self.rel_list):
            self.rel2idx[rel] = idx

        ## 2. read data
        if type(file_name_or_data) == str and os.path.exists(file_name_or_data):
            with open(file_name_or_data, "r", encoding="utf-8") as f:
                lines = f.readlines()
        elif type(file_name_or_data) == list:
            lines = file_name_or_data
        else:
            raise Exception("Not file name or data list!!!")

        ## 3. prepare data
        all_input_ids = []
        all_rel_ids = []
        all_seq_length = []
        all_rel_length = []
        all_label_ids = []
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                text = sample["text"]
                rels = sample["rels"]
                label = sample["score"]
                spans = sample["spans"]
                """
                filter_rels = []
                for rel, span in zip(rels, spans):
                    if span[0] != span[1]: # only inter-sentential relations
                        filter_rels.append(rel)
                rels = filter_rels
                """
                input_tokens = nltk.word_tokenize(text)
                input_ids = self.convert_tokens_to_ids(input_tokens)
                rel_ids = [self.rel2idx[rel] for rel in rels]
                label_id = self.label_list.index(label)

                np_input_ids = np.zeros(self.max_seq_length, dtype=np.int32)
                np_rel_ids = np.zeros(self.max_rel_length, dtype=np.int32)
                if len(input_ids) > self.max_seq_length:
                    input_ids = input_ids[:self.max_seq_length]
                if len(rel_ids) > self.max_rel_length:
                    rel_ids = rel_ids[:self.max_rel_length]
                seq_length = len(input_ids)
                rel_length = len(rel_ids)
                np_input_ids[:seq_length] = input_ids
                np_rel_ids[:rel_length] = rel_ids

                all_input_ids.append(np_input_ids)
                all_rel_ids.append(np_rel_ids)
                all_seq_length.append(seq_length)
                all_rel_length.append(rel_length)
                all_label_ids.append(label_id)

        self.input_ids = all_input_ids
        self.rel_ids = all_rel_ids
        self.seq_lengths = all_seq_length
        self.rel_lengths = all_rel_length
        self.label_ids = all_label_ids
        self.total_size = len(all_input_ids)
        # for p_id in range(10):
        #     print(all_rel_ids[p_id][:10])

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        return (
            torch.tensor(self.input_ids[index]),
            torch.tensor(self.rel_ids[index]),
            torch.tensor(self.seq_lengths[index]),
            torch.tensor(self.rel_lengths[index]),
            torch.tensor(self.label_ids[index])
        )

def merge_sents(sents, spans, rels, tokenizer):
    """
    Merge sents into a whole document, and return the boundary of each sents, each argument pair
    Return:
        text:
        sent_starts:
        sent_ends:
        arg1_starts:
        arg1_ends:
        arg2_starts:
        arg2_ends:
        rels
    """
    # 1. document
    whole_document = " ".join(sents)

    # 2. for sent boundary
    cur_text = ""
    sent_start_ids = []
    sent_end_ids = []
    for sent in sents:
        if cur_text == "":
            start_pos = 1
        else:
            start_pos = 1 + len(tokenizer.tokenize(cur_text))
        cur_text = cur_text + " " + sent
        end_pos = 1 + len(tokenizer.tokenize(cur_text))
        sent_start_ids.append(start_pos)
        sent_end_ids.append(end_pos)

    # 3. for arguments
    arg1_start_ids = []
    arg1_end_ids = []
    arg2_start_ids = []
    arg2_end_ids = []
    fine_spans = []
    fine_rels = []
    for span, rel in zip(spans, rels):
        sent_id_1 = span[0]
        sent_id_2 = span[1]
        arg1_start_offset = span[3][0]
        arg1_end_offset = span[3][1]
        arg2_start_offset = span[4][0]
        arg2_end_offset = span[4][1]
        if (arg1_start_offset == arg1_end_offset) or (arg2_start_offset == arg2_end_offset):
            continue

        if sent_id_1 == sent_id_2:
            if sent_id_1 == 0:
                arg1_start_pos = 1
                cur_text = ""
            else:
                cur_text = " ".join(sents[:sent_id_1])
                arg1_start_pos = 1 + len(tokenizer.tokenize(cur_text))
            cur_text = cur_text + " " + sents[sent_id_1][:arg1_end_offset]
            arg1_end_pos = 1 + len(tokenizer.tokenize(cur_text))
            arg2_start_pos = arg1_end_pos
            cur_text = " ".join(sents[:(sent_id_1 + 1)])
            arg2_end_pos = 1 + len(tokenizer.tokenize(cur_text))
        else:
            if sent_id_1 == 0:
                arg1_start_pos = 1
            else:
                cur_text = " ".join(sents[:sent_id_1])
                arg1_start_pos = 1 + len(tokenizer.tokenize(cur_text))
            cur_text = " ".join(sents[:sent_id_1 + 1])
            arg1_end_pos = 1 + len(tokenizer.tokenize(cur_text))
            arg2_start_pos = arg1_end_pos
            cur_text = " ".join(sents[:sent_id_2 + 1])
            arg2_end_pos = 1 + len(tokenizer.tokenize(cur_text))

        arg1_start_ids.append(arg1_start_pos)
        arg1_end_ids.append(arg1_end_pos)
        arg2_start_ids.append(arg2_start_pos)
        arg2_end_ids.append(arg2_end_pos)
        fine_spans.append(span)
        fine_rels.append(rel)
    res = (
        whole_document, sent_start_ids, sent_end_ids, arg1_start_ids,
        arg1_end_ids, arg2_start_ids, arg2_end_ids, fine_rels, fine_spans
    )
    return res


def inter_sentential_rels(sent_num, rels, spans):
    contained_rel_dict = {}
    for rel, span in zip(rels, spans):
        sent_id_1 = span[0]
        sent_id_2 = span[1]
        if sent_id_1 != sent_id_2:
            contained_rel_dict["{}+{}".format(sent_id_1, sent_id_2)] = rel
    sent_rels = []
    for sent_id in range(sent_num - 1):
        id_query = "{}+{}".format(sent_id, sent_id + 1)
        if id_query in contained_rel_dict:
            sent_rels.append(contained_rel_dict[id_query])
        else:
            # print("++")
            sent_rels.append("NoRel")

    return sent_rels


def flatten_emb(emb, emb_mask):
    batch_size = emb.size(0)
    seq_length = emb.size(1)
    flat_emb = emb.view(batch_size * seq_length, -1)
    flat_emb_mask = emb_mask.view(batch_size * seq_length)

    return flat_emb[flat_emb_mask.nonzero().squeeze(), :]


def get_span_representation(span_starts, span_ends, hidden_states, attention_mask):
    """
    refer to: https://github.com/huminghao16/SpanABSA/blob/master/bert/sentiment_modeling.py
    N: batch size
    M: span number
    L: seq length
    D: hidden size
    Args:
        span_starts: [N, M]
        span_ends: [N, M]
        hidden_states: [N, L, D]
        attention_mask: [N, L]
    """
    N = hidden_states.size(0)
    M = span_starts.size(1)
    input_len = torch.sum(attention_mask, dim=-1).to(hidden_states.device)  # [N]
    span_offset = torch.cumsum(input_len, dim=0).to(hidden_states.device)  # [N]
    span_offset -= input_len

    span_starts_offset = span_starts + span_offset.unsqueeze(1)
    span_ends_offset = span_ends + span_offset.unsqueeze(1)
    span_starts_offset = span_starts_offset.view(-1)
    span_ends_offset = span_ends_offset.view(-1)
    span_width = span_ends_offset - span_starts_offset
    max_span_width = torch.max(span_width)

    flat_hidden_states = flatten_emb(hidden_states, attention_mask)  # [<N*L, D], because exclude zero position
    flat_length = flat_hidden_states.size(0)

    # [N*M, max_span_width]
    span_indices = torch.arange(max_span_width).unsqueeze(0).to(hidden_states.device) + span_starts_offset.unsqueeze(1)
    span_indices = torch.min(span_indices, (flat_length - 1) * torch.ones_like(span_indices))  # in case out of boundary
    span_vectors = flat_hidden_states[span_indices, :]  # [N*M, max_span_width, D]
    span_mask = torch.arange(max_span_width).to(hidden_states.device)
    span_mask = span_mask < span_width.unsqueeze(-1)  # [N*M, max_span_width]

    # average the word states to get span representation
    expanded_span_mask = span_mask.unsqueeze(2)  # [N*M, max_span_width, 1]
    masked_span_vectors = span_vectors * expanded_span_mask  # [N*M, max_span_width, D]
    avg_span_vectors = torch.sum(masked_span_vectors, dim=1) / span_width.unsqueeze(1)  # [N*M, D]
    avg_span_vectors = avg_span_vectors.view(N, M, -1)

    return avg_span_vectors


class SentDataset(Dataset):
    def __init__(self, file_name, params):
        self.max_text_length = params["max_text_length"]
        self.max_arg_num = params["max_arg_num"]
        self.max_sent_num = params["max_sent_num"]
        self.rel_list = params["rel_list"]
        self.label_list = params["label_list"]
        self.tokenizer = params["tokenizer"]
        self.encoder = params["encoder"]
        self.pooler_type = params["pooler_type"]
        assert self.pooler_type in ["avg"], (self.pooler_type)

        mode = file_name.split("/")[-1].split(".")[0]
        dir_name = os.path.dirname(file_name)
        new_dir_name = os.path.join(dir_name, "fast_sent_vectors")
        os.makedirs(new_dir_name, exist_ok=True)
        if "xlnet" in self.encoder.__class__.__name__.lower():
            saved_file_name = "fast_{}_{}_for_xlnet_vector_{}-{}-{}.pkl".format(
                self.pooler_type, mode, self.max_text_length, self.max_sent_num, self.max_arg_num
            )
        elif "roberta" in self.encoder.__class__.__name__.lower():
            # saved_file_name = "fast_{}_{}_for_roberta_vector_{}-{}-{}.pkl".format(
            #     self.pooler_type, mode, self.max_text_length, self.max_sent_num, self.max_arg_num
            # )
            saved_file_name = "mask_fast_{}_{}_for_roberta_vector_{}-{}-{}.pkl".format(
                self.pooler_type, mode, self.max_text_length, self.max_sent_num, self.max_arg_num
            )
        elif self.encoder is None or "llama" in self.encoder.__class__.__name__.lower():
            saved_file_name = "fast_{}_{}_for_llama_vector_{}-{}-{}.pkl".format(
                self.pooler_type, mode, self.max_text_length, self.max_sent_num, self.max_arg_num
            )
        self.np_file = os.path.join(new_dir_name, saved_file_name)

        self.init_dataset(file_name)

    def init_dataset(self, file_name):
        if os.path.exists(self.np_file):
            with open(self.np_file, "rb") as f:
                results = pickle.load(f)
                all_doc_vectors = results[0]
                all_sent_vectors = results[1]
                all_sent_rel_ids = results[2]
                all_sent_mask = results[3]
                all_arg1_vectors = results[4]
                all_arg2_vectors = results[5]
                all_rel_ids = results[6]
                all_arg_mask = results[7]
                all_label_ids = results[8]
                all_sent_rel_mask = results[9]
        else:
            with open(file_name, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # 1. prepare ids
            all_input_ids = []
            all_attention_mask = []
            all_token_type_ids = []
            all_sent_start_ids = []
            all_sent_end_ids = []
            all_sent_mask = []
            all_sent_rel_ids = []
            all_arg1_start_ids = []
            all_arg1_end_ids = []
            all_arg2_start_ids = []
            all_arg2_end_ids = []
            all_arg_mask = []
            all_rel_ids = []
            all_label_ids = []

            all_sent_rel_mask = []

            for line in lines:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    sents = sample["sents"]
                    spans = sample["spans"]
                    rels = sample["rels"]
                    label = sample["score"]
                    if len(sents) == 0:
                        continue

                    processed_res = merge_sents(sents, spans, rels, self.tokenizer)
                    text = processed_res[0]
                    sent_start_ids, sent_end_ids = processed_res[1], processed_res[2]
                    arg1_start_ids, arg1_end_ids = processed_res[3], processed_res[4]
                    arg2_start_ids, arg2_end_ids = processed_res[5], processed_res[6]
                    rels, spans = processed_res[7], processed_res[8]
                    sent_rels = inter_sentential_rels(len(sent_start_ids), rels, spans)

                    # 2.1 whole document
                    doc_res = self.tokenizer(
                        text=text,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_text_length,
                        return_tensors="pt"
                    )
                    input_ids = doc_res.input_ids
                    attention_mask = doc_res.attention_mask
                    if "token_type_ids" in doc_res:
                        token_type_ids = doc_res["token_type_ids"]
                    else:
                        token_type_ids = torch.zeros_like(attention_mask)

                    # 2.2 sent
                    sent_rel_ids = [self.rel_list.index(rel) for rel in sent_rels]
                    np_sent_rel_ids = np.zeros(self.max_sent_num, dtype=np.int32)
                    sent_mask = np.zeros(self.max_sent_num, dtype=np.int32)
                    np_sent_start_ids = np.zeros(self.max_sent_num, dtype=np.int32)
                    np_sent_end_ids = np.ones(self.max_sent_num, dtype=np.int32)
                    real_sent_num = 0
                    if len(sents) > self.max_sent_num:
                        np_sent_rel_ids[:self.max_sent_num - 1] = sent_rel_ids[:self.max_sent_num - 1]
                        sent_mask[:self.max_sent_num] = 1
                        np_sent_start_ids[:self.max_sent_num] = sent_start_ids[:self.max_sent_num]
                        np_sent_end_ids[:self.max_sent_num] = sent_end_ids[:self.max_sent_num]
                        real_sent_num = self.max_sent_num
                    else:
                        np_sent_rel_ids[:len(sent_rels)] = sent_rel_ids
                        sent_mask[:len(sents)] = 1
                        np_sent_start_ids[:len(sents)] = sent_start_ids
                        np_sent_end_ids[:len(sents)] = sent_end_ids
                        real_sent_num = len(sents)

                    ## for the merged sent_rel mask
                    sent_sent = np.expand_dims(sent_mask, axis=0).repeat(self.max_sent_num, axis=0)
                    sent_rel = np.zeros((self.max_sent_num, self.max_sent_num), dtype=np.int32)
                    rel_sent = np.zeros((self.max_sent_num, self.max_sent_num), dtype=np.int32)
                    rel_rel = np.zeros((self.max_sent_num, self.max_sent_num), dtype=np.int32)
                    for idyy in range(real_sent_num - 1):
                        sent_rel[idyy][idyy] = 1
                        sent_rel[idyy + 1][idyy] = 1
                        rel_sent[idyy][idyy] = 1
                        rel_sent[idyy][idyy + 1] = 1
                        # rel_rel[idyy][idyy] = 1
                        rel_rel[idyy][:real_sent_num - 1] = 1
                    fusion_sent_mask = np.concatenate((sent_sent, sent_rel), axis=1)  # [sent_len, sent_len*2]
                    fusion_rel_mask = np.concatenate((rel_sent, rel_rel), axis=1)
                    sent_rel_mask = np.concatenate((fusion_sent_mask, fusion_rel_mask),
                                                   axis=0)  # [sent_len*2, sent_len*2]

                    # 2.3 rel and arg
                    rel_ids = [self.rel_list.index(rel) for rel in rels]
                    np_rel_ids = np.zeros(self.max_arg_num, dtype=np.int32)
                    arg_mask = np.zeros(self.max_arg_num, dtype=np.int32)
                    np_arg1_start_ids = np.zeros(self.max_arg_num, dtype=np.int32)
                    np_arg2_start_ids = np.zeros(self.max_arg_num, dtype=np.int32)
                    np_arg1_end_ids = np.ones(self.max_arg_num, dtype=np.int32)
                    np_arg2_end_ids = np.ones(self.max_arg_num, dtype=np.int32)
                    if len(rels) > self.max_arg_num:
                        arg_mask[:self.max_arg_num] = 1
                        np_rel_ids[:self.max_arg_num] = rel_ids[:self.max_arg_num]
                        np_arg1_start_ids[:self.max_arg_num] = arg1_start_ids[:self.max_arg_num]
                        np_arg1_end_ids[:self.max_arg_num] = arg1_end_ids[:self.max_arg_num]
                        np_arg2_start_ids[:self.max_arg_num] = arg2_start_ids[:self.max_arg_num]
                        np_arg2_end_ids[:self.max_arg_num] = arg2_end_ids[self.max_arg_num]
                    else:
                        arg_mask[:len(rels)] = 1
                        np_rel_ids[:len(rels)] = rel_ids
                        np_arg1_start_ids[:len(rels)] = arg1_start_ids
                        np_arg1_end_ids[:len(rels)] = arg1_end_ids
                        np_arg2_start_ids[:len(rels)] = arg2_start_ids
                        np_arg2_end_ids[:len(rels)] = arg2_end_ids

                    # labels
                    label_id = self.label_list.index(label)

                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_token_type_ids.append(token_type_ids)
                    all_sent_start_ids.append(torch.tensor(np_sent_start_ids).unsqueeze(0))
                    all_sent_end_ids.append(torch.tensor(np_sent_end_ids).unsqueeze(0))
                    all_sent_rel_ids.append(np_sent_rel_ids)
                    all_sent_mask.append(sent_mask)
                    all_arg1_start_ids.append(torch.tensor(np_arg1_start_ids).unsqueeze(0))
                    all_arg1_end_ids.append(torch.tensor(np_arg1_end_ids).unsqueeze(0))
                    all_arg2_start_ids.append(torch.tensor(np_arg2_start_ids).unsqueeze(0))
                    all_arg2_end_ids.append(torch.tensor(np_arg2_end_ids).unsqueeze(0))
                    all_rel_ids.append(np_rel_ids)
                    all_arg_mask.append(arg_mask)
                    all_label_ids.append(label_id)

                    all_sent_rel_mask.append(sent_rel_mask)

            # 2. generate vectors
            all_doc_vectors = []
            all_sent_vectors = []
            all_arg1_vectors = []
            all_arg2_vectors = []
            batch_size = 8
            batch_steps = len(all_input_ids) // batch_size + 1
            batch_iter = trange(0, batch_steps, desc="Step")
            for cur_idx in batch_iter:
                start_pos = cur_idx * batch_size
                end_pos = (cur_idx + 1) * batch_size
                if end_pos > len(all_input_ids):
                    end_pos = len(all_input_ids)
                if start_pos == end_pos:
                    break
                cur_batch_size = end_pos - start_pos
                batch_input_ids = torch.cat(all_input_ids[start_pos:end_pos], dim=0).to(self.encoder.device)
                batch_attention_mask = torch.cat(all_attention_mask[start_pos:end_pos], dim=0).to(self.encoder.device)
                batch_segment_ids = torch.cat(all_token_type_ids[start_pos:end_pos], dim=0).to(self.encoder.device)
                batch_sent_start_ids = torch.cat(all_sent_start_ids[start_pos:end_pos], dim=0).to(self.encoder.device)
                batch_sent_end_ids = torch.cat(all_sent_end_ids[start_pos:end_pos], dim=0).to(self.encoder.device)
                batch_arg1_start_ids = torch.cat(all_arg1_start_ids[start_pos:end_pos], dim=0).to(self.encoder.device)
                batch_arg1_end_ids = torch.cat(all_arg1_end_ids[start_pos:end_pos], dim=0).to(self.encoder.device)
                batch_arg2_start_ids = torch.cat(all_arg2_start_ids[start_pos:end_pos], dim=0).to(self.encoder.device)
                batch_arg2_end_ids = torch.cat(all_arg2_end_ids[start_pos:end_pos], dim=0).to(self.encoder.device)
                doc_inputs = {
                    "input_ids": batch_input_ids,
                    "attention_mask": batch_attention_mask,
                    # "token_type_ids": batch_segment_ids
                }
                with torch.no_grad():
                    doc_outputs = self.encoder(**doc_inputs)
                    seq_doc_outputs = doc_outputs.last_hidden_state
                    batch_doc_length = torch.sum(batch_attention_mask, dim=-1)
                    batch_sum_doc_reps = seq_doc_outputs * batch_attention_mask.unsqueeze(2)
                    batch_sum_doc_reps = torch.sum(batch_sum_doc_reps, dim=1)
                    doc_vectors = batch_sum_doc_reps / batch_doc_length.unsqueeze(1)

                    ## sent vectors
                    # print("## sent ##")
                    sent_vectors = get_span_representation(
                        span_starts=batch_sent_start_ids,
                        span_ends=batch_sent_end_ids,
                        hidden_states=seq_doc_outputs,
                        attention_mask=batch_attention_mask
                    )

                    ## arg1_vectors and arg2_vectors
                    # print("## arg1 ##")
                    arg1_vectors = get_span_representation(
                        span_starts=batch_arg1_start_ids,
                        span_ends=batch_arg1_end_ids,
                        hidden_states=seq_doc_outputs,
                        attention_mask=batch_attention_mask
                    )
                    # print("## arg2 ##")
                    arg2_vectors = get_span_representation(
                        span_starts=batch_arg2_start_ids,
                        span_ends=batch_arg2_end_ids,
                        hidden_states=seq_doc_outputs,
                        attention_mask=batch_attention_mask
                    )

                    doc_vectors = doc_vectors.detach().cpu()
                    sent_vectors = sent_vectors.detach().cpu()
                    arg1_vectors = arg1_vectors.detach().cpu()
                    arg2_vectors = arg2_vectors.detach().cpu()
                    all_doc_vectors.append(doc_vectors)
                    all_sent_vectors.append(sent_vectors)
                    all_arg1_vectors.append(arg1_vectors)
                    all_arg2_vectors.append(arg2_vectors)

            all_doc_vectors = torch.cat(all_doc_vectors, dim=0)
            all_sent_vectors = torch.cat(all_sent_vectors, dim=0)
            all_arg1_vectors = torch.cat(all_arg1_vectors, dim=0)
            all_arg2_vectors = torch.cat(all_arg2_vectors, dim=0)
            # """
            with open(self.np_file, "wb") as f:
                pickle.dump(
                    [all_doc_vectors, all_sent_vectors, all_sent_rel_ids,
                     all_sent_mask, all_arg1_vectors, all_arg2_vectors,
                     all_rel_ids, all_arg_mask, all_label_ids, all_sent_rel_mask], f
                )
            # """

        self.doc_vectors = all_doc_vectors
        self.sent_vectors = all_sent_vectors
        self.sent_rel_ids = all_sent_rel_ids
        self.sent_mask = all_sent_mask
        self.arg1_vectors = all_arg1_vectors
        self.arg2_vectors = all_arg2_vectors
        self.rel_ids = all_rel_ids
        self.arg_mask = all_arg_mask
        self.label_ids = all_label_ids
        self.sent_rel_mask = all_sent_rel_mask
        self.total_size = len(all_doc_vectors)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        return (
            self.doc_vectors[index],
            self.sent_vectors[index],
            torch.tensor(self.sent_rel_ids[index]),
            torch.tensor(self.sent_mask[index]),
            self.arg1_vectors[index],
            self.arg2_vectors[index],
            torch.tensor(self.rel_ids[index]),
            torch.tensor(self.arg_mask[index]),
            torch.tensor(self.label_ids[index]),
            torch.tensor(self.sent_rel_mask[index])
        )
