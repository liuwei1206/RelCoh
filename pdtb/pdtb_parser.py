# author = liuwei
# date = 2023-12-30

import logging
import os
import json
import pickle
import math
import random
import time
import datetime
from tqdm import tqdm, trange

import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers.models.roberta import RobertaConfig, RobertaTokenizer
import stanza

from pdtb_model import BaseClassifier
logging.disable(logging.WARNING)
from pdtb_utils import labels_from_file, is_exp_inter, is_exp_intra, split_into_sentences, pack_batch_data, split_into_sentences_stanza

# set logger, print to console and write to file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()  # 输出到控制台的handler
chlr.setFormatter(formatter)
logger.addHandler(chlr)

def get_argparse():
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument("--data_dir", default="data/dataset", type=str)
    parser.add_argument("--dataset", default="toefl", type=str, help="gcdc, toefl")
    parser.add_argument("--fold_id", default=-1, type=int, help="1-5 for gcdc, 1-8 for toefl")
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str, help="only large")
    parser.add_argument("--label_level", default=1, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)

    return parser


def parse_doc(input_doc, parse_params):
    """
    1. split the doc into sentences
    2. judge if connectives exist between sentences
    3. if yes, then use explicit model, else use implicit model
    4. save the results into doc

    requires:
        stanza,
        connective_list

    Args:
        input_doc:
        explicit_model:
        implicit_model:
        stanza_nlp:
        conn_list:
    """
    tokenizer = parse_params["tokenizer"]
    exp_model = parse_params["exp_model"]
    imp_model = parse_params["imp_model"]
    conn_list = parse_params["conn_list"]
    exp_rel_list = parse_params["exp_rel_list"]
    imp_rel_list = parse_params["imp_rel_list"]
    use_stanza = parse_params["use_stanza"]

    ## 1. split doc into sents, and judge if sent has connectives or not
    if use_stanza:
        stanza_dir = "/hits/basement/nlp/liuwi/resources/stanza_resources"
        stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize', dir=stanza_dir, download_method=None)
        doc_sents = split_into_sentences_stanza(input_doc, stanza_nlp)
    else:
        doc_sents = split_into_sentences(input_doc)
    sent_num = len(doc_sents)
    all_exp_items = []
    all_imp_items = []
    for idx in range(sent_num):
        # intra-sentence explicit relation
        has_conn, arg_pair = is_exp_intra(doc_sents[idx], conn_list=conn_list)
        if has_conn:
            all_exp_items.append((arg_pair[0], arg_pair[1], idx, idx, arg_pair[2], arg_pair[3]))

        # inter-sentence explicit relation
        if idx < sent_num-1:
            has_conn, _ = is_exp_inter(doc_sents[idx], doc_sents[idx+1], conn_list=conn_list)
            if has_conn:
                all_exp_items.append((
                    doc_sents[idx], doc_sents[idx+1], idx, idx+1,
                    (0, len(doc_sents[idx])), (0, len(doc_sents[idx+1]))
                ))
            else:
                all_imp_items.append((
                    doc_sents[idx], doc_sents[idx+1], idx, idx+1,
                    (0, len(doc_sents[idx])), (0, len(doc_sents[idx+1]))
                ))

    ## 2. parse both explicit and implicit
    pred_rels = []
    if len(all_exp_items) > 0:
        batch_data = pack_batch_data(all_exp_items, tokenizer, max_length=256)
        batch_data = tuple(t.to(exp_model.device) for t in batch_data)
        inputs = {
            "input_ids": batch_data[0],
            "attention_mask": batch_data[1],
            "token_type_ids": batch_data[2],
            "flag": "Eval"
        }
        with torch.no_grad():
            outputs = exp_model(**inputs)
            exp_pred_ids = list(outputs[0])
        # recover the position of relations
        assert len(all_exp_items) == len(exp_pred_ids), (len(all_exp_items), len(exp_pred_ids))
        for item, rel_id in zip(all_exp_items, exp_pred_ids):
            rel = exp_rel_list[rel_id]
            arg1_pos = item[2]
            arg2_pos = item[3]
            arg1_span = item[4]
            arg2_span = item[5]
            pred_rels.append((rel, arg1_pos, arg2_pos, "exp", arg1_span, arg2_span))
    if len(all_imp_items) > 0:
        batch_data = pack_batch_data(all_imp_items, tokenizer, max_length=256)
        batch_data = tuple(t.to(imp_model.device) for t in batch_data)
        inputs = {
            "input_ids": batch_data[0],
            "attention_mask": batch_data[1],
            "token_type_ids": batch_data[2],
            "flag": "Eval"
        }
        with torch.no_grad():
            outputs = imp_model(**inputs)
            imp_pred_ids = list(outputs[0])
        assert len(all_imp_items) == len(imp_pred_ids), (len(all_imp_items), len(imp_pred_ids))
        for item, rel_id in zip(all_imp_items, imp_pred_ids):
            rel = imp_rel_list[rel_id]
            arg1_pos = item[2]
            arg2_pos = item[3]
            arg1_span = item[4]
            arg2_span = item[5]
            pred_rels.append((rel, arg1_pos, arg2_pos, "imp", arg1_span, arg2_span))

    ## 3. sorted and merge relation
    pred_rels = sorted(pred_rels, key=lambda x:(x[1], x[2]))
    all_rels = []
    all_rel_spans = []
    for item in pred_rels:
        all_rels.append(item[0])
        all_rel_spans.append((item[1], item[2], item[3], item[4], item[5]))

    return doc_sents, all_rels, all_rel_spans


def parse_file(in_file, parse_params, out_file):
    ## 1. read out raw texts
    all_docs = []
    all_labels = []
    with open(in_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)

                text = sample["text"]
                label = sample["score"]
                all_docs.append(text)
                all_labels.append(label)

    ## 2. parse docs
    all_parsed_docs = []
    doc_num = len(all_docs)
    doc_iter = trange(1, doc_num+1, desc="Num")
    for doc_idx in doc_iter:
        doc, label = all_docs[doc_idx-1], all_labels[doc_idx-1]
        doc_sents, doc_rels, doc_rel_spans = parse_doc(doc, parse_params)
        all_parsed_docs.append((label, doc, doc_sents, doc_rels, doc_rel_spans))

    ## 3. write into file
    with open(out_file, "w", encoding="utf-8") as f:
        for item in all_parsed_docs:
            sample = {}
            sample["score"] = item[0]
            sample["text"] = item[1]
            sample["sents"] = item[2]
            sample["rels"] = item[3]
            sample["spans"] = item[4]

            f.write("%s\n"%(json.dumps(sample, ensure_ascii=False)))

def parse():
    args = get_argparse().parse_args()
    if torch.cuda.is_available():
        args.n_gpu = 1
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        args.n_gpu = 0
    args.device = device
    print("Training/evaluation parameters %s", args)

    ## 1. init model
    parser_dir = "data/parser/pdtb3"
    args.model_name_or_path = os.path.join(
        "/hits/basement/nlp/liuwi/resources/pretrained_models",
        args.model_name_or_path
    )
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.HP_dropout = 0.1
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)

    # 1.1 explicit model
    exp_dir = os.path.join(parser_dir, "exp")
    exp_dir = os.path.join(exp_dir, "l{}".format(args.label_level))
    conn_list = labels_from_file(os.path.join(exp_dir, "conn_list.txt"))
    exp_rel_list = labels_from_file(os.path.join(exp_dir, "labels.txt"))
    args.num_labels = len(exp_rel_list)
    exp_model = BaseClassifier(config=config, args=args)
    exp_model = exp_model.to(args.device)
    exp_checkpoint_file = os.path.join(exp_dir, "pytorch_model.bin")
    exp_model.load_state_dict(torch.load(exp_checkpoint_file))

    # 1.2 implicit model
    imp_dir = os.path.join(parser_dir, "imp")
    imp_dir = os.path.join(imp_dir, "l{}".format(args.label_level))
    imp_rel_list = labels_from_file(os.path.join(imp_dir, "labels.txt"))
    args.num_labels = len(imp_rel_list)
    imp_model = BaseClassifier(config=config, args=args)
    imp_model = imp_model.to(args.device)
    imp_checkpoint_file = os.path.join(imp_dir, "pytorch_model.bin")
    imp_model.load_state_dict(torch.load(imp_checkpoint_file))

    ## 2. read raw data
    data_dir = os.path.join(args.data_dir, args.dataset)
    data_dir = os.path.join(data_dir, str(args.fold_id))
    train_data_file = os.path.join(data_dir, "train.json")
    dev_data_file = os.path.join(data_dir, "dev.json")
    test_data_file = os.path.join(data_dir, "test.json")
    parsed_dir = os.path.join(data_dir, "parsed_l{}".format(args.label_level))
    os.makedirs(parsed_dir, exist_ok=True)
    train_output_file = os.path.join(parsed_dir, "parsed_train.json")
    dev_output_file = os.path.join(parsed_dir, "parsed_dev.json")
    test_output_file = os.path.join(parsed_dir, "parsed_test.json")

    ## 3. parse file
    parse_params = {
        "tokenizer": tokenizer,
        "exp_model": exp_model,
        "imp_model": imp_model,
        "conn_list": conn_list,
        "exp_rel_list": exp_rel_list,
        "imp_rel_list": imp_rel_list,
        "use_stanza": False
    }
    parse_file(train_data_file, parse_params, train_output_file)
    parse_file(dev_data_file, parse_params, dev_output_file)
    parse_file(test_data_file, parse_params, test_output_file)

if __name__ == "__main__":
    parse()

