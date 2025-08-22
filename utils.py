
import os
import json
import random

import nltk
import numpy as np
import pickle
import torch

random.seed(106524)
np.random.seed(106524)

def labels_from_file(label_file):
    label_list = []
    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                label_list.append(line.strip().lower())

    return label_list


def build_embedding_of_corpus(embed_file, vocab, embed_dim, saved_file):
    print(saved_file)
    if os.path.exists(saved_file):
        with open(saved_file, "rb") as f:
            corpus_embed = pickle.load(f)
    else:
        word2vec = {}
        matched_num = 0
        with open(embed_file, "r", encoding="utf-8") as f:
            idx = 0
            for line in f:
                line = line.strip()
                if line:
                    item = line.split()
                    if len(item) != (embed_dim+1):
                        continue
                    word = item[0]
                    vector = item[1:]
                    if word in vocab:
                        word2vec[word] = np.array(vector, dtype=np.float)
                        matched_num += 1
                        if matched_num / len(vocab) >= 0.99:
                            break
                    idx += 1
                    if idx % 21800 == 0:
                        print("loading per%d"%(idx / 21800))

        corpus_embed = np.empty([len(vocab), embed_dim])
        scale = np.sqrt(3.0 / embed_dim)
        num_matched = 0
        num_non_matched = 0
        missing_words = []
        for idx, word in enumerate(vocab):
            if word in word2vec:
                corpus_embed[idx, :] = word2vec[word]
                num_matched += 1
            elif word.lower() in word2vec:
                corpus_embed[idx, :] = word2vec[word.lower()]
                num_matched += 1
            else:
                corpus_embed[idx, :] = np.random.uniform(-scale, scale, [1, embed_dim])
                num_non_matched += 1
                missing_words.append(word)
        print("total: %d, matched: %d, non-matched: %d"%(len(vocab), num_matched, num_non_matched))
        with open(saved_file, "wb") as f:
            pickle.dump(corpus_embed, f)

    return corpus_embed


def random_embedding_of_corpus(vocab, embed_dim):
    corpus_embed = np.empty([len(vocab), embed_dim])
    print("vocab: ", len(vocab))
    scale = np.sqrt(3.0 / embed_dim)
    for idx, word in enumerate(vocab):
        corpus_embed[idx, :] = np.random.uniform(-scale, scale, [1, embed_dim])
    return corpus_embed


def reverse_padded_sequence(inputs, lengths, batch_first=True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    max_length, batch_size = inputs.size(0), inputs.size(1)
    if len(lengths) != batch_size:
        raise ValueError("inputs is incompatible with lengths.")
    ind = [list(reversed(range(0, length))) + list(range(length, max_length)) for length in lengths]
    ind = torch.LongTensor(ind).transpose(0, 1)
    for dim in range(2, inputs.dim()):
        ind = ind.unsqueeze(dim)
    ind = ind.expand_as(inputs)
    if inputs.is_cuda:
        ind = ind.cuda()
    reversed_inputs = torch.gather(inputs, 0, ind)
    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)

    return reversed_inputs

def get_vocab_from_corpus(data_dir, item_key="text"):
    """
    Args:
        data_dir: the data dir, contain train.json, dev.json, test.json
        item_key: the key in sample's dict
        require_unk: require a unk token or not
    Returns:
        vocab: a list of words
    """
    assert item_key in ["text", "rels"], ("##", item_key, "##")
    vocab = set()
    for file_name in ["train.json", "dev.json", "test.json"]:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    text = sample[item_key]
                    if item_key == "rels":
                        tokens = text
                    else:
                        tokens = nltk.word_tokenize(text)
                    for token in tokens:
                        vocab.add(token)
    vocab = list(vocab)
    vocab = sorted(vocab)
    vocab.insert(0, "<pad>")
    vocab.insert(1, "<unk>")

    return vocab