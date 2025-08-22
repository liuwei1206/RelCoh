# date=2024-04-24
# author = liuwei

import json
import os
from collections import defaultdict
import numpy as np
import random
from scipy import stats
import math

random.seed(106524)
np.random.seed(106524)


def stat_of_ngram(dataset, n=1):
    """the statistics of ngram distribution in the dataset"""

    def filter_intra_rels(rels, spans):
        new_rels = []
        new_spans = []
        for rel, span in zip(rels, spans):
            if span[0] == span[1]:
                continue
            new_rels.append(rel)
            new_spans.append(span)

        return new_rels, new_spans

    distribution = defaultdict(int)
    total_num = 0

    ## 1. data preparation
    data_file = "data/dataset/{}/1/train.json".format(dataset)
    with open(data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)

                rels = sample["rels"]
                spans = sample["spans"]
                rels, spans = filter_intra_rels(rels, spans)

                length = len(rels) - n + 1
                for idx in range(length):
                    rel_ngram = []
                    for idy in range(n):
                        rel_ngram.append(rels[idx + idy])
                    rel_ngram = " -> ".join(rel_ngram)
                    distribution[rel_ngram] += 1
                    total_num += 1

    print("#### %s, n=%d ####" % (dataset, n))
    distribution = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    for item in distribution:
        print(" %s: %d, %.4f" % (item[0], item[1], item[1] / total_num))
    print()


def coef_analysis(dataset, n=2):
    """
    Args:
        dataset: which corpus to analyze
        n: n-gram rels
    Return:
        coef:
        p-value:
    """

    def filter_intra_rels(rels, spans):
        new_rels = []
        new_spans = []
        for rel, span in zip(rels, spans):
            if span[0] == span[1]:
                continue
            new_rels.append(rel)
            new_spans.append(span)

        return new_rels, new_spans

    ## 1. data preparation
    data_file = "data/dataset/{}/1/train.json".format(dataset)
    all_samples = []
    sample_size = 0

    with open(data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample_size += 1
                sample = json.loads(line)
                all_samples.append(sample)

    ## 2. ngram vocab
    rel_freq = defaultdict(int)
    total_ngram_size = 0
    for sample in all_samples:
        rels = sample["rels"]
        spans = sample["spans"]
        rels, spans = filter_intra_rels(rels, spans)

        length = len(rels) - n + 1
        for idx in range(length):
            rel_ngram = []
            for idy in range(n):
                rel = rels[idx + idy]
                rel_type = spans[idx + idy][2]
                # rel = "{}-{}".format(rel_type, rel)
                rel_ngram.append(rel)
            rel_ngram = " -> ".join(rel_ngram)
            rel_freq[rel_ngram] += 1
            total_ngram_size += 1
    ngram_vocab = []
    rel_freq = sorted(rel_freq.items(), key=lambda x: x[1], reverse=True)
    for item in rel_freq:
        ngram_vocab.append(item[0])

    ## 3. calculate coef, and return the top 4
    # with intra relations
    # gcdc_setting = {1: 72, 2: 45, 3: 32, 4: 24}
    # toefl_setting = {1: 120, 2: 68, 3: 47, 4: 36}
    # without intra relations
    gcdc_setting = {1: 72, 2: 40, 3: 33, 4: 21}
    toefl_setting = {1: 120, 2: 56, 3: 45, 4: 36}
    if "toefl" in dataset.lower():
        min_freq = toefl_setting[n]
    else:
        min_freq = gcdc_setting[n]
    ngrams_coef = []
    # cal the coef for each ngram
    for ngram in ngram_vocab[:100]:
        label_freq_list = []
        label_dict = {"low": 1, "medium": 2, "high": 3, "1": 1, "2": 2, "3": 3}
        balance_size = 4  # to release the uneven bias
        ngram_freq_group = {}

        for sample in all_samples:
            score = sample["score"].strip().lower()
            count = 0
            rels = sample["rels"]
            spans = sample["spans"]
            rels, spans = filter_intra_rels(rels, spans)

            length = len(rels) - n + 1
            for idx in range(length):
                tmp_ngram = []
                for idy in range(n):
                    rel = rels[idx + idy]
                    rel_type = spans[idx + idy][2]
                    # rel = "{}-{}".format(rel_type, rel)
                    tmp_ngram.append(rel)
                tmp_ngram = " -> ".join(tmp_ngram)
                if tmp_ngram == ngram:
                    count += 1

            if count > 0:
                if score in ngram_freq_group:
                    ngram_freq_group[score].append((label_dict[score], count))
                else:
                    ngram_freq_group[score] = [(label_dict[score], count)]

        min_val = 10000
        min_cls = ""
        sampled_pairs = []
        for key in ngram_freq_group:
            if min_val > len(ngram_freq_group[key]):
                min_val = len(ngram_freq_group[key])
                min_cls = key
            sampled_pairs.extend(ngram_freq_group[key])

        for idx in range(balance_size):
            sampled_pairs.append((label_dict[min_cls], 0))

            # spearman's rank correlation, because both values are sorted value
        label_list = np.array([item[0] for item in sampled_pairs])
        freq_list = np.array([item[1] for item in sampled_pairs])
        coef, p = stats.spearmanr(label_list, freq_list)

        if not math.isnan(coef) and len(label_list) >= min_freq:
            ngrams_coef.append((ngram, coef, p, len(label_list)))

    ngrams_coef = sorted(ngrams_coef, key=lambda x: abs(x[1]), reverse=True)
    for item in ngrams_coef[:6]:
        print(" #### {} : {} ####".format(item[0], item[3]))
        print("  coef=%.4f, p-value=%.4f\n" % (item[1], item[2]))


def search_example(dataset, n, pattern):
    """search example with specific relation pattern"""

    def filter_intra_rels(rels, spans):
        new_rels = []
        new_spans = []
        for rel, span in zip(rels, spans):
            if span[0] == span[1]:
                continue
            new_rels.append(rel)
            new_spans.append(span)

        return new_rels, new_spans

    ## 1. data preparation
    data_file = "data/dataset/{}/1/train.json".format(dataset)
    report_size = 10
    report_id = 0
    with open(data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)

                rels = sample["rels"]
                spans = sample["spans"]
                sents = sample["sents"]
                score = sample["score"]
                rels, spans = filter_intra_rels(rels, spans)

                length = len(rels) - n + 1
                for idx in range(length):
                    rel_ngram = []
                    sent_ids = []
                    for idy in range(n):
                        rel_ngram.append(rels[idx + idy])
                        sent_ids.append((spans[idx + idy][0], spans[idx + idy][1]))
                    rel_ngram = " -> ".join(rel_ngram)
                    if rel_ngram == pattern:
                        report_id += 1
                        print("#### Example %d: %s ####" % (report_id, score))
                        for span in sent_ids:
                            print("%s +++ %s" % (sents[span[0]], sents[span[1]]))
                if report_id >= report_size:
                    break


if __name__ == "__main__":
    dataset = "toefl_p1"
    dataset = "enron"
    n = 1
    # coef_analysis(dataset, n)

    stat_of_ngram(dataset, n)

    pattern = "cause -> asynchronous"
    pattern = "cause -> norel -> conjunction"
    pattern = "cause -> level-of-detail -> conjunction"
    # search_example(dataset, n, pattern)
