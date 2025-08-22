# author = liuwei
# date = 2023-12-30

import os
import json
import re
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import torch
import math
import random
import nltk
from collections import defaultdict
from conn_filter import filter_so, filter_or, filter_as, filter_and, filter_after
from conn_filter import filter_because, filter_before, filter_but, filter_if
from conn_filter import filter_when, filter_since, filter_then
random.seed(106524)

def labels_from_file(label_file):
    label_list = []
    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                label_list.append(line.strip().lower())

    return label_list


def is_exp_intra(sent, conn_list):
    sent_words = nltk.word_tokenize(sent)
    special_conns = {
        "although": (True, True, False), "even if": (True, True, False),
        "even though": (True, True, False), "if": (True, True, False),
        "instead": (False, False, True), "later": (False, True, True),
        "though": (False, True, True), "yet": (True, False, True),
    }
    filter_dict = {
        "so": filter_so, "or": filter_or, "as": filter_as,
        "and": filter_and, "after": filter_after, "because": filter_because,
        "before": filter_before, "but": filter_but, "if": filter_if,
        "when": filter_when, "since": filter_since, "then": filter_then
    }
    for conn in conn_list:
        if ".." in conn:
            items = conn.split("..")
            if items[0].lower() in sent_words and items[1].lower() in sent_words:
                if "," in sent:
                    comma_pos = sent.find(",")  # the first one
                    arg1 = sent[:comma_pos + 1].strip()
                    arg2 = sent[comma_pos + 1:].strip()
                    return True, (arg1, arg2, (0, comma_pos+1), (comma_pos+1, len(sent)))
        if conn in special_conns:
            start_sent, in_sent, end_sent = special_conns[conn]
            has_conn, args = find_conn_within_sent(sent, conn, start_sent, in_sent, end_sent)
        else:
            has_conn, args = find_conn_within_sent(sent, conn, False, True, False)
        if has_conn:
            if conn in filter_dict:
                filter_func = filter_dict[conn]
                filter_flag = filter_func(args[0], args[1])
                if not filter_flag:
                    return True, args
            else:
                return True, args

    return False, None


def is_exp_inter(sent1, sent2, conn_list):
    special_conns = ["although", "even if", "even though", "if", "yet"]
    filter_dict = {
        "so": filter_so, "or": filter_or, "as": filter_as,
        "and": filter_and, "after": filter_after, "because": filter_because,
        "before": filter_before, "but": filter_but, "if": filter_if,
        "when": filter_when, "since": filter_since, "then": filter_then
    }
    for conn in conn_list:
        if conn not in special_conns:
            has_conn, args = find_conn_between_sent(sent1, sent2, conn)
            if has_conn:
                if conn in filter_dict:
                    filter_func = filter_dict[conn]
                    filter_flag = filter_func(args[0], args[1])
                    if not filter_flag:
                        return True, args
                else:
                    return True, args

    return False, None


def split_into_sentences(text):
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|me|edu)"
    digits = "([0-9])"

    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = text.replace("e.g.","e<prd>g<prd>")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]

    return sentences


def pack_batch_data(arg_pairs, tokenizer, max_length):
    """
    Args:
        arg_pairs:
        tokenizer:
        max_length:
    """
    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    for item in arg_pairs:
        arg1 = item[0]
        arg2 = item[1]
        tmp_text_res = tokenizer(
            text=arg1,
            text_pair=arg2,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = tmp_text_res.input_ids
        attention_mask = tmp_text_res.attention_mask
        if "token_type_ids" in tmp_text_res:
            token_type_ids = tmp_text_res["token_type_ids"]
        else:
            token_type_ids = torch.zeros_like(attention_mask)
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
    # all_input_ids = np.array(all_input_ids)
    # all_attention_mask = np.array(all_attention_mask)
    # all_token_type_ids = np.array(all_token_type_ids)
    all_input_ids = torch.cat(all_input_ids, dim=0)
    all_attention_mask = torch.cat(all_attention_mask, dim=0)
    all_token_type_ids = torch.cat(all_token_type_ids, dim=0)

    return (all_input_ids, all_attention_mask, all_token_type_ids)


def split_into_sentences_stanza(text, stanza_nlp):
    nlp_text = stanza_nlp(text)
    sentences = []
    for sentence in nlp_text.sentences:
        sentences.append(sentence.text)

    return sentences