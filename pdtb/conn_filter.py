# author = "liuwei"
# date = 2023-11-06

import os
import json
import random
import re
import math

import nltk
nltk.data.path.append("/hits/basement/nlp/liuwi/resources/nltk")
from tqdm import tqdm, trange
from multiprocessing import Process
from nltk import word_tokenize

def filter_so(arg1, conn_arg2):
    """
    filter so samples that so is not a connective
    """
    text = arg1.lower() + " " + conn_arg2.lower()
    tokens = word_tokenize(text)
    start_pos = len(word_tokenize(arg1.lower()))
    if "so" not in tokens[start_pos:]:
        return True
    so_idx = tokens.index("so", start_pos)
    # 1 so that, so many, so much, ...
    if tokens[so_idx+1] in ["that", "many", "much", "far", "on", "do", "am", "is", "does", "did", "was", "were", "as", "to"]:
        return True

    pos_res = nltk.pos_tag(tokens)

    # filter so + adj and so + adv, JJ = adj, RB = adv
    if "JJ" in pos_res[so_idx+1][1] or "RB" in pos_res[so_idx+1][1] or "VB" in pos_res[so_idx+1][1]:
        return True
    # filter do so or think so, verb + so, VB = verb
    if "VB" in pos_res[so_idx-1][1]:
        return True

    return False

def filter_or(arg1, conn_arg2):
    text = arg1.lower() + " " + conn_arg2.lower()
    tokens = word_tokenize(text)
    start_pos = len(word_tokenize(arg1.lower())) - 1
    if "or" not in tokens[start_pos:]:
        return True
    or_index = tokens.index("or", start_pos)
    if "either" in tokens[:start_pos] or "whether" in tokens[:start_pos]:
        return True
    if tokens[or_index+1] in ["not", "sometimes", "so", "something", "some", "more", "else", "someone", "anyone", "even", "every", "anything", "no", "any"]:
        return True
    if "," in tokens[or_index+1:or_index+4]:
        return True

    pos_res = nltk.pos_tag(tokens)
    if pos_res[or_index-1][1] == pos_res[or_index+1][1]:
        return True
    # if pos_res[or_index-2][1] == pos_res[or_index+1][1] and "," == tokens[or_index-1]:
    #     return True
    if (pos_res[or_index-2][1] == pos_res[or_index+1][1]):
        return True

    # if (pos_res[or_index-2][1] == pos_res[or_index+1][1]) and (pos_res[or_index-1][1] == pos_res[or_index+2][1]):
    #     return True
    # if (pos_res[or_index-3][1] == pos_res[or_index+1][1]) and (pos_res[or_index-2][1] == pos_res[or_index+2][1]) and (pos_res[or_index-1][1] == pos_res[or_index+3][1]):
    #     return True
    if "JJ" in pos_res[or_index-1][1] and "JJ" in pos_res[or_index+1][1]:
        return True
    if "NN" in pos_res[or_index-1][1] and "NN" in pos_res[or_index+1][1]:
        return True
    if "VB" in pos_res[or_index-1][1] and "VB" in pos_res[or_index+1][1]:
        return True
    if "NN" in pos_res[or_index-2][1] and "NN" in pos_res[or_index+1][1] and "," == tokens[or_index-1]:
        return True

    if tokens[or_index-2] in ["a", "an", "the"] and tokens[or_index+1] in ["a", "an", "the", "two"]:
        return True

    return False

def filter_as(arg1, conn_arg2):
    """
    filter as samples
    """
    text = arg1.lower() + " " + conn_arg2.lower()
    if "same as" in text or conn_arg2.startswith("as though") or conn_arg2.startswith("as if") or conn_arg2.startswith("as a matter of"):
        return True
    if conn_arg2.startswith("as the reason why") or conn_arg2.startswith("as the one who"):
        return True

    tokens = word_tokenize(text)
    pos_res = nltk.pos_tag(tokens)
    start_pos = len(word_tokenize(arg1.lower())) - 1
    if "as" not in tokens[start_pos:]:
        return True
    as_idx = tokens.index("as", start_pos)

    if ("regard" in tokens[:as_idx]) or ("regarded" in tokens[:as_idx]) or ("regards" in tokens[:as_idx]) or ("regarding" in tokens[:as_idx]):
        return True
    if ("consider" in tokens[:as_idx]) or ("considered" in tokens[:as_idx]) or ("considers" in tokens[:as_idx]) or ("considering" in tokens[:as_idx]):
        return True
    if ("discovering" in tokens[:as_idx]) or ("discovered" in tokens[:as_idx]):
        return True
    if ("describe" in tokens[:as_idx]) or ("described" in tokens[:as_idx]) or ("describes" in tokens[:as_idx]) or ("describing" in tokens[:as_idx]):
        return True
    if ("think" in tokens[:as_idx]) or ("though" in tokens[:as_idx]) or ("thinks" in tokens[:as_idx]) or ("thinking" in tokens[:as_idx]):
        return True
    if ("view" in tokens[:as_idx]) or ("viewed" in tokens[:as_idx]) or ("views" in tokens[:as_idx]) or ("viewing" in tokens[:as_idx]):
        return True
    if ("treat" in tokens[:as_idx]) or ("treated" in tokens[:as_idx]) or ("treats" in tokens[:as_idx]) or ("treating" in tokens[:as_idx]):
        return True
    if ("define" in tokens[:as_idx]) or ("defined" in tokens[:as_idx]) or ("defines" in tokens[:as_idx]) or ("defining" in tokens[:as_idx]):
        return True

    # as ... as
    if "as" in tokens[as_idx+1:]:
        return True
    if "," in tokens[as_idx+1:as_idx+6]:
        return True
    if "recognize" in tokens[as_idx-2] and tokens[as_idx-1] in ["it", "that", "this", "him", "her", "me", "them"]:
        return True
    if "same" in tokens[as_idx-4:as_idx]:
        return True
    if tokens[as_idx-1] in ["is", "are", "was", "were", "and", "but"]:
        return True
    if tokens[as_idx-2] in ["is", "are", "was", "were"] and "RB" in pos_res[as_idx-1][1]:
        return True
    if tokens[as_idx-1] in ["and"]:
        return True
    if "only" in tokens[as_idx+2:as_idx+4]:
        return True
    if "another" in tokens[as_idx+1:as_idx+3]:
        return True

    # As I said, As the forecast predicted
    if pos_res[as_idx+1][1] == "PRP" and "VB" in pos_res[as_idx+2][1]:
        return True
    if "NN" in pos_res[as_idx+1][1] and "VB" in pos_res[as_idx+2][1]:
        return True
    if pos_res[as_idx+1][1] == "DT" and "NN" in pos_res[as_idx+2][1] and "VB" in pos_res[as_idx+3][1]:
        return True
    # as a boy
    if pos_res[as_idx+1][0] in ["a", "an"] or pos_res[as_idx+1][1] == "CD":
        return True
    # if (pos_res[as_idx+1][0] in ["a", "an"] or pos_res[as_idx+1][1] == "CD") and "NN" in pos_res[as_idx+2][1]:
    #     return True
    # sth as sth
    if "NN" in pos_res[as_idx-1][1] and "NN" in pos_res[as_idx+1][1]:
        return True
    # much as, adverb as
    if "RB" in pos_res[as_idx-1][1]:
        return True

    if "JJ" in pos_res[as_idx+1][1]:
        return True
    # work as, arrived as
    if as_idx > 0 and "VB" in pos_res[as_idx-1][1]:
        return True
    # such as, as usual
    if pos_res[as_idx-1][0] == "such" or pos_res[as_idx+1][0] in ["usual", "for", "always", "well", "yours", "mine", "to", "did", "do", "does", "him", "her", "them", "long"]:
        return True
    # as my son
    if pos_res[as_idx+1][1] == "PRP$" and "NN" in tokens[as_idx+2]:
        return True
    # treat you as
    if "VB" in pos_res[as_idx-2][1] and pos_res[as_idx-1][1] == "PRP":
        return True
    if tokens[as_idx+1] in ["is", "are", "was", "were", "if"]:
        return True
    if "VB" in pos_res[as_idx+1][1]:
        return True
    if "NN" in pos_res[as_idx+1][1] and tokens[as_idx+2] == "and":
        return True
    if "NN" in pos_res[as_idx+2][1] and tokens[as_idx+3] == "and":
        return True
    if "NN" in pos_res[as_idx+1][1] and pos_res[as_idx+2][1] == "IN":
        return True
    if pos_res[as_idx-1][1] == "PRP" and "NN" in pos_res[as_idx+1][1]:
        return True
    if pos_res[as_idx+1][1] == "IN":
        return True
    if tokens[as_idx+1] == "the" and "NN" not in pos_res[as_idx+2][1]:
        return True
    # as the reason
    if tokens[as_idx+1] == "the" and "NN" in pos_res[as_idx+2][1] and pos_res[as_idx+3][1] == "PRP":
        return True
    # as my reward for
    if "NN" in pos_res[as_idx+2][1] and pos_res[as_idx+3][1] == "IN":
        return True
    # as I always did
    if "do" in tokens[as_idx+2:as_idx+6] or "does" in tokens[as_idx+2:as_idx+6] or "did" in tokens[as_idx+2:as_idx+6] or "done" in tokens[as_idx+2:as_idx+6]:
        return True

    if "," in tokens[as_idx+1:as_idx+7]:
        return True
    if ";" in tokens[as_idx+1:as_idx+7]:
        return True
    if ")" in tokens[as_idx+1:as_idx+7]:
        return True
    if "-" in tokens[as_idx+1:as_idx+7]:
        return True

    return False

def filter_and(arg1, conn_arg2):
    """
    filter and
    """
    text = arg1.lower() + " " + conn_arg2.lower()
    tokens = word_tokenize(text)
    pos_res = nltk.pos_tag(tokens)
    if "and" not in tokens:
        return True
    and_idx = tokens.index("and")

    if pos_res[and_idx-1][1] == pos_res[and_idx+1][1]:
        return True
    if "JJ" in pos_res[and_idx-1][1] and "NN" in pos_res[and_idx+1][1]:
        return True

    # and + verb, and + adj
    if "VB" in pos_res[and_idx+1][1] or "JJ" in pos_res[and_idx+1][1]:
        return True
    if "RB" in pos_res[and_idx+1][1] and "VB" in pos_res[and_idx+2][1]:
        return True
    if "RB" in pos_res[and_idx+1][1] and "JJ" in pos_res[and_idx+2][1]:
        return True

    if "go and" in text or "come and" in text:
        return True

    # if "NN" in pos_res[and_idx-1][1] and ("NN" in pos_res[and_idx+1][1] or \
    #    "NN" in pos_res[and_idx+2][1] or "NN" in pos_res[and_idx+3][1]):
    #     return True
    # if "and then" in text:
    #     return True
    if pos_res[and_idx+1][1] in ["IN", "TO"]:
        return True

    if pos_res[and_idx+1][1] in ["PRP", "WP", "WP$", "WRB", "DT"]:
        return False
    if "NN" in pos_res[and_idx+1][1]:
        return False

    return False

def filter_after(arg1, conn_arg2):
    text = arg1.lower() + " " + conn_arg2.lower()
    tokens = word_tokenize(text)
    pos_res = nltk.pos_tag(tokens)
    if "after" not in tokens:
        return True
    after_idx = tokens.index("after")
    if "," in tokens[after_idx+1:after_idx+8] or "-" in tokens[after_idx+1:after_idx+8] or ";" in tokens[after_idx+1:after_idx+8]:
        return True
    if tokens[after_idx+1] in ["all", "that", "which", "more", "a", "an"]:
        return True
    if tokens[after_idx+1] in ["the"] and "of" in tokens[after_idx+1:after_idx+6]:
        return True
    if tokens[after_idx+1] in ["the"] and "in" in tokens[after_idx+1:after_idx+6]:
        return True
    if tokens[after_idx-1] in ["look", "looks", "looked"]:
        return True
    if "," in tokens[after_idx-6:after_idx-1]:
        return True


    if pos_res[after_idx+1][1] == "CD":
        return True

    return False

def filter_because(arg1, conn_arg2):
    text = arg1.lower() + " " + conn_arg2.lower()
    tokens = word_tokenize(text)
    if "because" not in tokens:
        return True
    because_idx = tokens.index("because")
    if tokens[because_idx+1] == "of":
        return True
    if tokens[because_idx-1] in ["is", "was"]:
        return True

    return False

def filter_before(arg1, conn_arg2):
    text = arg1.lower() + " " + conn_arg2.lower()
    tokens = word_tokenize(text)
    if "before" not in tokens:
        return True
    before_idx = tokens.index("before")
    if "," in tokens[before_idx+1:before_idx+4]:
        return True
    if "." in tokens[before_idx+1:before_idx+4]:
        return True
    pos_res = nltk.pos_tag(tokens)
    if "VB" in pos_res[before_idx-1][1]:
        return True

    return False

def filter_but(arg1, conn_arg2):
    text = arg1.lower() + " " + conn_arg2.lower()
    tokens = word_tokenize(conn_arg2)
    if "but" not in tokens:
        return True
    but_index = tokens.index("but")
    if tokens[but_index-1] in ["all"]: # , "everything", "everyone", "nobody"]:
        return True
    if "no one but" in text:
        return True
    if tokens[but_index+1] in ["for", "then", "also"]:
        return True
    pos_res = nltk.pos_tag(tokens)
    if pos_res[but_index+1][1] in ["RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
        return True

    return False

def filter_if(arg1, conn_arg2):
    text = arg1.lower() + " " + conn_arg2.lower()
    tokens = word_tokenize(text)
    start_pos = len(word_tokenize(arg1.lower())) - 1
    if "if" not in tokens[start_pos:]:
        return True
    if_index = tokens.index("if", start_pos)
    if tokens[if_index-1] in ["as", "even", "that", "and", "but", "idea", "so", "only", "because", "what", "sure", "unsure", "although", "wonder", "wondering", "or"]:
        return True
    if tokens[if_index+1] in ["so", "not", "possible", "neccessary", "don't", "didn't", "doesn't"]:
        return True
    if "ask" in tokens[:if_index] or "asked" in tokens[:if_index] or "asks" in tokens[:if_index]:
        return True
    if "," in tokens[if_index+1:if_index+4]:
        return True
    if "find out" in tokens[if_index-3:if_index]:
        return True

    pos_res = nltk.pos_tag(tokens)
    if "VB" in pos_res[if_index-1][1]:
        return True

    return False

def filter_when(arg1, conn_arg2):
    text = arg1.lower() + " " + conn_arg2.lower()
    tokens = word_tokenize(text)
    if tokens[0] in ["it", "there", "that", "this"]:
        return True
    start_pos = len(word_tokenize(arg1.lower()))
    if "when" not in tokens[start_pos:]:
        return True
    if conn_arg2.endswith("?"):
        return True
    when_index = tokens.index("when", start_pos)
    if "," in tokens[when_index+1:when_index+6]:
        return True
    if "but" in tokens[when_index-3:when_index]:
        return True
    if "and" in tokens[when_index-3:when_index]:
        return True
    if tokens[when_index+1] in ["everything", "everyone", "all", "things", "there"]:
        return True
    if tokens[when_index-1] in ["day", "days", "time", "times", "ago", "night", "and", "but", "so", "to", "in", "situation", "even", "not", "moment", "morning", "minute", "minutes", "hours", "hour", "second", "seconds", "month", "months", "year", "years", "period", "noon", "nights", "past", "age", "childhood"]:
        return True
    if tokens[when_index-1] == "," and (tokens[when_index-2] in ["day", "days", "time", "times", "ago", "night", "nights", "moment", "minute", "minutes", "second", "seconds", "hour", "hours", "month", "months", "year", "years", "nights", "period", "noon", "past", "age"]):
        return True
    if "will" in tokens[:start_pos]:
        return True
    if "would" in tokens[:start_pos]:
        return True
    if "might" in tokens[:start_pos]:
        return True
    if "could" in tokens[:start_pos]:
        return True
    if "should" in tokens[:start_pos]:
        return True
    if "may" in tokens[:start_pos]:
        return True
    if "maybe" in tokens[:start_pos]:
        return True
    if "can" in tokens[:start_pos]:
        return True
    if "seems" in tokens[:start_pos]:
        return True
    if "seemed" in tokens[:start_pos]:
        return True
    if "going to" in arg1:
        return True
    if "\'d" in arg1:
        return True
    if "\'ll" in arg1:
        return True
    if "\'t" in arg1:
        return True
    if "usually" in tokens[:start_pos]:
        return True
    if "only" in tokens[:start_pos]:
        return True
    if "never" in tokens[:start_pos]:
        return True
    if "no" in tokens[:start_pos]:
        return True
    if "none" in tokens[:start_pos]:
        return True

    pos_res = nltk.pos_tag(tokens)
    if pos_res[when_index-1][1] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "IN", "RB", "CD"]:
        return True

    return False

def filter_since(arg1, conn_arg2):
    text = arg1.lower() + " " + conn_arg2.lower()
    tokens = word_tokenize(text)
    start_pos = len(word_tokenize(arg1.lower()))
    if "since" not in tokens[start_pos:]:
        return True
    since_index = tokens.index("since", start_pos)
    if "," in tokens[since_index+1:]:
        return True
    if tokens[since_index-1] in ["and", "but"]:
        return True
    if tokens[since_index+1] in ["then", "when"]:
        return True
    if tokens[since_index-1] in ["ever"]:
        return False

    pos_res = nltk.pos_tag(tokens)
    if pos_res[since_index+1][1] == "CD":
        return True

    return True

def filter_then(arg1, conn_arg2):
    text = arg1.lower() + " " + conn_arg2.lower()
    tokens = word_tokenize(text)
    start_pos = len(word_tokenize(arg1.lower())) - 1
    if "then" not in tokens[start_pos:]:
        return True
    then_idx = tokens.index("then", start_pos)
    if "if" in tokens[0:start_pos]:
        return True
    if tokens[then_idx - 1] not in [",", ".", ";"]:
        return True

    return False

