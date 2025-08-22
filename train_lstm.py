# author = liuwei
# date = 2024-01-03

import os
import json
import random
import time
import datetime
from tqdm import tqdm, trange

import argparse
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from dataset import LSTMDataset
from model import TextLSTM
from utils import get_vocab_from_corpus, labels_from_file

# for output
dt = datetime.datetime.now()
TIME_CHECKPOINT_DIR = "checkpoint_{}-{}-{}_{}:{}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
PREFIX_CHECKPOINT_DIR = "checkpoint"

def get_argparse():
    parser = argparse.ArgumentParser()

    # for data
    parser.add_argument("--data_dir", default="data/dataset", type=str)
    parser.add_argument("--dataset", default="test", type=str, help="toefl1234, gcdc")
    parser.add_argument("--fold_id", default=-1, type=int, help="[1-5] for toefl1234, [1-10] for gcdc")
    parser.add_argument("--output_dir", default="data/result", type=str, help="path to save checkpoint")
    parser.add_argument("--embed_file", default="glove.840B.300d.txt", type=str)
    parser.add_argument("--use_text_or_rel", default=0, type=int, help="0 for text, 1 for rel")
    parser.add_argument("--label_list", default="low, medium, high", type=str)
    parser.add_argument("--rel_level", default=2, type=int, help="1, 2")

    # for model
    parser.add_argument("--input_dim", default=300, type=int, help="the dimension size of fc layer")
    parser.add_argument("--hidden_size", default=256, type=int, help="the dimension size of fc layer")
    parser.add_argument("--num_layers", default=1, type=int, help="layer number of lstm")

    # for training
    parser.add_argument("--do_train", default=False, action="store_true", help="Whether to run training.")
    parser.add_argument("--do_dev", default=False, action="store_true", help="Whether to do evaluation")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--max_seq_length", default=256, type=int, help="max length of text sequence")
    parser.add_argument("--max_rel_length", default=64, type=int, help="max length of rel sequence")
    parser.add_argument("--train_batch_size", default=16, type=int, help="the training batch size")
    parser.add_argument("--eval_batch_size", default=36, type=int, help="the eval batch size")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="training epoch, only work when max_step==-1")
    parser.add_argument("--learning_rate", default=5e-2, type=float, help="The initial learning rate for Adam")
    parser.add_argument("--dropout", default=0.5, type=float, help="dropout value")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="the weight of L2 normalization")
    parser.add_argument("--warmup_ratio", default=0.0, type=float, help="warmup rate duing train")
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="max gradient value")
    parser.add_argument("--seed", default=106524, type=int, help="the seed used to initiate parameters")

    return parser

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dataloader(dataset, args, mode="train"):
    if mode.upper() == 'TRAIN':
        sampler = RandomSampler(dataset)
        batch_size = args.train_batch_size
    else:
        sampler = SequentialSampler(dataset)
        batch_size = args.eval_batch_size

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler
    )

    return data_loader

def get_optimizer(model, args, num_training_steps):
    no_deday = []
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_deday)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_deday)],
            "weight_decay": 0.0
        }
    ]
    optimizer = Adam(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    return optimizer, scheduler


def train(model, args, train_dataloader, dev_dataloader, test_dataloader):
    num_train_epochs = args.num_train_epochs
    print_step = len(train_dataloader) // 4 + 1
    num_train_steps = int(len(train_dataloader) * num_train_epochs)
    optimizer, scheduler = get_optimizer(model, args, num_train_steps)

    global_step = 0
    avg_loss = 0.0
    cur_loss = 0.0
    best_dev = 0.0
    best_dev_epoch = 0
    best_test = 0.0
    best_test_epoch = 0
    model.zero_grad()
    train_iter = trange(1, num_train_epochs + 1, desc="Epoch")
    for epoch in train_iter:
        epoch_iter = tqdm(train_dataloader, desc="Iteration")
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(epoch_iter):
            batch = tuple(t.to(args.device) for t in batch)
            if args.use_text_or_rel == 0:
                inputs = {
                    "input_ids": batch[0],
                    "seq_lengths": batch[2],
                    "labels": batch[4],
                    "flag": "Train"
                }
            elif args.use_text_or_rel == 1:
                inputs = {
                    "input_ids": batch[1],
                    "seq_lengths": batch[3],
                    "labels": batch[4],
                    "flag": "Train"
                }
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            cur_loss = loss.item()
            avg_loss += cur_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % print_step == 0:
                print("global step: %d, cur loss: %.4f, avg loss: %.4f" % (global_step, cur_loss, avg_loss / global_step))
        scheduler.step()
        # evaluation after each epoch
        model.eval()
        dev_acc, dev_f1 = evaluate(model, args, dev_dataloader, desc="dev")
        test_acc, test_f1 = evaluate(model, args, test_dataloader, desc="test")
        print()
        print(" Epoch=%d, Dev: Acc=%.4f, F1=%.4f" % (epoch, dev_acc, dev_f1))
        print(" Epoch=%d, Test: Acc=%.4f, F1=%.4f" % (epoch, test_acc, test_f1))
        if (dev_f1 > best_dev) or (dev_f1 == best_dev and test_f1 > best_test):
            best_dev = dev_f1
            best_dev_epoch = epoch
        if test_f1 > best_test:
            best_test = test_f1
            best_test_epoch = epoch
        output_dir = os.path.join(args.output_dir, "good")
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        # os.makedirs(output_dir, exist_ok=True)
        # torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    print(" Best dev: Epoch=%d, F1=%.4f" % (best_dev_epoch, best_dev))
    print(" Best test: Epoch=%d, F1=%.4f" % (best_test_epoch, best_test))


def evaluate(model, args, dataloader, desc="dev", write_file=False):
    all_label_ids = None
    all_pred_ids = None
    label_index = -1
    for batch in tqdm(dataloader, desc=desc):
        batch = tuple(t.to(args.device) for t in batch)
        if args.use_text_or_rel == 0:
            inputs = {
                "input_ids": batch[0],
                "seq_lengths": batch[2],
                "labels": batch[4],
                "flag": "Eval"
            }
            model_type = "text_lstm"
        elif args.use_text_or_rel == 1:
            inputs = {
                "input_ids": batch[1],
                "seq_lengths": batch[3],
                "labels": batch[4],
                "flag": "Eval"
            }
            model_type = "rel_lstm"
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs[0]

        label_ids = batch[label_index].detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        if all_label_ids is None:
            all_label_ids = label_ids
            all_pred_ids = preds
        else:
            all_label_ids = np.append(all_label_ids, label_ids)
            all_pred_ids = np.append(all_pred_ids, preds)

    acc = accuracy_score(y_true=all_label_ids, y_pred=all_pred_ids)
    f1 = f1_score(y_true=all_label_ids, y_pred=all_pred_ids, average="macro")
    """
    res = classification_report(
        y_true=all_label_ids,
        y_pred=all_pred_ids,
        target_names=args.label_list
    )
    print(res)
    """

    if write_file:
        res_file = "{}_res_{}.txt".format(desc, model_type)
        data_dir = os.path.join(args.data_dir, "preds")
        os.makedirs(data_dir, exist_ok=True)
        res_file = os.path.join(data_dir, res_file)
        error_num = 0
        with open(res_file, "w", encoding="utf-8") as f:
            for l, p in zip(all_label_ids, all_pred_ids):
                if l == p:
                    f.write("%s\t%s\n" % (args.label_list[l], args.label_list[p]))
                else:
                    error_num += 1
                    f.write("%s\t%s\t%d\n" % (args.label_list[l], args.label_list[p], error_num))

    return acc, f1


def main():
    args = get_argparse().parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    args.device = device
    print("Training/evaluation parameters %s", args)
    set_seed(args.seed)

    # 1. data related
    data_dir = os.path.join(args.data_dir, args.dataset)
    if args.use_text_or_rel == 0:
        saved_file = "text_embed.bin"
    else:
        saved_file = "rel_embed_l{}.bin".format(args.rel_level)
    args.saved_embed_file = os.path.join(data_dir, saved_file)
    if args.fold_id > 0:
        data_dir = os.path.join(data_dir, str(args.fold_id))
    args.data_dir = data_dir
    train_data_file = os.path.join(data_dir, "train.json")
    dev_data_file = os.path.join(data_dir, "dev.json")
    test_data_file = os.path.join(data_dir, "test.json")
    output_dir = os.path.join(args.output_dir, args.dataset)
    prefix = ["text", "rel"]
    output_dir = os.path.join(
        output_dir,
        "lstm_{}".format(prefix[args.use_text_or_rel])
    )
    if args.fold_id > 0:
        output_dir = os.path.join(output_dir, str(args.fold_id))
    args.output_dir = output_dir
    exp_rel_list = labels_from_file("data/parser/pdtb3/exp/l{}/labels.txt".format(args.rel_level))
    imp_rel_list = labels_from_file("data/parser/pdtb3/imp/l{}/labels.txt".format(args.rel_level))
    rel_list = set()
    _ = [rel_list.add(l) for l in exp_rel_list]
    _ = [rel_list.add(l) for l in imp_rel_list]
    rel_list = list(rel_list)
    rel_list = sorted(rel_list)
    print(rel_list)
    args.rel_list = rel_list
    label_list = args.label_list.split(",")
    label_list = [l.lower().strip() for l in label_list]
    args.label_list = label_list
    args.num_labels = len(label_list)
    vocab = get_vocab_from_corpus(args.data_dir)
    args.embed_file = os.path.join(
        "/hits/basement/nlp/liuwi/resources/embeddings",
        args.embed_file
    )
    print("vocab size: ", len(vocab))

    # 2. model
    if args.use_text_or_rel == 0:
        args.vocab = vocab
    else:
        args.vocab = rel_list
    model = TextLSTM(args)
    model = model.to(args.device)
    # 3. train and evaluation
    data_params = {
        "max_seq_length": args.max_seq_length,
        "max_rel_length": args.max_rel_length,
        "label_list": label_list,
        "rel_list": rel_list,
        "vocab": vocab
    }
    if args.do_train:
        print("Acc for {} fold {}: ".format(args.dataset, args.fold_id))
        train_dataset = LSTMDataset(train_data_file, data_params)
        dev_dataset = LSTMDataset(dev_data_file, data_params)
        test_dataset = LSTMDataset(test_data_file, data_params)
        train_dataloader = get_dataloader(train_dataset, args)
        dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")
        test_dataloader = get_dataloader(test_dataset, args, mode="test")
        train(model, args, train_dataloader, dev_dataloader, test_dataloader)

    if args.do_dev or args.do_test:
        tempfile = os.path.join(output_dir, "good/checkpoint_{}/pytorch_model.bin")
        dev_dataset = LSTMDataset(dev_data_file, data_params)
        test_dataset = LSTMDataset(test_data_file, data_params)
        dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")
        test_dataloader = get_dataloader(test_dataset, args, mode="test")

        for epoch in range(8, 9):
            checkpoint_file = tempfile.format(str(epoch))
            print(checkpoint_file)
            model.load_state_dict(torch.load(checkpoint_file))
            model.eval()

            if args.do_dev:
                acc, f1 = evaluate(model, args, dev_dataloader, desc="dev", write_file=True)
                print(" Epoch=%d, Dev Acc=%.4f, F1: %.4f" % (epoch, acc, f1))

            if args.do_test:
                acc, f1 = evaluate(model, args, test_dataloader, desc="test", write_file=True)
                print(" Epoch=%d, Test Acc=%.4f, F1: %.4f" % (epoch, acc, f1))


if __name__ == "__main__":
    main()
