# author = liuwei
# date = 2024-05-07

import logging
import os
import json
import pickle
import math
import random
import time
import datetime
from tqdm import tqdm, trange

from sklearn.metrics import f1_score, accuracy_score, classification_report

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel, BertModel
from transformers import XLNetConfig, XLNetTokenizer, XLNetModel
from transformers import LlamaConfig, LlamaTokenizer, LlamaModel
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModel
from dataset import SentDataset
from model import TransformerSent, TransformerSentRel, TransformerSentRelFusion, WholeDoc

from utils import labels_from_file

# logging.disable(logging.WARNING)

# set logger, print to console and write to file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()  # 输出到控制台的handler
chlr.setFormatter(formatter)
logger.addHandler(chlr)

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
    parser.add_argument("--label_list", default="low, medium, high", type=str)
    parser.add_argument("--model_type", default="transformer_sent", type=str, help="transformer_sent, transformer_sent_rel")
    parser.add_argument("--pooler_type", default="avg", type=str, help="cls, avg")
    parser.add_argument("--embed_dim", default=300, type=int, help="the dimension size of fc layer")
    parser.add_argument("--embed_file", default="glove.840B.300d.txt", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str, help="roberta-base")

    # for training
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_dev", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=48, type=int)
    parser.add_argument("--max_text_length", default=512, type=int)
    parser.add_argument("--max_arg_num", default=32, type=int)
    parser.add_argument("--max_sent_num", default=24, type=int, help="number of sents in a text")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="training epoch")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="learning rate")
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--seed", default=106524, type=int, help="random seed")

    # for transformer encoder
    parser.add_argument("--num_layers", default=1, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--scaled", default=True)

    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloader(dataset, args, mode="train"):
    print("  {} dataset length: ".format(mode), len(dataset))
    if mode.lower() == "train":
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
    specific_params = []
    no_deday = ["bias", "LayerNorm.weigh"]
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(num_training_steps * args.warmup_ratio),
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler


def train(model, args, train_dataloader, dev_dataloader, test_dataloader):
    ## 1. prepare data
    t_total = int(len(train_dataloader) * args.num_train_epochs)
    num_train_epochs = args.num_train_epochs
    print_step = int(len(train_dataloader) // 4)

    ## 2.optimizer
    optimizer, scheduler = get_optimizer(model, args, t_total)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Batch size per device = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    logging_loss = 0.0
    best_dev = 0.0
    best_dev_epoch = 0
    best_test = 0.0
    best_test_epoch = 0
    train_iterator = trange(1, int(num_train_epochs) + 1, desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        model.train()
        model.zero_grad()
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            if args.model_type.lower() == "transformer_sent":
                inputs = {
                    "sent_vectors": batch[1],
                    "sent_mask": batch[3],
                    "labels": batch[8],
                    "flag": "Train"
                }
            elif args.model_type.lower() == "transformer_sent_rel":
                inputs = {
                    "sent_vectors": batch[1],
                    "sent_rel_ids": batch[2],
                    "sent_mask": batch[3],
                    "labels": batch[8],
                    "flag": "Train"
                }
            elif args.model_type.lower() == "transformer_sent_rel_fusion":
                inputs = {
                    "sent_vectors": batch[1],
                    "sent_rel_ids": batch[2],
                    "sent_mask": batch[3],
                    "sent_rel_mask": batch[9],
                    "labels": batch[8],
                    "flag": "Train"
                }
            elif args.model_type.lower() == "flat":
                inputs = {
                    "input_vectors": batch[0],
                    "labels": batch[8],
                    "flag": "Train"
                }
            outputs = model(**inputs)
            loss = outputs[0]

            # optimizer.zero_grad()
            loss.backward()
            global_step += 1
            logging_loss = loss.item()
            tr_loss += logging_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # print(logging_loss)

            if global_step % print_step == 0:
                print(" Current loss=%.4f, global average loss=%.4f" % (logging_loss, tr_loss / global_step))

        # evaluation and save
        model.eval()
        dev_acc, dev_f1 = evaluate(
            model, args, dev_dataloader, epoch, desc="dev"
        )
        test_acc, test_f1 = evaluate(
            model, args, test_dataloader, epoch, desc="test"
        )
        print()
        print(" Dev Acc=%.4f, F1=%.4f" % (dev_acc, dev_f1))
        print(" Test Acc=%.4f, F1=%.4f" % (test_acc, test_f1))
        if dev_f1 > best_dev:
            best_dev = dev_f1
            best_dev_epoch = epoch
        if test_f1 > best_test:
            best_test = test_f1
            best_test_epoch = epoch

        output_dir = os.path.join(args.output_dir, "good")
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        if args.model_type not in ["base", "avg_sent", "lstm_sent", "lstm_sent_rel", "lattice_sent_rel"]:
            os.makedirs(output_dir, exist_ok=True)
            # torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        else:
            print(" ## Not save ##")
    print(" Best dev: Epoch=%d, F1=%.4f\n" % (best_dev_epoch, best_dev))
    print(" Best test: Epoch=%d, F1=%.4f\n" % (best_test_epoch, best_test))

    return best_dev_epoch, best_test_epoch


def evaluate(model, args, dataloader, epoch, desc="dev", write_file=False):
    all_label_ids = None
    all_pred_ids = None
    for batch in tqdm(dataloader, desc=desc):
        batch = tuple(t.to(args.device) for t in batch)
        if args.model_type.lower() == "transformer_sent":
            inputs = {
                "sent_vectors": batch[1],
                "sent_mask": batch[3],
                "labels": batch[8],
                "flag": "Eval"
            }
        elif args.model_type.lower() == "transformer_sent_rel":
            inputs = {
                "sent_vectors": batch[1],
                "sent_rel_ids": batch[2],
                "sent_mask": batch[3],
                "labels": batch[8],
                "flag": "Eval"
            }
        elif args.model_type.lower() == "transformer_sent_rel_fusion":
            inputs = {
                "sent_vectors": batch[1],
                "sent_rel_ids": batch[2],
                "sent_mask": batch[3],
                "sent_rel_mask": batch[9],
                "labels": batch[8],
                "flag": "Eval"
            }
        elif args.model_type.lower() == "flat":
                inputs = {
                    "input_vectors": batch[0],
                    "labels": batch[8],
                    "flag": "Eval"
                }
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs[0]

        label_ids = batch[8].detach().cpu().numpy()
        pred_ids = preds.detach().cpu().numpy()
        if all_label_ids is None:
            all_label_ids = label_ids
            all_pred_ids = pred_ids
        else:
            all_label_ids = np.append(all_label_ids, label_ids)
            all_pred_ids = np.append(all_pred_ids, pred_ids)

    # print(all_label_ids.shape, all_pred_ids.shape)
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
        res_file = "{}_res_{}.txt".format(desc, args.model_type)
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
        args.n_gpu = 1
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        args.n_gpu = 0
    args.device = device
    logger.info(" Training/evaluation parameters %s", args)
    logger.info(" ####### Acc, F1 for fold %d ########", args.fold_id)
    # print("Training/evaluation parameters %s", args)
    set_seed(args.seed)

    ## 1. prepare data
    data_dir = os.path.join(args.data_dir, args.dataset)
    saved_file = "rel_embed.bin"
    args.saved_embed_file = os.path.join(data_dir, saved_file)
    print(args.saved_embed_file)
    data_dir = os.path.join(data_dir, str(args.fold_id))
    train_data_file = os.path.join(data_dir, "train.json")
    dev_data_file = os.path.join(data_dir, "dev.json")
    test_data_file = os.path.join(data_dir, "test.json")
    args.data_dir = data_dir
    output_dir = os.path.join(args.output_dir, args.dataset)
    output_dir = os.path.join(
        output_dir,
        "fast_flat_{}+{}".format(args.model_type, args.model_name_or_path.split("/")[-1])
    )
    if args.fold_id > 0:
        output_dir = os.path.join(output_dir, str(args.fold_id))
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    exp_rel_list = labels_from_file("data/parser/pdtb3/exp/l2/labels.txt")
    imp_rel_list = labels_from_file("data/parser/pdtb3/imp/l2/labels.txt")
    rel_list = set()
    _ = [rel_list.add(l) for l in exp_rel_list]
    _ = [rel_list.add(l) for l in imp_rel_list]
    rel_list = list(rel_list)
    rel_list = sorted(rel_list)
    args.rel_list = rel_list
    label_list = args.label_list.split(",")
    label_list = [l.lower().strip() for l in label_list]
    args.label_list = label_list
    args.num_labels = len(label_list)

    args.embed_file = os.path.join(
        "/hits/basement/nlp/liuwi/resources/embeddings",
        args.embed_file
    )

    ## 2. define models
    args.model_name_or_path = os.path.join("/hits/basement/nlp/liuwi/resources/pretrained_models", args.model_name_or_path)
    print(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if "llama" in args.model_name_or_path:
        """
        encoder = AutoModel.from_pretrained(
            args.model_name_or_path,
            config=config,
            torch_dtype=torch.bfloat16
        ) 
        """
        encoder = None
    else:
        encoder = AutoModel.from_pretrained(
            args.model_name_or_path,
            config=config,
        )
    if encoder is not None:
        for name, param in encoder.named_parameters():
            param.requires_grad = False
        encoder.eval()
        encoder = encoder.to(args.device)

    config.HP_dropout = args.dropout
    args.input_size = config.hidden_size
    if args.model_type.lower() == "transformer_sent":
        model = TransformerSent(args=args)
    elif args.model_type.lower() == "transformer_sent_rel":
        model = TransformerSentRel(args=args)
    elif args.model_type.lower() == "transformer_sent_rel_fusion":
        model = TransformerSentRelFusion(args=args)
    elif args.model_type.lower() == "flat":
        model = WholeDoc(args=args)
    model = model.to(args.device)

    ## 3. prepare dataset
    dataset_params = {
        "tokenizer": tokenizer,
        "encoder": encoder,
        "pooler_type": args.pooler_type,
        "max_text_length": args.max_text_length,
        "max_arg_num": args.max_arg_num,
        "max_sent_num": args.max_sent_num,
        "label_list": label_list,
        "rel_list": rel_list,
    }

    if args.do_train:
        train_dataset = SentDataset(train_data_file, params=dataset_params)
        dev_dataset = SentDataset(dev_data_file, params=dataset_params)
        test_dataset = SentDataset(test_data_file, params=dataset_params)
        train_dataloader = get_dataloader(train_dataset, args, mode="train")
        dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")
        test_dataloader = get_dataloader(test_dataset, args, mode="test")
        train(model, args, train_dataloader, dev_dataloader, test_dataloader)

    if args.do_dev or args.do_test:
        temp_file = os.path.join(output_dir, "good/checkpoint_{}/pytorch_model.bin")

        dev_dataset = SentDataset(dev_data_file, params=dataset_params)
        test_data_file = "data/dataset/toefl_p2/{}/test.json".format(args.fold_id)
        test_dataset = SentDataset(test_data_file, params=dataset_params)
        dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")
        test_dataloader = get_dataloader(test_dataset, args, mode="test")
        """
        # 1. for doc length, toefl p1
        model_epoch = {
            "transformer_sent": [6, 17, 6, 18, 9],
            "transformer_sent_rel": [6, 20, 11, 7, 20],
            "transformer_sent_rel_fusion": [6, 6, 12, 10, 8]
        }
        """
        """
        # 2. for roberta, gcdc
        model_epoch = {
            "transformer_sent": [15, 20, 1, 1, 1, 12, 13, 20, 7, 9],
            "transformer_sent_rel": [6, 9, 1, 1, 1, 9, 13, 10, 17, 8],
            "transformer_sent_rel_fusion": [12, 5, 1, 1, 1, 12, 10, 10, 18, 19]
        }
        """
        """
        # 3. for llama gcdc
        model_epoch = {
            "transformer_sent": [1, 1, 19, 1, 15, 11, 1, 1, 7, 9],
            "transformer_sent_rel": [1, 1, 15, 1, 10, 9, 1, 1, 17, 8],
            "transformer_sent_rel_fusion": [1, 1, 19, 1, 14, 16, 1, 10, 18, 19]
        }
        """
        # 4. for llama toefl
        model_epoch = {
            "transformer_sent": [5, 1, 1, 5, 1],
            "transformer_sent_rel": [16, 1, 1, 16, 1],
            "transformer_sent_rel_fusion": [18, 1, 1, 18, 1]
        }
        eval_epoch = model_epoch[args.model_type.lower()][args.fold_id-1]
        # print(args.model_type, args.fold_id, eval_epoch)
        for epoch in range(eval_epoch, eval_epoch+1):
            checkpoint_file = temp_file.format(str(epoch))
            print(" Epoch=%d, checkpoint=%s" % (epoch, checkpoint_file))
            model.load_state_dict(torch.load(checkpoint_file))
            model.eval()
            do_write = True

            if args.do_dev:
                acc, f1 = evaluate(
                    model, args, dev_dataloader, epoch, "dev", do_write
                )
                print(" Dev Acc=%.4f, F1=%.4f" % (acc, f1))

            if args.do_test:
                acc, f1 = evaluate(
                    model, args, test_dataloader, epoch, "test", do_write
                )
                print(" Test Acc=%.4f, F1=%.4f" % (acc, f1))

if __name__ == "__main__":
    main()
