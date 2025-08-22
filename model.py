# author = liuwei
# date = 2024-01-05

import math
import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from module import BiLSTM, Embedding, Absolute_Position_Embedding, Transformer_Encoder, Relative_Position_Embedding, Transformer_Encoder_Fusion

class TextLSTM(nn.Module):
    def __init__(self, args):
        super(TextLSTM, self).__init__()

        self.num_labels = args.num_labels
        self.word_embedding = Embedding(
            vocab=args.vocab,
            embed_dim=args.input_dim,
            embed_file=args.embed_file,
            saved_file=args.saved_embed_file
        )
        self.bilstm = BiLSTM(
            args.input_dim,
            args.hidden_size,
            num_layers=args.num_layers
        )
        self.fc = nn.Linear(args.hidden_size, args.hidden_size // 4)
        self.classifier = nn.Linear(args.hidden_size // 4, self.num_labels)
        self.dropout = nn.Dropout(p=args.dropout)

        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.normal_(mean=0.0, std=0.02)
        self.fc.bias.data.zero_()
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()

    def forward(
        self,
        input_ids,
        seq_lengths,
        labels,
        flag="Train"
    ):
        input_embeddings = self.word_embedding(input_ids)
        input_embeddings = self.dropout(input_embeddings)
        lstm_out = self.bilstm(input_embeddings, seq_lengths)
        lstm_out = self.dropout(lstm_out)
        fc_out = self.fc(lstm_out)
        fc_out = self.dropout(fc_out)
        logits = self.classifier(fc_out)

        preds = torch.argmax(logits, dim=-1) # [N]
        outputs = (preds, )
        if flag.lower() == "train":
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, ) + outputs

        return outputs


class WholeDoc(nn.Module):
    def __init__(self, args):
        super(WholeDoc, self).__init__()

        self.num_labels = args.num_labels
        self.fc = nn.Linear(args.input_size, args.hidden_size)
        self.classifier = nn.Linear(args.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc.weight.data.normal_(mean=0.0, std=0.02)
        self.fc.bias.data.zero_()
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()
        self.relu = nn.ReLU()

    def forward(
        self,
        input_vectors,
        labels=None,
        flag="Train"
    ):
        if input_vectors.dtype == torch.bfloat16:
            input_vectors = input_vectors.float()
        input_vectors = self.dropout(input_vectors)
        # print(self.fc.weight.dtype, input_vectors.dtype)
        input_vectors = input_vectors.float()
        input_vectors = self.fc(input_vectors)
        input_vectors = self.relu(input_vectors)
        input_vectors = self.dropout(input_vectors)
        logits = self.classifier(input_vectors)

        _, preds = torch.max(logits, dim=-1)
        outputs = (preds,)
        if flag.upper() == 'TRAIN':
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

class AvgSent(nn.Module):
    def __init__(self, args):
        super(AvgSent, self).__init__()

        self.num_labels = args.num_labels
        self.fc = nn.Linear(args.input_size, args.hidden_size)
        self.classifier = nn.Linear(args.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc.weight.data.normal_(mean=0.0, std=0.02)
        self.fc.bias.data.zero_()
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()
        self.relu = nn.ReLU()

    def forward(
        self,
        sent_vectors,
        sent_mask,
        labels=None,
        flag="Train"
    ):
        batch_size, sent_num, _ = sent_vectors.size()
        expand_sent_mask = sent_mask.unsqueeze(2)  # [N, S, 1]
        masked_sent_reps = sent_vectors * expand_sent_mask  # [N, S, D]
        real_sent_num = torch.sum(sent_mask, dim=-1).unsqueeze(1)  # [N, 1]
        input_vectors = torch.sum(masked_sent_reps, dim=1) / real_sent_num  # [N, D]
        input_vectors = self.dropout(input_vectors)
        input_vectors = self.fc(input_vectors)
        input_vectors = self.relu(input_vectors)
        input_vectors = self.dropout(input_vectors)
        logits = self.classifier(input_vectors)

        _, preds = torch.max(logits, dim=-1)
        outputs = (preds,)
        if flag.upper() == 'TRAIN':
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

class TransformerSent(nn.Module):
    def __init__(self, args):
        super(TransformerSent, self).__init__()

        self.num_labels = args.num_labels
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.sent_proj = nn.Linear(self.input_size, self.hidden_size)
        self.abs_position_embedding = Absolute_Position_Embedding(self.hidden_size, learnable=False)
        self.transformer = Transformer_Encoder(
            {"num_layers": 1, "hidden_size": args.hidden_size,
             "num_heads": args.num_heads, "scaled": args.scaled}
        )
        self.fc = nn.Linear(args.hidden_size, args.hidden_size // 4)
        self.classifier = nn.Linear(args.hidden_size // 4, self.num_labels)
        self.dropout = nn.Dropout(p=args.dropout)
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()

    def forward(
        self,
        sent_vectors,
        sent_mask,
        labels=None,
        flag="Train"
    ):
        if sent_vectors.dtype == torch.bfloat16:
            sent_vectors = sent_vectors.float()
        sent_vectors = self.dropout(sent_vectors)
        sent_vectors = self.sent_proj(sent_vectors)
        sent_with_pe = self.abs_position_embedding(sent_vectors)
        output_vectors = self.transformer(sent_with_pe, sent_mask)
        output_vectors = self.dropout(output_vectors)
        output_vectors = self.fc(output_vectors)
        output_vectors = self.dropout(output_vectors)
        logits = self.classifier(output_vectors)

        _, preds = torch.max(logits, dim=-1)
        outputs = (preds,)
        if flag.upper() == 'TRAIN':
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

class TransformerSentRel(nn.Module):
    def __init__(self, args):
        super(TransformerSentRel, self).__init__()

        self.rel_embedding = Embedding(
            vocab=args.rel_list,
            embed_dim=args.embed_dim,
            embed_file=args.embed_file,
            saved_file=args.saved_embed_file
        )
        self.num_labels = args.num_labels
        self.input_size = args.input_size
        self.embed_dim = args.embed_dim
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.sent_proj = nn.Linear(self.input_size, self.hidden_size)
        self.rel_proj = nn.Linear(self.embed_dim, self.hidden_size)
        self.abs_position_embedding = Absolute_Position_Embedding(self.hidden_size, learnable=False)
        self.sent_transformer = Transformer_Encoder(
            {"num_layers": 1, "hidden_size": args.hidden_size,
             "num_heads": args.num_heads, "scaled": args.scaled}
        )
        self.rel_transformer = Transformer_Encoder(
            {"num_layers": 1, "hidden_size": args.hidden_size,
             "num_heads": args.num_heads, "scaled": args.scaled}
        )
        self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size // 4)
        self.classifier = nn.Linear(args.hidden_size // 4, self.num_labels)
        self.dropout = nn.Dropout(p=args.dropout)
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()

    def sent_to_rel_mask(self, sent_mask):
        last_points = torch.sum(sent_mask, dim=1).unsqueeze(1) - 1
        zero_src = torch.zeros_like(last_points).to(sent_mask.device)
        rel_mask = sent_mask.scatter(dim=1, index=last_points, src=zero_src.int())

        return rel_mask

    def forward(
        self,
        sent_vectors,
        sent_mask,
        sent_rel_ids,
        labels=None,
        flag="Train"
    ):
        # for sent
        if sent_vectors.dtype == torch.bfloat16:
            sent_vectors = sent_vectors.float()
        sent_vectors = self.dropout(sent_vectors)
        sent_vectors = self.sent_proj(sent_vectors)
        sent_with_pe = self.abs_position_embedding(sent_vectors)
        sent_output = self.sent_transformer(sent_with_pe, sent_mask)

        # for rel
        rel_vectors = self.rel_embedding(sent_rel_ids)
        rel_vectors = self.dropout(rel_vectors)
        rel_vectors = self.rel_proj(rel_vectors)
        rel_with_pe = self.abs_position_embedding(rel_vectors)
        rel_mask = self.sent_to_rel_mask(sent_mask)
        rel_output = self.rel_transformer(rel_with_pe, rel_mask)

        output = torch.cat((sent_output, rel_output), dim=1)
        output = self.dropout(output)
        output = self.fc(output)
        output = self.dropout(output)
        logits = self.classifier(output)

        _, preds = torch.max(logits, dim=-1)
        outputs = (preds,)
        if flag.upper() == 'TRAIN':
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

class TransformerSentRelFusion(nn.Module):
    def __init__(self, args):
        super(TransformerSentRelFusion, self).__init__()

        self.rel_embedding = Embedding(
            vocab=args.rel_list,
            embed_dim=args.embed_dim,
            embed_file=args.embed_file,
            saved_file=args.saved_embed_file
        )
        self.num_labels = args.num_labels
        self.input_size = args.input_size
        self.embed_dim = args.embed_dim
        self.hidden_size = args.hidden_size
        self.max_sent_len = args.max_sent_num
        self.dropout = args.dropout

        self.sent_proj = nn.Linear(self.input_size, self.hidden_size)
        self.rel_proj = nn.Linear(self.embed_dim, self.hidden_size)
        self.rel_position_embedding = Relative_Position_Embedding(self.hidden_size, self.max_sent_len*2)
        self.fusion_transformer = Transformer_Encoder_Fusion(
            {"num_layers": 1, "hidden_size": args.hidden_size,
             "num_heads": args.num_heads, "scaled": args.scaled}
        )
        self.fc = nn.Linear(self.hidden_size, self.hidden_size // 4)
        self.classifier = nn.Linear(args.hidden_size // 4, self.num_labels)
        self.dropout = nn.Dropout(p=args.dropout)
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()

    def sent_to_rel_mask(self, sent_mask):
        last_points = torch.sum(sent_mask, dim=1).unsqueeze(1) - 1
        zero_src = torch.zeros_like(last_points).to(sent_mask.device)
        rel_mask = sent_mask.scatter(dim=1, index=last_points, src=zero_src.int())

        return rel_mask

    def merge(self, sent_vectors, rel_vectors):
        """merge two input, to produce new vector, mask, and relative position"""
        batch_size, sent_len = sent_vectors.size()[:2]
        rel_len = rel_vectors.size(1)

        ## 1. concat vectors, we don't need to make the real rel vectors close to
        # the real sent vectors since we don't use abslolute embedding
        # [batch, sent_len+rel_len, dim]
        merge_vectors = torch.cat((sent_vectors, rel_vectors), dim=1)

        ## 2.relative postion, [batch_size, sent_len+rel_len]
        sent_pos = torch.arange(0, sent_len).repeat(batch_size, 1).to(sent_vectors.device)
        rel_start_pos = torch.arange(0, rel_len).repeat(batch_size, 1).to(sent_vectors.device)
        rel_end_pos = torch.arange(1, rel_len+1).repeat(batch_size, 1).to(sent_vectors.device)
        merge_start_pos = torch.cat((sent_pos, rel_start_pos), dim=1)
        merge_end_pos = torch.cat((sent_pos, rel_end_pos), dim=1)

        return merge_vectors, merge_start_pos, merge_end_pos

    def forward(
        self,
        sent_vectors,
        sent_mask,
        sent_rel_ids,
        sent_rel_mask,
        labels=None,
        flag="Train"
    ):
        # for sent
        if sent_vectors.dtype == torch.bfloat16:
            sent_vectors = sent_vectors.float()
        sent_vectors = self.dropout(sent_vectors)
        sent_vectors = self.sent_proj(sent_vectors)

        # for rel
        rel_vectors = self.rel_embedding(sent_rel_ids)
        rel_vectors = self.dropout(rel_vectors)
        rel_vectors = self.rel_proj(rel_vectors)

        # fusion
        input_vectors, pos_start, pos_end = self.merge(
            sent_vectors, rel_vectors
        )
        rel_pos_input = self.rel_position_embedding(pos_start, pos_end)
        rel_mask = self.sent_to_rel_mask(sent_mask)
        zero_mask = torch.zeros_like(rel_mask).to(sent_mask.device)
        concat_mask = torch.cat((sent_mask, rel_mask), dim=1)
        expand_sent_mask = torch.cat((sent_mask, zero_mask), dim=1)
        output = self.fusion_transformer(
            hidden_states=input_vectors,
            attention_mask=expand_sent_mask,
            fusion_mask=sent_rel_mask,
            # fusion_mask=concat_mask,
            rel_pos_input=rel_pos_input
        )

        output = self.dropout(output)
        output = self.fc(output)
        output = self.dropout(output)
        logits = self.classifier(output)

        _, preds = torch.max(logits, dim=-1)
        outputs = (preds,)
        if flag.upper() == 'TRAIN':
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs