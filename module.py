# author = liuwei
# date = 2024-01-02
import math
import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init, functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from utils import reverse_padded_sequence, build_embedding_of_corpus, random_embedding_of_corpus

from transformers.activations import gelu


class BiLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BiLSTM, self).__init__()
        self.f_lstm = nn.LSTM(input_size, hidden_size // 2, num_layers=num_layers, batch_first=True)
        self.b_lstm = nn.LSTM(input_size, hidden_size // 2, num_layers=num_layers, batch_first=True)

    def forward(self, inputs, sent_mask):
        """
        Args:
            inputs: [batch_size, seq_length, dim_size]
            seq_lengths: [batch_size]
        """
        seq_lengths = torch.sum(sent_mask, dim=-1)
        lengths = list(map(int, seq_lengths))
        reversed_inputs = reverse_padded_sequence(inputs, lengths)

        f_lstm_out, f_hidden = self.f_lstm(inputs)
        b_lstm_out, b_hidden = self.b_lstm(reversed_inputs)
        # get the last token state
        b_lstm_out = reverse_padded_sequence(b_lstm_out, lengths)
        all_states = torch.cat((f_lstm_out, b_lstm_out), dim=-1)
        f_lstm_out = reverse_padded_sequence(f_lstm_out, lengths)
        f_last_state = f_lstm_out[:, 0, :].squeeze()
        b_last_state = b_lstm_out[:, 0, :].squeeze()
        lstm_out = torch.cat((f_last_state, b_last_state), dim=-1)

        expand_sent_mask = sent_mask.unsqueeze(2)  # [N, S, 1]
        masked_lstm_states = all_states * expand_sent_mask  # [N, S, D]
        avg_lstm_out = torch.sum(masked_lstm_states, dim=1) / seq_lengths.unsqueeze(1)  # [N, D]

        return lstm_out, avg_lstm_out


class Embedding(torch.nn.Module):
    def __init__(self, vocab, embed_dim, embed_file, saved_file):
        super(Embedding, self).__init__()

        self.vocab = vocab
        self.embed_dim = embed_dim
        self.embed_file = embed_file
        self.saved_file = saved_file
        self.embedding = nn.Embedding(len(vocab), embed_dim)

        self.init_embedding()

    def init_embedding(self):
        if self.embed_file is None or self.embed_file == "":
            print("Random Embedding-----------")
            corpus_embed = random_embedding_of_corpus(self.vocab, self.embed_dim)
        else:
            print("Pretrained Embedding++++++++++++")
            corpus_embed = build_embedding_of_corpus(self.embed_file, self.vocab, self.embed_dim, self.saved_file)
        assert corpus_embed.shape[0] == len(self.vocab), (corpus_embed.shape[0], len(self.vocab))
        assert corpus_embed.shape[1] == self.embed_dim, (corpus_embed.shape[1], self.embed_dim)

        self.embedding.weight.data.copy_(torch.from_numpy(corpus_embed))

    def forward(self, input_ids):
        return self.embedding(input_ids)


class Absolute_Position_Embedding(nn.Module):
    def __init__(self, hidden_size, learnable, max_seq_len=5000):
        """
        Args:
            hidden_size:
            learnable:
            max_len:
        """
        super(Absolute_Position_Embedding, self).__init__()

        def get_embedding(max_seq_len, embed_dim):
            pe = torch.zeros(max_seq_len, embed_dim)
            position = torch.arange(0, max_seq_len).unsqueeze(1)
            div_term = torch.exp(
                (torch.arange(0, embed_dim, 2, dtype=torch.float)) *
                -(math.log(10000) / embed_dim)
            )
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
            pe = pe.unsqueeze(0)

            return pe

        self.hidden_size = hidden_size
        pe = get_embedding(max_seq_len, hidden_size)
        self.pe = nn.Parameter(pe, requires_grad=learnable)

    def forward(self, inp):
        batch = inp.size(0)
        output = inp + self.pe[:, :inp.size(1)]

        return output


class Multi_Head_Attention(nn.Module):
    """Refer to transformers.BertModel"""

    def __init__(self, hidden_size, num_heads, dropout=0.1, scaled=True):
        super(Multi_Head_Attention, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        assert (self.per_head_size * self.num_heads == self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)

        self.ff_final = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-6)

    def transpose_for_score(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.per_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            key: [batch, key_len, dim]
            value: [batch, value_len, dim]
            query: [batch, query_len, dim]
            mask: [batch, query_len, key_len]
        """
        key = self.key(hidden_states)
        query = self.query(hidden_states)
        value = self.value(hidden_states)

        key = self.transpose_for_score(key)
        value = self.transpose_for_score(value)
        query = self.transpose_for_score(query)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        if self.scaled:
            attention_scores = attention_scores / math.sqrt(self.per_head_size)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                extended_attention_mask = attention_mask[:, None, None, :]
            elif atten_mask.dim() == 3:
                extended_attention_mask = attention_mask[:, None, :, :]
            extended_attention_mask = (1 - extended_attention_mask) * (-1e18)
            attention_scores = attention_scores + extended_attention_mask
        # attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)
        context = context.view(new_context_shape)

        output = self.ff_final(context)
        output = self.dropout(output)
        output = self.LayerNorm(output + hidden_states)

        return output


class Relative_Position_Embedding(nn.Module):
    def __init__(self, hidden_size, max_seq_len):
        super(Relative_Position_Embedding, self).__init__()

        def get_embedding(max_seq_len, embed_dim):
            """for relative position embedding, it starts from -max_seq_len to max_seq_len+1"""
            pe = torch.zeros(max_seq_len * 2 + 1, embed_dim)
            position = torch.arange(-max_seq_len, max_seq_len + 1).unsqueeze(1)
            div_term = torch.exp(
                (torch.arange(0, embed_dim, 2, dtype=torch.float)) *
                -(math.log(10000) / embed_dim)
            )
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)

            return pe

        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        pe = get_embedding(max_seq_len, embed_dim=hidden_size)
        self.pe = nn.Embedding(2 * max_seq_len + 1, hidden_size, _weight=pe)
        self.pos_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, pos_start, pos_end):
        """
        Args:
            pos_start: [batch, seq_len]
            pos_end: [batch, seq_len
        """
        batch_size = pos_start.size(0)
        seq_len = pos_start.size(1)
        pos_ss = pos_start.unsqueeze(-1) - pos_start.unsqueeze(-2) + self.max_seq_len
        pos_se = pos_start.unsqueeze(-1) - pos_end.unsqueeze(-2) + self.max_seq_len
        pos_es = pos_end.unsqueeze(-1) - pos_start.unsqueeze(-2) + self.max_seq_len
        pos_ee = pos_end.unsqueeze(-1) - pos_end.unsqueeze(-2) + self.max_seq_len

        pos_ss = pos_ss.view((batch_size, seq_len, seq_len, -1))
        pos_se = pos_se.view((batch_size, seq_len, seq_len, -1))
        pos_es = pos_es.view((batch_size, seq_len, seq_len, -1))
        pos_ee = pos_ee.view((batch_size, seq_len, seq_len, -1))

        pos_4 = torch.cat([pos_ss, pos_se, pos_es, pos_ee], dim=-1)
        pos_4 = pos_4.view((-1, 4))
        pos_unique, inverse_indices = torch.unique(pos_4, sorted=True, return_inverse=True, dim=0)
        pos_unique_embedding = self.pe(pos_unique)
        pos_unique_embedding = pos_unique_embedding.view((pos_unique_embedding.size(0), -1))
        pos_unique_embedding_fusion = self.pos_fusion(pos_unique_embedding)
        rel_pos_embedding = pos_unique_embedding_fusion[inverse_indices]
        rel_pos_embedding = rel_pos_embedding.view((batch_size, seq_len, seq_len, -1))

        return rel_pos_embedding


class Multi_Head_Attention_Fusion(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1, scaled=True):
        super(Multi_Head_Attention_Fusion, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        assert (self.per_head_size * self.num_heads == self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)

        self.key_rel = nn.Linear(self.hidden_size, self.hidden_size)
        self.u = nn.Parameter(torch.Tensor(self.num_heads, self.per_head_size))
        self.v = nn.Parameter(torch.Tensor(self.num_heads, self.per_head_size))
        nn.init.xavier_normal_(self.u)
        nn.init.xavier_normal_(self.v)

        self.ff_final = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-6)

    def transpose_for_score(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.per_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, rel_pos_embedding):
        """
        Args:
            key: [batch, key_len, dim]
            value: [batch, value_len, dim]
            query: [batch, query_len, dim]
            mask: [batch, query_len, key_len]
            rel_pos_embedding: relative position embedding
        """
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)

        key = self.key(hidden_states)
        query = self.query(hidden_states)
        value = self.value(hidden_states)
        rel_pos_embedding = self.key_rel(rel_pos_embedding)  # [batch, seq_len, seq_len, hidden_size]

        key = self.transpose_for_score(key)
        value = self.transpose_for_score(value)
        query = self.transpose_for_score(query)

        key = key.transpose(-1, -2)  # [batch, num_heads, per_head_size, seq_len]

        ## A, C [batch, num_heads, seq_len, seq_len]
        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)  # [1, num_heads, 1, per_head_size]
        A_and_C = query + u_for_c  # [batch, num_heads, seq_len, per_head_size]
        AC_part = torch.matmul(A_and_C, key)

        ## B
        rel_pos_embedding = torch.reshape(
            rel_pos_embedding,
            [batch_size, seq_len, seq_len, self.num_heads, self.per_head_size]
        )
        # [batch, num_heads, seq_len, head_size, seq_len]
        rel_pos_embedding = rel_pos_embedding.permute(0, 3, 1, 4, 2)
        query_for_b = query.view(
            [batch_size, self.num_heads, seq_len, 1, self.per_head_size]
        )
        B_part = torch.matmul(query_for_b, rel_pos_embedding).squeeze(-2)

        ## D
        v_for_d = self.v.view(1, self.num_heads, 1, 1, self.per_head_size)
        D_part = torch.matmul(v_for_d, rel_pos_embedding).squeeze(-2)

        attention_scores = AC_part + B_part + D_part

        if self.scaled:
            attention_scores = attention_scores / math.sqrt(self.per_head_size)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                extended_attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                extended_attention_mask = attention_mask[:, None, :, :]
            extended_attention_mask = (1 - extended_attention_mask) * (-1e18)
            attention_scores = attention_scores + extended_attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)
        context = context.view(new_context_shape)

        output = self.ff_final(context)
        output = self.dropout(output)
        output = self.LayerNorm(output + hidden_states)

        return output


class Positionwise_FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(Positionwise_FeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden_states = self.w_1(x)
        hidden_states = gelu(hidden_states)
        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + x)

        return hidden_states


class Transformer_Layer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1, scaled=True):
        super(Transformer_Layer, self).__init__()

        self.attention = Multi_Head_Attention(hidden_size, num_heads, dropout, scaled)
        self.feedforward = Positionwise_FeedForward(hidden_size, hidden_size * 4, dropout)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        ffd_output = self.feedforward(attention_output)

        return ffd_output


class Transformer_Layer_Fusion(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1, scaled=True):
        super(Transformer_Layer_Fusion, self).__init__()

        self.attention = Multi_Head_Attention_Fusion(hidden_size, num_heads, dropout, scaled)
        self.feedforward = Positionwise_FeedForward(hidden_size, hidden_size * 4, dropout)

    def forward(self, hidden_states, attention_mask, rel_pos_inputs):
        attention_output = self.attention(hidden_states, attention_mask, rel_pos_inputs)
        ffd_output = self.feedforward(attention_output)

        return ffd_output


class Doc_Pooler(nn.Module):
    def __init__(self, hidden_size):
        super(Doc_Pooler, self).__init__()

        # self.W1 = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.v1 = nn.Parameter(torch.zeros(hidden_size, 1))
        # nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.v1)

    def forward(self, hidden_states, attention_mask):
        batch, sent_len, hidden_size = hidden_states.size()
        attn_weight = torch.matmul(hidden_states, self.v1).squeeze()  # [batch, len]
        extended_attention_mask = (1 - attention_mask) * (-1e18)
        attn_weight = attn_weight + extended_attention_mask
        attn_weight = torch.nn.Softmax(dim=1)(attn_weight).unsqueeze(2)  # [batch, len, 1]
        pooled_states = hidden_states * attn_weight  # [batch, len, dim]
        pooled_states = torch.sum(pooled_states, dim=1)  # [batch, dim]

        return pooled_states


class Transformer_Encoder(nn.Module):
    def __init__(self, params):
        super(Transformer_Encoder, self).__init__()

        self.num_layers = params["num_layers"]
        self.hidden_size = params["hidden_size"]
        self.num_heads = params["num_heads"]
        self.dropout = 0.1
        self.scaled = params["scaled"]
        self.layers = nn.ModuleList(Transformer_Layer(
            self.hidden_size, self.num_heads, self.dropout, self.scaled
        ) for _ in range(self.num_layers))

        self.pooler = Doc_Pooler(self.hidden_size)

    def forward(self, hidden_states, attention_mask):
        for i, layer_module in enumerate(self.layers):
            layer_output = layer_module(
                hidden_states, attention_mask
            )
            hidden_states = layer_output

        pooled_states = self.pooler(hidden_states, attention_mask)

        return pooled_states


class Transformer_Encoder_Fusion(nn.Module):
    def __init__(self, params):
        super(Transformer_Encoder_Fusion, self).__init__()

        self.num_layers = params["num_layers"]
        self.hidden_size = params["hidden_size"]
        self.num_heads = params["num_heads"]
        self.dropout = 0.1
        self.scaled = params["scaled"]
        self.layers = nn.ModuleList(Transformer_Layer_Fusion(
            self.hidden_size, self.num_heads, self.dropout, self.scaled
        ) for _ in range(self.num_layers))

        self.pooler = Doc_Pooler(self.hidden_size)

    def forward(self, hidden_states, attention_mask, fusion_mask, rel_pos_input):
        for i, layer_module in enumerate(self.layers):
            layer_output = layer_module(
                hidden_states, fusion_mask, rel_pos_input
            )
            hidden_states = layer_output

        pooled_states = self.pooler(hidden_states, attention_mask)

        return pooled_states
