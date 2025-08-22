# author = liuwei
# date = 2023-12-30

"""
Our parser is based on Roberta-large
"""

import math
import os
import json

import numpy as np
import torch
import random
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import gelu
from transformers import PreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaModel

import warnings
warnings.filterwarnings("ignore")


class BaseClassifier(PreTrainedModel):
    def __init__(self, config, args):
        super(BaseClassifier, self).__init__(config)

        self.encoder = RobertaModel.from_pretrained(args.model_name_or_path, config=config)
        self.dropout = nn.Dropout(p=config.HP_dropout)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.num_labels = args.num_labels

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        labels=None,
        flag="Train"
    ):
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooler_output = encoder_output.pooler_output
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)
        probs = F.softmax(logits, dim=-1)
        _, preds = torch.max(logits, dim=-1)
        outputs = (preds, probs)

        if flag.lower() == "train":
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, ) + outputs

        return outputs