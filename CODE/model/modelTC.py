#! -*- encoding:utf-8 -*-
"""
@File    :   modelTC.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   分类模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel


class BertForTC(BertPreTrainedModel):

    def __init__(self, config):
        
        super(BertForTC, self).__init__(config)
        self.config = config

        self.bert = BertModel(config)

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, 20)
        )

        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        """
        input
        - input_ids: [B, L]
        - labels: [B, 20]
        output
        - loss: CELoss
        - right_num
        """
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooler_output = outputs.pooler_output

        logits = self.classifier(pooler_output)
        loss = F.cross_entropy(logits, labels)      # get the CELoss

        with torch.no_grad():
            logits = F.softmax(logits, dim=1)       # get the score
            predicts = torch.argmax(logits, dim=1)  # find the result
            right_num = torch.sum(predicts == labels)

        return loss, right_num

    def predict(self, input_ids, attention_mask, token_type_ids):
        """
        return: [B, 22]
        """
        return self.forward(input_ids, attention_mask, token_type_ids)