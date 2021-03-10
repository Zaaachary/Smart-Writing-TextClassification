#! -*- encoding:utf-8 -*-
"""
@File    :   data_processor.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   swtc
"""
import os
import json
import pdb

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from swtc_task.example import SWTCExample


class Base_Processor(object):

    def __init__(self, data_dir, dataset_type, content_max_len=200):
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.content_max_len = content_max_len
        self.data = None
        self.config = None

    def load_swtc(self):
        data = []

        # load raw data
        f = open(os.path.join(self.data_dir, 'swtc', f"smartwrite_{self.dataset_type}.json"), 'r', encoding='utf-8')
        data = json.load(f)
        f.close()

        # load config
        f = open(os.path.join(self.data_dir, 'swtc', 'config.json'), 'r', encoding='utf-8')
        config = json.load(f)
        f.close()
        self.config = config

        # convert raw data 2 SWTCExample
        for index, case in enumerate(data):
            example = SWTCExample.load_from_dict(case, config['label2id'], self.content_max_len)
            data[index] = example

        self.data = data

    def make_dataloader(self, tokenizer, batch_size, drop_last, max_seq_len, shuffle=True):
        T, L = [], []

        for example in tqdm(self.data):
            text, label = example.tokenize(tokenizer, max_seq_len)
            T.append(text)
            L.append(label)
        
        self.data = (T, L)  # len(T) = len(L)
        return self._convert_to_tensor(batch_size, drop_last, shuffle)

    def _convert_to_tensor(self, batch_size, drop_last, shuffle):
        tensors = []
        features = self.data[0]     # tensor, label
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        # features
        # tensors.extend((all_input_ids, all_input_mask, all_segment_ids))
        tensors.extend((all_input_ids, all_input_mask, all_segment_ids))
        
        # labels
        tensors.append(torch.tensor(self.data[1], dtype=torch.long))

        dataset = TensorDataset(*tensors)
        sampler = RandomSampler(dataset) if shuffle else None
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=drop_last)

        return dataloader
