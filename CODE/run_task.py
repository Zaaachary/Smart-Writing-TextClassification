#! -*- encoding:utf-8 -*-
"""
@File    :   run_task.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   
"""
import argparse
import pdb
import time

import logging; logging.getLogger("transformers").setLevel(logging.WARNING)
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import torch
from tqdm import tqdm
from transformers import AlbertTokenizer, ElectraTokenizerFast, BertTokenizer, BertConfig

from swtc_task import data_processor
from swtc_task.controller import TextClassification
from swtc_task.trainer import Trainer

from model.modelTC import BertForTC
from utils.common import mkdir_if_notexist, set_seed


def select_tokenizer(args):
    if "bert" in args.PTM_model_vocab_dir:
        return BertTokenizer.from_pretrained(args.PTM_model_vocab_dir)
    else:
        print('tokenizer load error')

def select_task(args):
    if args.task_name == "BERT_MLP":
        return BertForTC, data_processor.Base_Processor

def save_predict(args, logits, predicts):
    pass

def main(args):
    start = time.time()
    set_seed(args)
    print("start in {}".format(start))

    # load data and preprocess
    print("loading tokenizer")
    tokenizer = select_tokenizer(args)
    model, Processor = select_task(args)

    if args.mission == 'train':
        print("loading train set")
        processor = Processor(args.dataset_dir, 'train')
        processor.load_swtc()
        train_dataloader = processor.make_dataloader(tokenizer, args.batch_size, False, 128)

        print('loading dev set')
        processor = Processor(args.dataset_dir, 'dev')
        processor.load_swtc()
        deval_dataloader = processor.make_dataloader(tokenizer, args.batch_size, False, 128)

    else:
        print(f'loading {args.mission} set')
        # file = 'test' if args.mission == 'predict' else 'test'
        file = 'target' if args.mission == 'predict' else 'test'
        processor = Processor(args.dataset_dir, file)
        processor.load_swtc()
        test_dataloader = processor.make_dataloader(tokenizer, args.batch_size, False, 128, False)


    # choose model and initalize controller
    controller = TextClassification(args)
    controller.init(model)

    # run task accroading to mission
    if args.mission == 'train':
        controller.train(train_dataloader, deval_dataloader, save_last=args.save_last)

    elif args.mission == 'test':
        controller.test(test_dataloader)

    elif args.mission == "predict":
        result, predicts = controller.predict(test_dataloader)
        processor.add_predict(predicts)
        processor.save_data(args.pred_file_dir)

        # with open(args.pred_file_name, 'w', encoding='utf-8') as f:
        #     f.write(content)

    end = time.time()
    logger.info("start in {:.0f}, end in {:.0f}".format(start, end))
    logger.info("运行时间:%.2f秒"%(end-start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # other param
    parser.add_argument('--task_name', type=str, default='AlbertAttnMerge')
    parser.add_argument('--mission', type=str, choices=['train','test','predict'])
    parser.add_argument('--fp16', type=int, default=0)
    parser.add_argument('--gpu_ids', type=str, default='-1')
    parser.add_argument('--print_step', type=int, default=250)
    parser.add_argument('--save_last', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)
    # train param
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.1)

    # path param
    parser.add_argument('--dataset_dir', type=str, default='../DATA')
    parser.add_argument('--pred_file_dir', type=str)       # output of predict file
    parser.add_argument('--model_save_dir', type=str, default=None)     # 
    parser.add_argument('--PTM_model_vocab_dir', type=str, default=None)



    args_str = """
    --task_name BERT_MLP
    --batch_size 8
    --lr 1e-5
    --num_train_epochs 3
    --warmup_proportion 0.1
    --weight_decay 0.1
    --fp16 0
    --gpu_ids 0
    --print_step 100
    --mission test
    --pred_file_name  ../DATA/result/task_result.json
    --PTM_model_vocab_dir D:/CODE/Python/Transformers-Models/bert-base-chinese
    --model_save_dir ../DATA/result/TCmodel/
    """
    # args = parser.parse_args(args_str.split())
    args = parser.parse_args()

    # args = parser.parse_args('--batch_size 2 --lr 1e-5 --num_train_epochs 1 --warmup_proportion 0.1 --weight_decay 0.1 --gpu_ids 0 --fp16 0 --print_step 100 --mission train --train_file_name DATA/csqa/train_data.json --dev_file_name DATA/csqa/dev_data.json --test_file_name DATA/csqa/trial_data.json --pred_file_name  DATA/result/task_result.json --model_save_dir DATA/result/model/ --PTM_model_vocab_dir DATA/model/albert-large-v2/'.split())

    print(args)

    main(args)
