#! -*- encoding:utf-8 -*-
"""
@File    :   controller.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   swtc 程序控制器
"""
import torch
from tqdm import tqdm
from utils.common import get_device

from swtc_task.trainer import Trainer

class TextClassification:
    
    def __init__(self, args):
        self.config = args

    def init(self, ModelClass):
        '''
        ModelClass: e.g. modelTC
        '''
        gpu_ids = list(map(int, self.config.gpu_ids.split()))
        multi_gpu = (len(gpu_ids) > 1)
        self.device = get_device(gpu_ids)

        if self.config.mission == 'train':
            model_dir = self.config.PTM_model_vocab_dir
        else:
            model_dir = self.config.model_save_dir
        print('init_model', model_dir)
        model = ModelClass.from_pretrained(model_dir)
        print(model)

        if multi_gpu:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)

        self.trainer = Trainer(
            model, multi_gpu, self.device,
            self.config.print_step, self.config.model_save_dir, self.config.fp16)
        self.model = model

    def train(self, train_dataloader, devlp_dataloader, save_last=True):
        t_total = len(train_dataloader) * self.config.num_train_epochs
        warmup_proportion = self.config.warmup_proportion

        optimizer = self.trainer.make_optimizer(self.config.weight_decay, self.config.lr)
        scheduler = self.trainer.make_scheduler(optimizer, warmup_proportion, t_total)

        self.trainer.set_optimizer(optimizer)
        self.trainer.set_scheduler(scheduler)

        self.trainer.train(
            self.config.num_train_epochs, train_dataloader, devlp_dataloader, save_last=save_last)

    def test(self, dataloader):
        record = self.trainer.evaluate(dataloader)
        test_loss = record[0].avg()
        drn, dan = record.list()[1:]
        print(f"Test: loss {test_loss:.4f}; acc {int(drn)/int(dan):.4f} ({int(drn)}/{int(dan)})")

    def predict(self, dataloader):
        result = []
        predicts = []

        self.model.eval()
        for batch in tqdm(dataloader):
            with torch.no_grad():
                # batch = map(lambda x:x.to(self.device), batch)
                batch = list(map(lambda x:x.to(self.device), batch[:-1]))
                labels, logits = self.model.predict(*batch)

                result.extend(logits.cpu().numpy().tolist())
                predicts.extend(labels.cpu().numpy().tolist())

        return result, predicts

    @classmethod
    def load_from_model(cls, config, ConfigClass, ModelClass):
        gpu_ids = list(map(int, config.gpu_ids.split()))
        multi_gpu = (len(gpu_ids) > 1)
        device = get_device(gpu_ids)

        ctrl = cls(config)
        ctrl.device = device
        ctrl.trainer = Trainer(
            ConfigClass, ModelClass, multi_gpu, device,
            config.print_step, config.output_model_dir, config.fp16)

        return ctrl