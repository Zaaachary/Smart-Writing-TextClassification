from .common import Vn
from .common import mkdir_if_notexist

import torch
from . import logger
from transformers.optimization import AdamW
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers.file_utils import WEIGHTS_NAME, CONFIG_NAME

import os
from tqdm.autonotebook import tqdm


class BaseTrainer:
    """
    训练模型的基本流程

    1. self.train(...)
    2. self.evaluate(...)
    3. self.set_optimizer(optimizer)
    4. self.set_scheduler(scheduler)
    5. self.make_optimizer(...)
    6. self.make_scheduler(...)
    7. self.save_model()
    8. cls.load_model(...)

    需要针对不同的任务 重写
    9. self._report()
    0. self._forward()
    """
    def __init__(self, model, multi_gpu, device, print_step, output_model_dir, v_num):
        """
        device: 主device
        multi_gpu: 是否使用了多个gpu
        v_num: 显示的变量数
        """
        self.model = model.to(device)
        self.device = device
        self.multi_gpu = multi_gpu
        self.print_step = print_step
        self.output_model_dir = output_model_dir
        self.v_num = v_num
        self.train_record = Vn(v_num)

    def train(self, epoch_num, train_dataloader, dev_dataloader,
              save_last=True):
        """
        save_last: 直到最后才保存模型，否则保存验证集上loss最低的模型
        """

        best_dev_loss = float('inf')
        best_dev_acc = 0
        self.global_step = 0
        self.train_record.init()
        self.model.zero_grad()

        for epoch in range(int(epoch_num)):
            print(f'---- Epoch: {epoch+1:02} ----')
            for step, batch in enumerate(tqdm(train_dataloader, desc='Train')):
                self.model.train()
                self._step(batch)
                if self.global_step % self.print_step == 0:

                    self._report(self.train_record, mode='single')
                    self.train_record.init()

                    # if not save_last and (dev_record.avg()[0] < best_dev_loss):
                    #     best_dev_loss = dev_record.avg()[0]
                    #     self.save_model()
                    # print("current_acc is {}".format(current_acc))
                    # print("best_dev_acc is {}".format(best_dev_acc))
                    # self.save_model()

            # 要放在epoch 一层
            dev_record = self.evaluate(dev_dataloader)
            current_acc = dev_record.list()[1]
            self.model.zero_grad()

            if not save_last and current_acc > best_dev_acc:
                best_dev_acc = current_acc
                self.save_model()

            self._report(self.train_record, dev_record)

        if save_last:
            self.save_model()

    def _forward(self, batch, record):
        """
        rewrite accroding to actual situation
        """
        batch = tuple(t.to(self.device) for t in batch)
        loss, acc = self.model(*batch)
        loss, acc = self._mean((loss, acc))
        record.inc([loss.item(), acc.item()])
        return loss

    def _mean(self, tuples):
        """
        vars 需要是元组
        """
        if self.multi_gpu:
            return tuple(v.mean() for v in tuples)
        return tuples

    def _step(self, batch):
        loss = self._forward(batch, self.train_record)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1)  # max_grad_norm = 1

        self.optimizer.step()
        self.scheduler.step()
        self.model.zero_grad()
        self.global_step += 1

    def evaluate(self, dataloader, desc='Eval'):
        record = Vn(self.v_num)

        # for batch in tqdm(dataloader, desc, miniters=10):
        for batch in tqdm(dataloader):
            self.model.eval()
            with torch.no_grad():
                self._forward(batch, record)

        return record

    def _report(self, train_record, devlp_record):
        '''
        rewrite accroding to actual situation
        '''
        tloss, tacc = train_record.avg()
        dloss, dacc = devlp_record.avg()
        print("\t\tTrain loss %.4f acc %.4f | Dev loss %.4f acc %.4f" % (
                tloss, tacc, dloss, dacc))

    def make_optimizer(self, weight_decay, lr):
        params = list(self.model.named_parameters())

        no_decay_keywords = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        def _no_decay(n):
            return any(nd in n for nd in no_decay_keywords)

        parameters = [
            {'params': [p for n, p in params if _no_decay(n)], 'weight_decay': 0.0},
            {'params': [p for n, p in params if not _no_decay(n)],
             'weight_decay': weight_decay}
        ]

        optimizer = AdamW(parameters, lr=lr, eps=1e-8)
        return optimizer

    def make_scheduler(self, optimizer, warmup_proportion, t_total):
        return get_cosine_with_hard_restarts_schedule_with_warmup(
          optimizer, num_warmup_steps=warmup_proportion * t_total,
          num_training_steps=t_total)

    def save_model(self):
        mkdir_if_notexist(self.output_model_dir)
        logger.info('保存模型 {}'.format(self.output_model_dir))
        # self.model.save_pretrained(self.output_model_dir)

        output_model_file = os.path.join(self.output_model_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(self.output_model_dir, CONFIG_NAME)
        torch.save(self.model.state_dict(), output_model_file)
        self.model.config.to_json_file(output_config_file)
        # tokenizer.save_vocabulary(output_dir)

    # TODO
    @classmethod
    def load_model(cls, ConfigClass, ModelClass,
                   multi_gpu, device, print_step, output_model_dir,fp16,  **params):

        output_model_file = os.path.join(output_model_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_model_dir, CONFIG_NAME)
        cls.fp16 = fp16

        print('load_model', output_model_file, output_config_file)

        config = ConfigClass(output_config_file)

        model = ModelClass(config, **params)
        model.load_state_dict(torch.load(output_model_file))
        model.to(device)

        return cls(model, multi_gpu, device, print_step,
                   output_model_dir)

    def set_optimizer(self, optimizer):
        if self.fp16:
            model, optimizer = amp.initialize(self.model, optimizer, opt_level='O1')
            self.model = model
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler