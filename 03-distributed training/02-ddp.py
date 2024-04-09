#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/7 11:37
# @Author  : yj.wang
# @File    : 02-ddp.py
# @Describe: 这个是数据并行的学习教程
# 提升数据并行速度的方式：1、增加batch size，对于nn.DataParallel你设置的batch size会被平均分配到多张显卡里面，相当于每一张显卡的batch_size_per_device = batch_size / n
# 使用transformers中的trainer，会自动判断是否可以使用nn.DataParallel数据并行的方式。
# -*- coding: UTF-8 -*-

from transformers import BertTokenizer, BertForSequenceClassification
import torch.distributed as dist
import torch

# 初始化任务的进程组
dist.init_process_group(backend="nccl")  # 这里只需要知道linux上使用多进程就指定nccl就可以了

import pandas as pd

# 创建dataset
from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()


    def __getitem__(self, index):
        return self.data.iloc[index]['review'], self.data.iloc[index]['label']

    def __len__(self):
        return len(self.data)

dataset = MyDataset()

# 数据集划分
from torch.utils.data import random_split
'''
在单进程里跑的时候，一次的数据划分，划分好了就是划分好了，不会出现多个进程划分之后的数据结果不一致。
现在有多个进程，每个进程里面都单独进行了一次数据划分，这就不能保证每个进程里面的数据划分是一致的。
generator需要保证每一个进程内的数据集划分结果是一致的，如果数据集划分不一致，有可能有些数据在有些进程里面是训练集，在某些进程里面是验证集，
就会存在最终的训练结果精度虚高。
'''
trainset, validset = random_split(dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))

# 创建dataloader

tokenizer = BertTokenizer.from_pretrained("./model/")
def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs['labels'] = torch.tensor(labels)
    return inputs

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler  # 这个采样器可以在不同的GPU之间采样不同的数据了，下面的DataLoader就不需要shuffle了
# trainloader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate_func)
# validloader = DataLoader(validset, batch_size=64, shuffle=False, collate_fn=collate_func)
trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_func, sampler=DistributedSampler(trainset))
validloader = DataLoader(validset, batch_size=64, collate_fn=collate_func, sampler=DistributedSampler(validset))




# 创建模型和优化器
from torch.optim import Adam
import os
from torch.nn.parallel import DistributedDataParallel as DDP  # 使用分布式需要将模型包装起来
model = BertForSequenceClassification.from_pretrained("./model/")
if torch.cuda.is_available():
    model = model.to(int(os.environ['LOCAL_RANK']))
    model = DDP(model)

optimizer = Adam(model.parameters(), lr=2e-5)


def print_rank_0(info):
    if int(os.environ["RANK"]) == 0:  # 这里的RANK就是指的是全局的rank，如果特定的LOCAL_RANK就是每个机器的每张显卡的id
        print(info)

# 评估和训练
def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.to(int(os.environ['LOCAL_RANK'])) for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch['labels'].long()).float().sum()
    dist.all_reduce(acc_num)  # 这边默认是的op是dist.ReduceOp.SUM
    return acc_num / len(validset)


def train(epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        trainloader.sampler.set_epoch(ep)  # 这样子设置的话，在每一轮训练的时候都会将数据打乱。相当于shuffle
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.to(int(os.environ['LOCAL_RANK'])) for k, v in batch.items()}
            optimizer.zero_grad()  # 这里为什么需要zero_grad()
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)  # 这边使用all_reduce进行各个进程之间的数据汇总和运算操作，这就使得各个进程上的运算结果是一致的
                print_rank_0(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
            global_step += 1
        acc = evaluate()
        print_rank_0(f"ep: {ep}, acc: {acc}")

train()

if __name__ == '__main__':
    pass
