#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/7 11:37
# @Author  : yj.wang
# @File    : 03-acclerate.py
# @Describe: 这个是accelerate的学习教程

# -*- coding: UTF-8 -*-
import os
import torch
import pandas as pd
import torch.distributed as dist

from torch.optim import Adam
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler  # 这个采样器可以在不同的GPU之间采样不同的数据了，下面的DataLoader就不需要shuffle了
from torch.nn.parallel import DistributedDataParallel as DDP  # 使用分布式需要将模型包装起来
from transformers import BertTokenizer, BertForSequenceClassification


# 创建dataset
class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]['review'], self.data.iloc[index]['label']

    def __len__(self):
        return len(self.data)


def prepare_dataloader():
    dataset = MyDataset()
    # 数据集划分
    trainset, validset = random_split(dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))
    # 创建dataloader
    tokenizer = BertTokenizer.from_pretrained("hfl/rbt3")

    def collate_func(batch):
        texts, labels = [], []
        for item in batch:
            texts.append(item[0])
            labels.append(item[1])
        inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        inputs['labels'] = torch.tensor(labels)
        return inputs

    trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_func, shuffle=True)
    validloader = DataLoader(validset, batch_size=64, collate_fn=collate_func, shuffle=False)
    return trainloader, validloader


def prepare_model_and_optimizer():
    # 创建模型和优化器
    model = BertForSequenceClassification.from_pretrained("hfl/rbt3")
    optimizer = Adam(model.parameters(), lr=2e-5)
    return model, optimizer


def print_rank_0(info):
    if int(os.environ["RANK"]) == 0:  # 这里的RANK就是指的是全局的rank，如果特定的LOCAL_RANK就是每个机器的每张显卡的id
        print(info)


# 评估和训练
def evaluate(model, validloader):
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
    return acc_num / len(validloader.dataset)


def train(model, optimizer, trainloader, validloader, accelerator, epoch=10, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        # trainloader.sampler.set_epoch(ep)  # 这样子设置的话，在每一轮训练的时候都会将数据打乱。相当于shuffle
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.to(int(os.environ['LOCAL_RANK'])) for k, v in batch.items()}
            optimizer.zero_grad()  # 这里为什么需要zero_grad()
            output = model(**batch)
            loss = output.loss
            accelerator.backward(loss)
            optimizer.step()
            if global_step % log_step == 0:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)  # 这边使用all_reduce进行各个进程之间的数据汇总和运算操作，这就使得各个进程上的运算结果是一致的
                print_rank_0(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
            global_step += 1
        acc = evaluate(model, validloader)
        print_rank_0(f"ep: {ep}, acc: {acc}")


def main():
    # 初始化任务的进程组
    # dist.init_process_group(backend="nccl")  # 这里只需要知道linux上使用多进程就指定nccl就可以了
    accelerator = Accelerator()
    trainloader, validloader = prepare_dataloader()
    model, optimizer = prepare_model_and_optimizer()
    model, optimizer, trainloader, validloader = accelerator.prepare(model, optimizer, trainloader, validloader)
    train(model, optimizer, trainloader, validloader, accelerator)


if __name__ == '__main__':
    main()  # 运行代码torchrun --nproc_per_node=x 03-accelerate.py
