#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/7 11:37
# @Author  : yj.wang
# @File    : 01-dp.py
# @Describe: 这个是数据并行的学习教程
# 提升数据并行速度的方式：1、增加batch size，对于nn.DataParallel你设置的batch size会被平均分配到多张显卡里面，相当于每一张显卡的batch_size_per_device = batch_size / n
# 使用transformers中的trainer，会自动判断是否可以使用nn.DataParallel数据并行的方式。
# -*- coding: UTF-8 -*-

from transformers import BertTokenizer, BertForSequenceClassification
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
trainset, validset = random_split(dataset, lengths=[0.9, 0.1])

# 创建dataloader
import torch
tokenizer = BertTokenizer.from_pretrained("hfl/rbt3")
def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs['labels'] = torch.tensor(labels)
    return inputs

from torch.utils.data import DataLoader
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate_func)
validloader = DataLoader(validset, batch_size=64, shuffle=False, collate_fn=collate_func)


# 创建模型和优化器
from torch.optim import Adam
model = BertForSequenceClassification.from_pretrained("hfl/rbt3")
if torch.cuda.is_available():
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=None)
    print(model.device_ids)
optimizer = Adam(model.parameters(), lr=2e-5)


# 评估和训练
def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}

            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch['labels'].long()).float().sum()
    return acc_num / len(validset)


def train(epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            optimizer.zero_grad()  # 这里为什么需要zero_grad()
            output = model(**batch)
            # new code
            # print(output.loss)  # 这里的loss会有两个，因此要做一下汇总，变成一个
            # output.loss.mean().backward()  # 数据并行的情况
            output.loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                # print(f"ep: {ep}, global_step: {global_step}, loss: {output.loss.mean().item()}")  # 数据并行的情况
                print(f"ep: {ep}, global_step: {global_step}, loss: {output.loss.item()}")
            global_step += 1
        acc = evaluate()
        print(f"ep: {ep}, acc: {acc}")

train()

if __name__ == '__main__':
    pass
