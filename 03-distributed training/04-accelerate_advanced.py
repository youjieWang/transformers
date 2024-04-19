#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/7 11:37
# @Author  : yj.wang
# @File    : 03-acclerate.py
# @Describe: 这个是accelerate的学习教程

# -*- coding: UTF-8 -*-
import os
import math
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
def evaluate(model, validloader, accelerator):
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            # if torch.cuda.is_available():
            #     batch = {k: v.to(int(os.environ['LOCAL_RANK'])) for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            pred, refs = accelerator.gather_for_metrics((pred, batch['labels']))
            acc_num += (pred.long() == refs.long()).float().sum()
    # dist.all_reduce(acc_num)  # 这边默认是的op是dist.ReduceOp.SUM
    return acc_num / len(validloader.dataset)


def train(model, optimizer, trainloader, validloader, accelerator, resume,  epoch=10, log_step=10):
    global_step = 0
    
    resume_step = 0
    resume_epoch = 0

    # 如果resume不为空，说明需要做断点续训
    if resume is not None:
        accelerator.load_state(resume)  # 这个时候需要把之前的sava_state的东西加载起来
        # 接下来还需要知道我们当前训练到达哪一步了
        # 1、计算一个epoch需要迭代多少步 = 总的数据长度 / 梯度累积的步数（也就是多少个梯度batch进行一次梯度累积）
        # 如果这里是一个batch进行一次梯度累积的话，那么一个epoch需要迭代多少步 = 总的数据长度 / batch_size
        steps_per_epoch = math.ceil(len(trainloader) / accelerator.gradient_accumulation_steps)
        # 2、记录当前是多少步
        resume_step = global_step = int(resume.split("step_")[-1])
        # 3、计算当前训练到第几轮了 = 当前的步数 / 每一轮需要迭代多少步
        resume_epoch = resume_step // steps_per_epoch  # 向下取整
        # 4、计算当前这一轮里面有多少步已经训练完了 = 记录当前是多少步 - (当前已经训练的轮数 * 每一轮走多少步)
        resume_step -= resume_epoch * steps_per_epoch
        accelerator.print(f"resume from checkpoint -> {resume}")
    for ep in range(resume_epoch, epoch):
        model.train()
        # trainloader.sampler.set_epoch(ep)  # 这样子设置的话，在每一轮训练的时候都会将数据打乱。相当于shuffle
        if resume and ep == resume_epoch and resume_step != 0:
            # 跳过多少轮需要乘上一个当前步数 * 一个梯度累计的
            activate_dataloader = accelerator.skip_first_batches(trainloader, resume_step * accelerator.gradient_accumulation_steps)
        else:
            # 如果不是上述情况就让他等于原始的trainloader就好了
            activate_dataloader = trainloader
        for batch in trainloader:
            with accelerator.accumulate(model):
            # if torch.cuda.is_available():
            #     batch = {k: v.to(int(os.environ['LOCAL_RANK'])) for k, v in batch.items()}
                optimizer.zero_grad()  # 这里为什么需要zero_grad()
                output = model(**batch)
                loss = output.loss
                accelerator.backward(loss)
                optimizer.step()
                if accelerator.sync_gradients:
                    global_step += 1
                    if global_step % log_step == 0:
                        # dist.all_reduce(loss, op=dist.ReduceOp.AVG)  # 这边使用all_reduce进行各个进程之间的数据汇总和运算操作，这就使得各个进程上的运算结果是一致的
                        loss = accelerator.reduce(loss, "mean")
                        accelerator.print(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
                        accelerator.log({"loss": loss.item()}, global_step)
                    if global_step % 50 ==0 and global_step != 0:
                        accelerator.print(f"save checkpoint -> step_{global_step}")
                        accelerator.save_state(accelerator.project_dir + f"/step_{global_step}")
                        # accelerator.save_model(model, accelerator.project_dir + f"/step_{global_step}")
                        accelerator.unwrap_model(model).save_pretrained(save_directory=accelerator.project_dir + f"/step_{global_step}/model", is_main_process=accelerator.is_main_process, state_dict=accelerator.get_state_dict(model), save_function=accelerator.save)
        acc = evaluate(model, validloader, accelerator)
        accelerator.print(f"ep: {ep}, acc: {acc}")
        accelerator.log({"acc": acc}, global_step)
    accelerator.end_training()



def main():
    # 初始化任务的进程组
    # dist.init_process_group(backend="nccl")  # 这里只需要知道linux上使用多进程就指定nccl就可以了
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=2, log_with="tensorboard", project_dir="ckpts")
    accelerator.init_trackers(project_name="runs")
    trainloader, validloader = prepare_dataloader()
    model, optimizer = prepare_model_and_optimizer()
    model, optimizer, trainloader, validloader = accelerator.prepare(model, optimizer, trainloader, validloader)
    train(model, optimizer, trainloader, validloader, accelerator, resume="ckpts/step_100")


if __name__ == '__main__':
    main()  # 运行代码torchrun --nproc_per_node=x 03-accelerate.py
