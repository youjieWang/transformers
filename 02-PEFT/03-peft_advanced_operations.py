#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/2 9:41
# @Author  : yj.wang
# @File    : peft_advanced_operations.py
# @Describe: None
# -*- coding: UTF-8 -*-
import torch
from torch import nn
from peft import LoraConfig, get_peft_model, PeftModel

# 自定义模型适配
net1 = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)


def get_adapter_model(net1):
    config = LoraConfig(target_modules=["0"])
    model1 = get_peft_model(net1, config)
    model1.save_pretrained("./loraA")

    config1 = LoraConfig(target_modules=["2"])
    model1 = get_peft_model(net1, config1)
    model1.save_pretrained(("./loraB"))


if __name__ == '__main__':
    print(net1)
    model1 = PeftModel.from_pretrained(net1, model_id="./loraA", adapter_name="loraA")
    print(model1)
    model1.load_adapter("./loraB", adapter_name="loraB")
    print(model1)
    print(model1.active_adapter)  # 查看当前加载的是哪一个adapter

    # 调用模型
    res1 = model1(torch.arange(0, 10).view(1, 10).float())
    print(res1)

    # 打印模型权重
    for name, param in model1.named_parameters():
        if name in ["base_model.model.0.lora_A.loraA.weight", "base_model.model.0.lora_B.loraA.weight"]:
            param.data = torch.ones_like(param)
    print(model1(torch.arange(0, 10).view(1, 10).float()))

    # 切换适配器
    print("----change adapter----")
    model1.set_adapter("loraB")
    print(model1.active_adapter)
    print(model1)
    print(model1(torch.arange(0, 10).view(1, 10).float()))

    # 如果想要获取原始模型的输出，需要禁用适配器
    print("forbid adapter")
    model1.set_adapter("loraA")
    print(model1.active_adapter)
    print(model1(torch.arange(0, 10).view(1, 10).float()))
    with model1.disable_adapter():  # 这句话就是可以直接禁用适配器
        print(model1(torch.arange(0, 10).view(1, 10).float()))

