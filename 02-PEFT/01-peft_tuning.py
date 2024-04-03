#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/26 16:41
# @Author  : yj.wang
# @File    : chatbot_prefix_tuning.py
# @Describe: 模型微调实战代码整理
# -*- coding: UTF-8 -*-
# 导入相关包
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

tuning_function_list = ["bitfit", "soft_prompt_tuning", "hard_prompt_tuning", "ptuning", "prefix_tuning"]


def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        "\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ")
    response = tokenizer(example["output"] + tokenizer.eos_token)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:  # 这边就是做一下截断操作
        input_ids = input_ids[: MAX_LENGTH]
        attention_mask = attention_mask[: MAX_LENGTH]
        labels = labels[: MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def get_model_params(model):
    model_params = 0
    for param in model.parameters():
        model_params += param.numel()
    return model_params


def use_bitfit(model):
    num_param = 0
    for name, param in model.named_parameters():
        if "bias" not in name:
            param.requires_grad = False
        else:
            num_param += param.numel()
    print("use bitfit trained params: ", num_param)


def use_soft_prompt_tuning(model):
    from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit
    config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10)
    model = get_peft_model(model, config)
    print(model.print_trainable_parameters())
    return model


def use_hard_prompt_tuning(model):
    from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit
    # 需要初始化提示词，并且传入tokenizer的长度
    config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM,
                                prompt_tuning_init=PromptTuningInit.TEXT,
                                prompt_tuning_init_text="下面是一段人与机器人的对话：",
                                num_virtual_tokens=len(tokenizer("下面是一段人与机器人的对话：")["input_ids"]),
                                tokenizer_name_or_path="./tokenizer/Langboat/bloom-389m-zh")
    model = get_peft_model(model, config)
    print(model.print_trainable_parameters())
    return model


def use_ptuning(model):
    from peft import PromptEncoderConfig, TaskType, get_peft_model, PromptEncoderReparameterizationType
    # 默认 encoder_reparameterization_type=<PromptEncoderReparameterizationType.MLP: 'MLP'>
    config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10,
                                 encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM,
                                 encoder_dropout=0., encoder_num_layers=1, encoder_hidden_size=1024)
    model = get_peft_model(model, config)
    return model


def use_prefix_tuning(model):
    from peft import PrefixTuningConfig, get_peft_model, TaskType
    config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM,
                                num_virtual_tokens=10,
                                prefix_projection=True)  # 多加了两层全连接层
    model = get_peft_model(model, config)
    print(model.prompt_encoder)
    print(model.print_trainable_parameters())
    return model


def use_lora(model):
    from peft import LoraConfig, TaskType, get_peft_model
    config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                        target_modules=["query_key_value"],  # target_modules指定哪些位置使用lora的方法
                        modules_to_save=[
                            "word_embeddings"])  # modules_to_save除了lora的位置需要训练，模型还有哪些地方也指定可以训练，如果不指定该参数，默认模型不训练
    model = get_peft_model(model, config)
    return model


def use_IA3(model):
    from peft import IA3Config, TaskType, get_peft_model
    config = IA3Config(task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, config)  # 根据论文，使用该方法，将学习率调整到3e-3这时候收敛效果比较好
    return model


def increment_path(prefix="chatbot", tuning_name="lora", sep='_'):
    import os
    path = f'{prefix}{sep}{tuning_name}'
    if not os.path.exists(path):
        return path
    # Method 1
    for n in range(2, 9999):
        path = f'{prefix}{sep}{tuning_name}{sep}{n}'  # increment path
        if not os.path.exists(path):  #
            break
    return path


if __name__ == '__main__':
    tuning_function = "prefix_tuning"
    # 1、加载数据集
    ds = Dataset.load_from_disk("./alpaca_data_zh/")  # 这个好像本地就有
    # 数据预处理
    tokenizer = AutoTokenizer.from_pretrained("./tokenize/Langboat/bloom-389m-zh")
    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
    # 创建模型
    model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-389m-zh", low_cpu_mem_usage=True)

    if tuning_function in tuning_function_list:
        func_name = "use_" + tuning_function + "(model)"
        model = eval(func_name)
    else:
        print("please input a correct tuning function name")

    # output_dir = "./chatbot_" + tuning_function  # 使用yolov5的文件夹的命名方式
    output_dir = increment_path(prefix="./chatbot_", tuning_name=tuning_function, sep="_")  # 使用yolov5的文件夹的命名方式
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        logging_steps=10,
        num_train_epochs=4,
        logging_dir="./train_logs"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    trainer.train()

    #
    # use_ptuning()
    # TODO 然后上传github上， 然后整理一下，学习一下
