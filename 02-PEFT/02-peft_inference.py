#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/27 15:36
# @Author  : yj.wang
# @File    : load_peft_model.py
# @Describe: None
# -*- coding: UTF-8 -*-

from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel

tuning_function_list = ["bitfit", "soft_prompt_tuning", "hard_prompt_tuning", "ptuning", "prefix_tuning"]


def load_lora_model(model, model_id="./chatbot_lora/checkpoint-500/"):
    # 这个其实还是分支时候的lora模型，需要将两个分支进行和并可以加速推理
    p_model = PeftModel.from_pretrained(model, model_id=model_id)
    # 合并模型：将lora分支合并到主分支里面
    merge_model = p_model.merge_and_unload()
    # 保存合并后的lora主模型到本地
    merge_model.save_pretrained("./chatbot/merge_model")
    # 如果保存好合并后的模型，后续在加载的模型的话，就可以通过下面的方式直接加载了
    # model = AutoModelForCausalLM.from_pretrained("./chatbot/merge_model", low_cpu_mem_usage=True)
    return merge_model


if __name__ == '__main__':
    tuning_function = "lora"
    model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-389m-zh", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-389m-zh")

    if tuning_function in tuning_function_list:
        func_args = "(model, model_id={})".format("./chatbot_lora/checkpoint-500/")
        func_name = "load_" + tuning_function + "model" + func_args
        merge_model = eval(func_name)

        ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧?", "").strip() + "\n\nAssistant: ", return_tensors="pt")
        tokenizer.decode(merge_model.generate(**ipt, do_sample=False)[0], skip_special_tokens=True)
    else:
        print("please input a correct tuning function name")
