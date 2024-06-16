# 该版本为断点续训的版本
import json
import math
import os
import argparse
from pathlib import Path
from PIL import Image

import torch
import torch.utils
import torch.utils.data
from torchvision import transforms

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from transformers import  CLIPTokenizer


class MyDataset(torch.utils.data.Dataset):
    '''
    按照diffusers训练的数据集格式来加载数据
    list of dict: [{"image_file": "1.png", "text": "A dog"}]
    '''
    def __init__(self, json_file, tokenizer, size=512, image_root_path=""):
        super.__init__()
        self.tokenizer = tokenizer  # 如果有文本数据，需要对文本数据进行分词转成id序列，才能输入网络进行训练
        self.size = size  # 需要调整的分辨率大小
        
        '''
        TODO 以下的就是其他参数，如果有需要
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        '''
        self.image_root_path = image_root_path  # 图片的路径，因为json里面的image_file是文件名

        # 加载数据
        self.data = json.load(open(json_file))

        # 数据增强操作
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        pass

    def __getitem__(self, index):
        item = self.data[index]  # 获取下标为index的数据
        text = item["text"]  # 该条数据的文本数据
        image_file = item["image_file"]  # 该条数据的图片数据

        # 1、图片的操作
        #   1）、去读取图片
        #   2）、图片增强
        #   3)、其他操作（如果有需要）
        # raw = Image.open(Path(self.image_root_path, image_file))
        # 或者下面的操作
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        image = self.transform(raw_image.convert("RGB"))

        '''
        TODO 其他的操作，如果有需要
        '''

        # 2、文本的操作
        # 需要的只是ids序列
        text_input_ids = self.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors='pt'
            ).input_ids

        return {
            "image": image,
            "text_input_ids": text_input_ids,
            # TODO 其他的参数，如果有需要
        }

    def __len__(self,):
        return len(self.data)

# DataLoader数据集的一个collate_fn,相当于是后处理，将所有的数据拼接在一起
def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    # TODO 其他数据cat操作，如果有需要

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        # TODO 其他的参数，如果有需要
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
            "--pretrained_model_name_or_path",
            type=str,
            default=None,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models.",
        )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ckpt",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="ckpt",
        help="The resume directory where save the mode checkpoints state.",
    )
    
    

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(mixed_precision=args.mixed_precision,
                              gradient_accumulation_steps=args.gradient_accumulation_steps,
                              log_with=args.report_to,
                              project_config=accelerator_project_config)
    
    # 主GPU进程创建目录
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    # 追踪，记录实验数据，只使用init_tracker只能使用log数据        
    accelerator.init_trackers("runs")
    # 以下方法还可以记录图片数据
    tensorboard_tracker = accelerator.get_tracker("tensorboard")

    '''
    TODO: 1、这部分就是加载你自己的模型
    '''
    model = None
    # 这边需要定义一个tokenizer来进行分词，但是分词器自己选择
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    '''
    TODO: 2、这部分就是加载优化器
    '''
    optimizer = None


    '''
    TODO: 3、这部分就是定义数据集
    '''
    train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, size=args.resolution, image_root_path=args.data_root_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # 将模型，优化器，数据进一步包装在accelerator里面
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    '''
    断点续训，这里需要指定一下从哪一个状态开始训练
    '''
    global_step = 0
    # 如果是第一次使用的话，需要设置epoch和step为0
    resume_step = 0
    resume_epoch = 0
    # 如果有resume，那么就需要重新计算上面的两个值
    if args.resume is not None:
        accelerator.print(f"load resume checkpoint state ...-> {args.resume}")
        # 加载当前的状态
        accelerator.load_state(args.resume)
        # 计算每一个epoch有多少次迭代
        # 如果没有梯度累计的话，那么就是len(dataloader), 因为dataloader每一次调用是以batch的形式的，他的大小不是按照dataset里面的item计算的
        steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)  # 向下取整
        accelerator.print(f"steps_per_epoch -> {steps_per_epoch}")
        # 当前是第几个step，可以通过传进来的路径获取
        # 这边global step的状态也是要改变的，为刚开始的resume_step
        resume_step = global_step = int(args.resume.split("checkpoint-")[-1])
        accelerator.print(f"total resume_step -> {resume_step}")
        # 计算当前是否运行到第几个epoch = 当前的step/一个epoch有多少个step
        resume_epoch = resume_step // steps_per_epoch
        accelerator.print(f"resume_epoch -> {resume_epoch}")
        # 当前epoch已经训练了多少step，因为她这边是按照step进行保存的，保不齐它会跑到epoch里面跑了几个step再保存的
        resume_step -= resume_epoch * steps_per_epoch
        accelerator.print(f"now epoch resume step -> {resume_step}")
        accelerator.print(f"resume from checkpoint -> {args.resume}")


    
    for epoch in range(resume_epoch, args.num_train_epochs):
        # 这边需要考虑跳过多少个step，如果有resume
        if args.resume and epoch == resume_epoch and resume_step != 0:
            # 这边需要考虑跳过多少个step,其实就是跳过多少个数据
            # 由于这里使用了梯度累计，因此在计算跳过的步数的时候需要乘上一个梯度累计，才是最终要跳过多少步数据
            activa_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step * args.gradient_accumulation_steps)
        else:
            # 如果是从0开始，那么就是完整的train_dataloader
            activa_dataloader = train_dataloader
        for step, batch in enumerate(activa_dataloader):
            # 因为这边使用了梯度累计，所以训练的代码需要包裹在里面with accelerator.accumulate(model)
            with accelerator.accumulate(model):

                '''
                TODO 训练前向传播和计算loss
                '''
                loss = None

                # Gather the losses across all processes for logging (if we use distributed training).
                # 平均所有的loss
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                # Backpropagate
                accelerator.backward(loss)  # 需要将loss放在accelerator里面进行反向传播            
                optimizer.step()
                optimizer.zero_grad()
                
                
                # 可以使用sync_geadients来判断是否进行了梯度的跟新，这里返回的是一个bool值
                # 然后可以做一些梯度更新以后的操作了
                if accelerator.sync_gradients:
                    global_step += 1
                    # accelerator.log({"training_loss": loss}, step=iter)  # 用来记录实验的数据
                    tensorboard_tracker.log({"training_loss": avg_loss}, step=global_step)

                    # 间隔save_steps保存中间训练模型
                    if global_step % args.save_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.print(f"save checkpoint -> checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        accelerator.print(f"save model -> checkpoint-{global_step}/model")
                        # 保存模型，这种方式会保存config.json文件，模型会保存在model文件夹下面
                        accelerator.unwrap_model(model).save_pretrained(
                            save_directory=accelerator.project_dir + f"/checkpoint-{global_step}/model",  # 模型保存的路径
                            is_main_process=accelerator.is_main_process,  # 主进程才会保存模型
                            state_dict=accelerator.get_state_dict(model),  # 保存state_dict
                            save_func=accelerator.save  # 以什么样的方式保存
                        )
                        
    # 训练结束之后需要指定track结束
    accelerator.end_training()

if __name__ == "__main__":
    main()