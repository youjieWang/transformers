# 该版本是从头开始训练的版本
# epoch：0-epoch
import json
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

    global_step = 0
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
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
                        accelerator.print(f"save model -> checkpoint-{global_step}/model")
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # 保存模型，这种方式会保存config.json文件
                        accelerator.unwrap_model(model).save_pretrained(
                            save_directory=accelerator.project_dir + f"/step_{global_step}/model",  # 模型保存的路径
                            is_main_process=accelerator.is_main_process,  # 主进程才会保存模型
                            state_dict=accelerator.get_state_dict(model),  # 保存state_dict
                            save_func=accelerator.save  # 以什么样的方式保存
                        )
                        # accelerator.save_state(save_path)
    # 训练结束之后需要指定track结束
    accelerator.end_training()

if __name__ == "__main__":
    main()