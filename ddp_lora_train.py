import logging
import argparse

import torch
import torch.nn as nn
from torch.optim import Adam
from peft import LoraConfig, get_peft_model

from train_engine import Trainer
from dataset import BaseDataset
from networks import NetworkRegistry 
from preprocess import TransformRegistry

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

def init_distributed_mode(args):
    """Initialize distributed training environment."""
    dist.init_process_group(backend='nccl')  # You can use 'gloo' if not on a CUDA machine
    torch.cuda.set_device(args.local_rank)
    print(f"Training on GPU {args.local_rank}")

def main(args):
    init_distributed_mode(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler())

    # LoRA config
    lora_config = LoraConfig(
        r=8, # 逐步实验看8 16 32的效果
        lora_alpha=16,  
        lora_dropout=0.1,
        target_modules=["proj", "qkv", "fc1", "fc2"]
    )

    if args.model not in NetworkRegistry.list_registered():
        raise ValueError(f"Model {args.model} is not registered in NetworkRegistry.")
    
    model = NetworkRegistry[args.model](
        backbone_ckpt_path=args.backbone_ckpt_path,
        head_ckpt_path=args.head_ckpt_path
    )
    
    lora_model = get_peft_model(model, lora_config)
    lora_model = DDP(lora_model, device_ids=[args.local_rank], output_device=args.local_rank)
    print_trainable_parameters(lora_model)

    if args.transform in TransformRegistry.list_registered():
        transform = TransformRegistry[args.transform]()
    else:
        transform = None

    train_dataset = BaseDataset(
        img_paths=args.dataroot,
        transform=transform,
    )

    train_sampler = DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Disabled because DistributedSampler handles shuffling
        sampler=train_sampler
    )


    optimizer = Adam(
        lora_model.parameters(), 
        lr=args.lr,
        betas=(0.9, 0.999)
    )
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    trainer = Trainer(
        model=lora_model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        logger=logger,
        epoch=args.epoch,
        save_dir=f"./checkpoints/{args.model}_lora",
        loss_freq=10
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='AIMClassifier', help="Defined in the NetworkRegistry: AIMClassifier or PatchCraft")
    parser.add_argument('--epoch', type=int, default=1, help="The number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="The learning rate during training")
    parser.add_argument('--dataroot', type=str, required=True, help="The dir path of input images")
    parser.add_argument('--batch_size', type=int, default=32, help="The batch size during training")
    parser.add_argument('--transform', type=str, default=None, help="Transformation method registered in TransformRegistry")
    parser.add_argument('--backbone_ckpt_path', type=str, default='/home/data2/jingmh/code/ml-aim/pretrain_checkpoints/aim_3b_5bimgs_attnprobe_backbone.pth'),
    parser.add_argument('--head_ckpt_path', type=str, default='/home/data2/jingmh/code/ml-aim/pretrain_checkpoints/aim_3b_5bimgs_attnprobe_head_best_layers.pth'),
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for DistributedDataParallel (DDP)')
     
    args = parser.parse_args()
    main(args)

# torchrun --nproc_per_node=4 --master_addr="localhost" --master_port=12355 train.py --dataroot /path/to/imagenet --batch_size 32 --model AIMClassifier --transform UAR