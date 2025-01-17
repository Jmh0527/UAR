import argparse
import logging
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from train_engine import Trainer
from dataset import BaseDataset
from networks import NetworkRegistry
from preprocess import TransformRegistry

def init_distributed_mode(args):
    """Initialize distributed training environment."""
    dist.init_process_group(backend='nccl')  # You can use 'gloo' if not on a CUDA machine
    torch.cuda.set_device(args.local_rank)
    print(f"Training on GPU {args.local_rank}")

def main(args):
    # Initialize the distributed environment
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
    
    # Initialize model
    if args.model not in NetworkRegistry.list_registered():
        raise ValueError(f"Model {args.model} is not registered in NetworkRegistry.")
    
    model = NetworkRegistry[args.model](
        backbone_ckpt_path=args.backbone_ckpt_path,
        head_ckpt_path=args.head_ckpt_path
    ).to(args.local_rank)  # Send the model to the local GPU
    
    # Convert model to DDP
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # Initialize transform
    if args.transform in TransformRegistry.list_registered():
        transform = TransformRegistry[args.transform]()
    else:
        transform = None
    
    # Dataset and DataLoader
    train_dataset = BaseDataset(
        img_paths=args.dataroot,
        transform=transform,
    )

    # DistributedSampler ensures each process gets a different subset of data
    train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Disabled because DistributedSampler handles shuffling
        sampler=train_sampler
    )

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999)
    )
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    trainer = Trainer(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        logger=logger,
        epoch=args.epoch,
        save_dir=f"./checkpoints/{args.model}",
        loss_freq=10
    )

    trainer.train()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='AIMClassifier', help="Model to train (AIMClassifier or PatchCraft)")
    parser.add_argument('--epoch', type=int, default=1, help="The number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--dataroot', type=str, required=True, help="Directory path of input images")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size per GPU")
    parser.add_argument('--transform', type=str, default=None, help="Transformation method registered in TransformRegistry")
    parser.add_argument('--backbone_ckpt_path', type=str, default='/path/to/backbone.pth')
    parser.add_argument('--head_ckpt_path', type=str, default='/path/to/head.pth')
    
    # Arguments for distributed training
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for DistributedDataParallel (DDP)')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

# torchrun --nproc_per_node=4 --master_addr="localhost" --master_port=12355 train.py --dataroot /path/to/imagenet --batch_size 32 --model AIMClassifier --transform UAR
