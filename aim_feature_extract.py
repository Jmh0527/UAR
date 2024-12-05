import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image 
from tqdm import tqdm

from aim.v1.utils import load_pretrained
from preprocess.patchcraft_preprocess import Patch


class ImageFeatureDataset(Dataset):
    """
    Dataset for processing images and generating feature outputs.
    """
    def __init__(self, input_path, output_path, processor):
        """
        Args:
            input_path (str): Path to the input file containing image paths.
            output_path (str): Base path for saving the output feature files.
            processor (callable): Preprocessing pipeline for images.
        """
        self.processor = processor
        self.image_paths, self.output_paths = self._prepare_paths(input_path, output_path)

    @staticmethod
    def _prepare_paths(input_path, output_path):
        """Prepares the input and output paths."""
        image_paths, output_paths = [], []
        for dirpath in Path(input_path).rglob('*'):
            if dirpath.is_file():
                if dirpath.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
                    raise ValueError(f"Unsupported file format: {dirpath.suffix}")
                image_paths.append(dirpath)
                output_file = Path(output_path) / (dirpath.stem + '.npy')
                output_paths.append(output_file)
        return image_paths, output_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Loads an image, preprocesses it, and retrieves the output path."""
        img_path = self.image_paths[idx]
        output_path = self.output_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            patch_processor = Patch(size=224)
            rich_image, _ = patch_processor(np.array(img).astype(np.float32))
            rich_image = Image.fromarray(rich_image.astype(np.uint8))
            inputs = self.processor(rich_image)
        except:
            img = self._load_fallback_image(img_path)
            inputs = self.processor(img)
        
        return inputs, str(output_path)

    @staticmethod
    def _load_fallback_image(img_path):
        """Fallback image loader in case of processing errors."""
        img = Image.open(img_path).convert('RGB')
        return transforms.Resize((224, 224))(img)


def main(args):
    processor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    dataset = ImageFeatureDataset(args.input_path, args.output_path, processor)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    model = load_pretrained(
        "aim-3B-5B-imgs",
        backend="torch",
        backbone_ckpt_path=args.backbone_ckpt_path,
        head_ckpt_path=args.head_ckpt_path,
        strict=False
    )
    model.cuda() 
    model.eval()

    os.makedirs(args.output_path, exist_ok=True)
    with torch.no_grad():
        for inputs, output_path in tqdm(data_loader, total=len(data_loader), desc="Processing images"):
            if os.path.exists(output_path[0]):
                continue

            inputs = inputs.cuda()
            _, features = model(inputs) 
            np.save(output_path[0], features.cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images and save features.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the file listing image paths.")
    parser.add_argument('--output_path', type=str, required=True, help="Directory to save output feature files.")
    parser.add_argument('--backbone_ckpt_path', type=str, required=True, help="Path to the backbone checkpoint.")
    parser.add_argument('--head_ckpt_path', type=str, required=True, help="Path to the head checkpoint.")
    args = parser.parse_args()

    main(args)



"""
python /home/data2/jmh/aim_feature_extract.py \
        --input_path /home/data2/jmh/demo/demo_images/1_fake \
        --output_path /home/data2/jmh/demo/demo_aimnpy/1_fake \
        --backbone_ckpt_path /home/data2/jingmh/code/ml-aim/pretrain_checkpoints/aim_3b_5bimgs_attnprobe_backbone.pth \
        --head_ckpt_path /home/data2/jingmh/code/ml-aim/pretrain_checkpoints/aim_3b_5bimgs_attnprobe_head_best_layers.pth
"""