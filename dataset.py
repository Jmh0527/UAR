import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from preprocess.patchcraft_preprocess import Patch

class BaseDataset(Dataset):
    """
    BaseDataset serves as a generic template for loading data.
    Supports image and .npy data with optional labels and transformations.
    """

    SUPPORTED_FORMATS = {
        'image': ('.jpg', '.jpeg', '.png'),
        'npy': ('.npy')
    }
    def __init__(self, img_paths, transform=None):
        """
        Args:
            img_paths (list[str]): List of paths to the data files.
            transform (callable or None): A function/transform to apply to the data.
        """
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        self._preprocess = None
        self.img_paths = self.prepare_paths(img_paths)

    def prepare_paths(self, input_path):
        image_paths = []
        for dirpath in Path(input_path).rglob('*'):
            if dirpath.is_file():
                if dirpath.suffix.lower() not in self.SUPPORTED_FORMATS['image']:
                    raise ValueError(f"Unsupported file format: {dirpath.suffix}")
                image_paths.append(dirpath)
        return image_paths

    @staticmethod
    def _load_fallback_image(img_path):
        """Fallback image loader in case of processing errors."""
        img = Image.open(img_path).convert('RGB')
        return transforms.Resize((224, 224))(img)
    
    @property
    def preprocess(self):
        return self._preprocess

    @preprocess.setter
    def preprocess(self, fn):
        self._preprocess = fn

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
    
        try:
            img = Image.open(img_path).convert('RGB')
            patch_processor = Patch(size=224)
            rich_image, _ = patch_processor(np.array(img).astype(np.float32))
            rich_image = Image.fromarray(rich_image.astype(np.uint8))
            inputs = self.transform(rich_image)
        except:
            img = self._load_fallback_image(img_path)
            inputs = self.transform(img)

        if '0_real' in str(self.img_paths[idx]):
            label = 0.0
        elif '1_fake' in str(self.img_paths[idx]):
            label = 1.0
        else:
            raise ValueError(f"Unsupported label type: {str(self.img_paths[idx])}")
        return inputs, label
