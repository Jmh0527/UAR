import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """
    BaseDataset serves as a generic template for loading data.
    Supports image and .npy data with optional labels and transformations.
    """

    SUPPORTED_FORMATS = {
        'image': ('.jpg', '.jpeg', '.png'),
        'npy': ('.npy')
    }
    def __init__(self, img_paths, transform=None, data_type='image'):
        """
        Args:
            img_paths (list[str]): List of paths to the data files.
            transform (callable or None): A function/transform to apply to the data.
            data_type (str): Type of data ('image' or 'npy').
        """
        if data_type not in ['image', 'npy']:
            raise ValueError(f"Unsupported data_type: {data_type}")
        self.img_paths = img_paths
        self.transform = transform
        self.data_type = data_type
        self._preprocess = None
        self.img_paths = self.prepare_paths(img_paths)

    def prepare_paths(self, input_path):
        image_paths = []
        for dirpath in Path(input_path).rglob('*'):
            if dirpath.is_file():
                if dirpath.suffix.lower() not in self.SUPPORTED_FORMATS.get(self.data_type, set()):
                    raise ValueError(f"Unsupported file format: {dirpath.suffix}")
                image_paths.append(dirpath)
        return image_paths

    @property
    def preprocess(self):
        return self._preprocess

    @preprocess.setter
    def preprocess(self, fn):
        self._preprocess = fn

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.data_type == 'image':
            data = Image.open(self.img_paths[idx]).convert('RGB')
        elif self.data_type == 'npy':
            data = np.load(self.img_paths[idx])
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

        if self._preprocess:
            data = self._preprocess(data)

        if self.transform:
            data = self.transform(data)

        if '0_real' in str(self.img_paths[idx]):
            label = 0.0
        elif '1_fake' in str(self.img_paths[idx]):
            label = 1.0
        else:
            raise ValueError(f"Unsupported label type: {str(self.img_paths[idx])}")
        return data, label
