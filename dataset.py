import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class BaseDataset(Dataset):
    """
    BaseDataset serves as a generic template for loading data.
    Supports image and .npy data with optional labels and transformations.
    """
    def __init__(self, img_paths, labels=None, transform=None, data_type='image'):
        """
        Args:
            img_paths (list[str]): List of paths to the data files.
            labels (list[int] or None): List of labels corresponding to the data, optional.
            transform (callable or None): A function/transform to apply to the data.
            data_type (str): Type of data ('image' or 'npy').
        """
        if data_type not in ['image', 'npy']:
            raise ValueError(f"Unsupported data_type: {data_type}")

        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.data_type = data_type
        self._preprocess = None

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

        label = self.labels[idx]
        return data, label
