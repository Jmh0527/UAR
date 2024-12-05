import random
from typing import Tuple

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from register import TransformRegistry


@TransformRegistry.register('HighPassFilter')
class HighPassFilter(nn.Module):
    """
    Applies a predefined high-pass filter using SRM kernels.
    """
    def __init__(self):
        super(HighPassFilter, self).__init__()
        self.high_pass = nn.Conv2d(in_channels=3, out_channels=30, 
                                kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        filter = torch.tensor(np.load('./preprocess/SRM_Kernels.npy'), dtype=torch.float32)
        filter = filter.repeat(1, 3, 1, 1)
        self.high_pass.weight = nn.Parameter(filter, requires_grad=False)
        return self.high_pass(x)


@TransformRegistry.register('Patch')
class Patch:
    """
    Extracts and sorts patches from an image based on pixel fluctuation diversity.
    """
    def __init__(self, size=32, patch_num=192):
        self.size = size 
        self.patch_num = patch_num
    
    @staticmethod
    def get_pixel_fluctuation(patch: np.ndarray) -> int:
        """
        Computes the fluctuation diversity of a patch based on pixel differences.
        """
        fluctuation = 0
        fluctuation += np.sum(np.abs(patch[:, :-1] - patch[:, 1:]))
        fluctuation += np.sum(np.abs(patch[:-1, :] - patch[1:, :]))
        fluctuation += np.sum(np.abs(patch[:-1, :-1] - patch[1:, 1:]))
        fluctuation += np.sum(np.abs(patch[1:, :-1] - patch[:-1, 1:]))
        return fluctuation
    
    def smash_recons(self, image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples patches, sorts them by diversity, and reconstructs rich/poor texture images.
        In the rich/poor texture reconstructed images, each patch is sorted from top left to bottom right based on their diversity.
        The patch located in the top-left corner contains the poorest/richest texture.
        
        Args:
            image (np.ndarray): Input image.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Rich-texture and poor-texture reconstructed images.
        """
        if image.shape[0] < self.size or image.shape[1] < self.size:
            raise ValueError(f"Image dimensions must be at least {self.size}x{self.size}.")

        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        patches = []
        pixel_fluctuations = []
        for _ in range(self.patch_num):
            y = random.randint(0, image.shape[0] - self.size)
            x = random.randint(0, image.shape[1] - self.size)
            patch = image[y:y+self.size, x:x+self.size]
            patches.append(patch)
            pixel_fluctuations.append(self.get_pixel_fluctuation(patch))

        sorted_indices = np.argsort(pixel_fluctuations) # incremental
        first_third = sorted_indices[:len(sorted_indices) // 3]
        last_third = sorted_indices[-len(sorted_indices) // 3:]
        
        rich_patches = [patches[i] for i in last_third[::-1]]
        poor_patches = [patches[i] for i in first_third]

        rich_image = np.vstack([np.hstack(rich_patches[i*8:(i+1)*8]) for i in range(8)])
        poor_image = np.vstack([np.hstack(poor_patches[i*8:(i+1)*8]) for i in range(8)])
        return rich_image, poor_image
    
    def __call__(self, x):
        rich_image, poor_image = self.smash_recons(x)
        return HighPassFilter()(rich_image), HighPassFilter()(poor_image)
