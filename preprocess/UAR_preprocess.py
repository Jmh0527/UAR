import random
from typing import Tuple

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from register import TransformRegistry


@TransformRegistry.register('UAR')
class SelectCrop:
    def __init__(self, size=224, patch_num=192):
        super(Patch, self).__init__()
        self.size = size 
        self.patch_num = patch_num
        self.processor =  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    
    @staticmethod
    def get_pixel_flictuation(x):
        res = 0
        res += np.sum(np.abs(x[:, :-1] - x[:, 1:])) 
        res += np.sum(np.abs(x[:-1, :] - x[1:, :])) 
        res += np.sum(np.abs(x[:-1, :-1] - x[1:, 1:]))
        res += np.sum(np.abs(x[1:, :-1] - x[:-1, 1:]))
        return res
    
    def smash_recons(self, x):
        '''
        Randomly sample 192 224*224 patches in image
        In the rich/poor texture reconstructed images, each patch is sorted from top left to bottom right based on their diversity. 
        The patch located in the top-left corner contains the poorest/richest texture
        '''
        patches = []
        pixel_flictuation = []
        for _ in range(self.patch_num):
            sample1 = random.randint(0, x.shape[0] - self.size)
            sample2 = random.randint(0, x.shape[1] - self.size)
            patch = x[sample1:sample1+self.size, sample2:sample2+self.size]
            patches.append(patch)
            pixel_flictuation.append(Patch.get_pixel_flictuation(patch))
        pixel_flictuation = np.array(pixel_flictuation)
        sorted_indices = np.argsort(pixel_flictuation)

        rich_image = patches[sorted_indices[-1]]
        poor_image = patches[sorted_indices[0]]
        return rich_image, poor_image
    
    def __call__(self, x):
        x = np.array(x)
        rich_image, poor_image = self.smash_recons(x)
        
        try:
            rich_image, poor_image = self.smash_recons(np.array(img).astype(np.float32))
            rich_image = Image.fromarray(rich_image.astype(np.uint8))
            inputs = self.processor(rich_image).unsqueeze(0)
        except:
            img = transforms.Resize((224, 224))(img)
            nputs = self.processor(img).unsqueeze(0)
        
        return inputs.squeeze(0)

