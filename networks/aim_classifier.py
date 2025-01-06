from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image 
from tqdm import tqdm

from aimv1.aim.v1.torch import models
from register import NetworkRegistry 


def load_state_dict(
    loader: Callable[..., Dict[str, torch.Tensor]],
    backbone_loc: str,
    head_loc: Optional[str],
) -> Dict[str, torch.Tensor]:
    backbone_state_dict = loader(backbone_loc, map_location="cpu")

    if head_loc is None:
        raise RuntimeError("Unable to load the head, no location specified.")
    head_state_dict = loader(head_loc, map_location="cpu")
    return merge_state_dicts(
        backbone_state_dict, head_state_dict, allow_override=False
    )

def merge_state_dicts(
    backbone_state_dict: Dict[str, torch.Tensor],
    head_state_dict: Dict[str, torch.Tensor],
    *,
    allow_override: bool = False,
) -> Dict[str, torch.Tensor]:
    overlapping_keys = set(backbone_state_dict.keys()) & set(head_state_dict.keys())
    if overlapping_keys and not allow_override:
        raise ValueError(
            f"Backbone and head state dicts have "
            f"following overlapping keys: {sorted(overlapping_keys)}."
        )

    return {**backbone_state_dict, **head_state_dict}


@NetworkRegistry.register('AIMClassifier')
class AIMClassifier(nn.Module):
    def __init__(self, in_features=3072, out_features=1, **kwargs):
        super(AIMClassifier, self).__init__()

        self.model = models.aim_3B(probe_layers=(12, 13, 14, 15, 16, 17))
        if 'backbone_ckpt_path' in kwargs.keys():
            state_dict = load_state_dict(torch.load, 
                                        backbone_loc=kwargs['backbone_ckpt_path'], 
                                        head_loc=kwargs['head_ckpt_path'])
            self.model.load_state_dict(state_dict, strict=False)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        _, x = self.model(x)
        return self.linear(x)


if __name__ == '__main__':
    x = torch.ones((1,3,224,224)).cuda()
    model = AIMClassifier(backbone_ckpt_path='/home/data2/jingmh/code/ml-aim/pretrain_checkpoints/aim_3b_5bimgs_attnprobe_backbone.pth', 
                          head_ckpt_path='/home/data2/jingmh/code/ml-aim/pretrain_checkpoints/aim_3b_5bimgs_attnprobe_head_best_layers.pth').cuda()
    print(model(x))
    