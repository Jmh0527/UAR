import torch
import torch.nn as nn
import torch.nn.functional as F

from register import NetworkRegistry 

@NetworkRegistry.register('AIMClassifier')
class AIMClassifier(nn.Module):
    def __init__(self, in_features=3072, out_features=1):
        super(AIMClassifier, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)