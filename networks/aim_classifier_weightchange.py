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
        # 处理权重：将负数权重置为 0
        with torch.no_grad():  # 防止梯度更新时被计算
            self.linear.weight[self.linear.weight < 0] = 0

        x = F.relu(x)
        x = self.linear(x)
        return x
