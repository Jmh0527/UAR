import torch
import torch.nn as nn
import torch.nn.functional as F

from register import NetworkRegistry

@NetworkRegistry.register('AIMClassifier')
class AIMClassifier(nn.Module):
    def __init__(self, in_features=3072, out_features=1):
        super(AIMClassifier, self).__init__()
        self.linear1 = nn.Linear(in_features, in_features)
        self.linear2 = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)  # 使用torch.nn.functional中的ReLU
        x = self.linear2(x)
        return x


@NetworkRegistry.register('AIMClassifier_keeppos')
class AIMClassifier_keeppos(nn.Module):
    def __init__(self, in_features=3072, out_features=1):
        super(AIMClassifier_keeppos, self).__init__()
        self.linear1 = nn.Linear(in_features, in_features)
        self.linear2 = nn.Linear(in_features, out_features)


    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        # 处理权重：将负数权重置为 0
        with torch.no_grad():  # 防止梯度更新时被计算
            self.linear2.weight[self.linear2.weight < 0] = 0
        x = self.linear2(x)
        return x
    
    
@NetworkRegistry.register('AIMClassifier_keepneg')
class AIMClassifier_keepneg(nn.Module):
    def __init__(self, in_features=3072, out_features=1):
        super(AIMClassifier_keepneg, self).__init__()
        self.linear1 = nn.Linear(in_features, in_features)
        self.linear2 = nn.Linear(in_features, out_features)


    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        # 处理权重：将负数权重置为 0
        with torch.no_grad():  # 防止梯度更新时被计算
            self.linear2.weight[self.linear2.weight > 0] = 0
        x = self.linear2(x)
        return x
    

@NetworkRegistry.register('AIMClassifier_L1')
class AIMClassifier_L1(nn.Module):
    def __init__(self, in_features=3072, out_features=1):
        super(AIMClassifier_L1, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        return x

    @property
    def l1_penalty(self):
        return torch.sum(torch.abs(self.linear.weight))


@NetworkRegistry.register('AIMClassifier_ReLU_L1')
class AIMClassifier_ReLU_L1(nn.Module):
    def __init__(self, in_features=3072, out_features=1):
        super(AIMClassifier_ReLU_L1, self).__init__()
        self.linear1 = nn.Linear(in_features, in_features)
        self.linear2 = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
    
    @property
    def l1_penalty(self):
        return torch.sum(torch.abs(self.linear2.weight))


@NetworkRegistry.register('AIMClassifier_keepneg_L1')
class AIMClassifier_keepneg_L1(nn.Module):
    def __init__(self, in_features=3072, out_features=1):
        super(AIMClassifier_keepneg_L1, self).__init__()
        self.linear1 = nn.Linear(in_features, in_features)
        self.linear2 = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        # 处理权重：将负数权重置为 0
        with torch.no_grad():  # 防止梯度更新时被计算
            self.linear2.weight[self.linear2.weight < 0] = 0
        x = self.linear2(x)
        return x
    
    @property
    def l1_penalty(self):
        return torch.sum(torch.abs(self.linear2.weight))
