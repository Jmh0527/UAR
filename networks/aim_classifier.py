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
        x = self.linear(x)
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

@NetworkRegistry.register('AIMClassifier_keepneg_singlelinear')
class AIMClassifier_keepneg_singlelinear(nn.Module):
    def __init__(self, in_features=3072, out_features=1):
        super(AIMClassifier_keepneg_singlelinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        x = F.relu(x)
        # 处理权重：将负数权重置为 0
        with torch.no_grad():  # 防止梯度更新时被计算
            self.linear.weight[self.linear.weight < 0] = 0
        x = self.linear(x)
        return x
    
    
@NetworkRegistry.register('AIMClassifier_keeppos_singlelinear')
class AIMClassifier_keeppos_singlelinear(nn.Module):
    def __init__(self, in_features=3072, out_features=1):
        super(AIMClassifier_keeppos_singlelinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        x = F.relu(x)
        # 处理权重：将负数权重置为 0
        with torch.no_grad():  # 防止梯度更新时被计算
            self.linear.weight[self.linear.weight > 0] = 0
        x = self.linear(x)
        return x


@NetworkRegistry.register('AIMClassifier_L1_singlelinear')
class AIMClassifier_L1_singlelinear(nn.Module):
    def __init__(self, in_features=3072, out_features=1):
        super(AIMClassifier_L1_singlelinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        x = F.relu(x)
        x = self.linear(x)
        return x

    @property
    def l1_penalty(self):
        return torch.sum(torch.abs(self.linear.weight))


@NetworkRegistry.register('AIMClassifier_featureselectsamll')
class AIMClassifier_featureselectsamll(nn.Module):
    def __init__(self, in_features=512, out_features=1):
        super(AIMClassifier_featureselectsamll, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        # 预加载 checkpoint 并计算出权重 tensor 中数值最小的 2048 个元素对应的索引
        ckpt = torch.load('/home/kh31/IJCV/UAR/checkpoints/AIMClassifier_ReLU_L1_png_jpeg_data/epoch_8_model.pth')
        weight_tensor = ckpt['linear2.weight']
        
        # 2400 2048 1600 1200 800
        _, small_indices = torch.topk(weight_tensor, k=512, dim=1, largest=False)
        # 将索引 squeeze 成一维张量 (2048,)
        self.small_indices = small_indices.squeeze(0)
        
    def forward(self, x):
        # 根据预计算好的索引筛选特征
        x = x[..., self.small_indices]
        x = self.linear(x)
        return x
    
    
@NetworkRegistry.register('AIMClassifier_featureselectlarge')
class AIMClassifier_featureselectlarge(nn.Module):
    def __init__(self, in_features=2048, out_features=1):
        super(AIMClassifier_featureselectlarge, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        # 预加载 checkpoint 并计算出权重 tensor 中数值最小的 2048 个元素对应的索引
        ckpt = torch.load('/home/kh31/IJCV/UAR/checkpoints/AIMClassifier_ReLU_L1_png_jpeg_data/epoch_8_model.pth')
        weight_tensor = ckpt['linear2.weight']
        
        # 2400 2048 1600 1200 800
        _, small_indices = torch.topk(weight_tensor, k=2048, dim=1, largest=True)
        # 将索引 squeeze 成一维张量 (2048,)
        self.small_indices = small_indices.squeeze(0)
        
    def forward(self, x):
        # 根据预计算好的索引筛选特征
        x = x[..., self.small_indices]
        x = self.linear(x)
        return x
