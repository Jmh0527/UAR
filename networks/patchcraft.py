import torch
import torch.nn as nn

from register import NetworkRegistry 


class ConvBlock(nn.Module):
    """
    A reusable convolutional block with Conv2d, BatchNorm2d, and Hardtanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.Hardtanh(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


@NetworkRegistry.register('PatchCraft')
class PatchCraft(nn.Module):
    """
    PatchCraft model for feature filtering and classification.
    """

    def __init__(self):
        super(PatchCraft, self).__init__()
        self.conv_rich = ConvBlock(30, 30)
        self.conv_poor = ConvBlock(30, 30)
        self.classifier = self.build_classifier(30, 32, num_blocks=10, pool_kernel=2, pool_stride=2)
        self.linear = nn.Linear(2048, 1)
    
    def build_classifier(self, in_channels, out_channels, num_blocks, pool_kernel, pool_stride):
        """
        Builds the classifier with a specified number of blocks and a pooling layer applied
        after the fourth block and every two blocks thereafter.
        """
        layers = []
        for block_idx in range(num_blocks):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
            if 3 <= block_idx < 9 and (block_idx - 3) % 2 == 0:
                layers.append(nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(8, 8)))
        return nn.Sequential(*layers)

    def forward(self, x):
        rich, poor = x[0], x[1]
        # Process rich and poor features separately
        rich_features = self.conv_rich(rich)
        poor_features = self.conv_poor(poor)
        # Difference operation
        features = rich_features - poor_features
        # Pass through the classifier
        features = self.classifier(features)
        features = features.view(features.size(0), -1)  # Flatten
        return self.linear(features)

