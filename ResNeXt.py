"""
ResNeXt is a simple, highly modularized network architecture for image classification. The
network is constructed by repeating a building block that aggregates a set of transformations 
with the same topology. The simple design results in a homogeneous, multi-branch architecture 
that has only a few hyper-parameters to set. This strategy exposes a new dimension, which is 
referred as “cardinality” (the size of the set of transformations), as an essential factor in 
addition to the dimensions of depth and width.

We can think of cardinality as the set of separate conv block representing same complexity as 
when those blocks are combined together to make a single block.

Blog: https://towardsdatascience.com/review-resnext-1st-runner-up-of-ilsvrc-2016-image-classification-15d7f17b42ac

#### Citation ####

PyTorch Code: https://github.com/prlz77/ResNeXt.pytorch

@article{Xie2016,
  title={Aggregated Residual Transformations for Deep Neural Networks},
  author={Saining Xie and Ross Girshick and Piotr Dollár and Zhuowen Tu and Kaiming He},
  journal={arXiv preprint arXiv:1611.05431},
  year={2016}
}

"""

import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    expansion = 2  
    """
    '''Grouped convolution block.'''
    groups: Integer 
    It controls the connections between inputs and outputs. in_channels and out_channels must 
    both be divisible by groups. For example,

    At groups=1, all inputs are convolved to all outputs.
    At groups=2, the operation becomes equivalent to having two conv layers side by side, each 
    seeing half the input channels and producing half the output channels, and both subsequently 
    concatenated.
    At groups= in_channels, each input channel is convolved with its own set of filters 
    (of size (out_channels/in_channels))

    group parameter in Conv2d, splits the output_channel by cardinality.

    """
  
    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        """group=cardinality, it divides the out_channel by 32(cardinality) i.e. thus, divides channel 128 into 4"""
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential(
              nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
              nn.BatchNorm2d(self.expansion*group_width)
          )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, input_channel, num_blocks, cardinality, bottleneck_width, n_classes=10):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        # self.layer4 = self._make_layer(num_blocks[3], 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(cardinality*bottleneck_width*8, n_classes)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def ResNeXt29_2x64d(input_channel, n_classes):
    return ResNeXt(input_channel=input_channel, num_blocks=[3,3,3], cardinality=2, bottleneck_width=64, n_classes=n_classes)

def ResNeXt29_4x64d(input_channel, n_classes):
    return ResNeXt(input_channel, num_blocks=[3,3,3], cardinality=4, bottleneck_width=64, n_classes=n_classes)

def ResNeXt29_8x64d(input_channel, n_classes):
    return ResNeXt(input_channel, num_blocks=[3,3,3], cardinality=8, bottleneck_width=64, n_classes=n_classes)

def ResNeXt29_32x4d(input_channel, n_classes):
    return ResNeXt(input_channel, num_blocks=[3,3,3], cardinality=32, bottleneck_width=4, n_classes=n_classes)
