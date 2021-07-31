"""
ShuffleNet is a convolutional neural network designed specially for mobile devices 
with very limited computing power. 

The architecture utilizes two new operations, pointwise group convolution and channel 
shuffle, to reduce computation cost while maintaining accuracy. ShuffleNet uses wider 
feature maps as smaller networks has lesser number of channels.

Channel Shuffle: https://paperswithcode.com/method/channel-shuffle#

It is an operation to help information flow across feature channels in 
CNN.

If we allow a group convolution to obtain input data from different groups, the input 
and output channels will be fully related. Specifically, for the feature map generated 
from the previous group layer, we can first divide the channels in each group into 
several subgroups, then feed each group in the next layer with different subgroups.

The above can be efficiently and elegantly implemented by a channel shuffle operation:

suppose a convolutional layer with g groups whose output has (g x n) channels; we first 
reshape the output channel dimension into (g, n), transposing and then flattening it back 
as the input of next layer. Channel shuffle is also differentiable, which means it can be 
embedded into network structures for end-to-end training.

ShuffleNet achieves 13x speedup over AlexNet with comparable accuracy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Shuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)

class Bottleneck(nn.Module):
    def __init__(self, input_channel, output_channel, stride, groups):
        super().__init__()
        self.stride = stride

        in_between_channel = int(output_channel / 4)
        g = 1 if input_channel==24 else groups

        #Group Convolution
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(input_channel, in_between_channel, kernel_size=1, groups=g, bias=False),
            nn.BatchNorm2d(in_between_channel), nn.ReLU(inplace=True))
        self.shuffle = Shuffle(groups=g)

        #Depthwise Convolution
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(in_between_channel, in_between_channel, kernel_size=3, stride=stride, padding=1, groups=in_between_channel, bias=False),
            nn.BatchNorm2d(in_between_channel), nn.ReLU(inplace=True))
        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(in_between_channel, output_channel, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(output_channel))
            
        self.shortcut = nn.Sequential()
        if stride==2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = self.conv1x1_1(x)
        out = self.shuffle(out)
        out = self.conv1x1_2(out)
        out = self.conv1x1_3(out)
        res = self.shortcut(x)
        out = F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out+res)
        return out

class ShuffleNet(nn.Module):
    def __init__(self, cfg, input_channel, n_classes):
        super().__init__()
        output_channels = cfg['out']
        n_blocks = cfg['n_blocks']
        groups = cfg['groups']
        self.in_channels = 24
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self.make_layer(output_channels[0], n_blocks[0], groups)
        self.layer2 = self.make_layer(output_channels[1], n_blocks[1], groups)
        self.layer3 = self.make_layer(output_channels[2], n_blocks[2], groups)
        self.linear = nn.Linear(output_channels[2], n_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)
    def make_layer(self, out_channel, n_blocks, groups):
        layers = []
        for i in range(n_blocks):
            stride = 2 if i==0 else 1
            cat_channels = self.in_channels if i==0 else 0
            layers.append(Bottleneck(self.in_channels, out_channel-cat_channels, stride=stride, groups=groups))        
            self.in_channels = out_channel

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
