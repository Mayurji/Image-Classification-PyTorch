"""
Squeeze and Excitation Network

A typical convolution network has kernels running through image channels and combining
the feature maps generated per channel. For each channel, we'll have separate kernel which
learns the weights through backpropagation.

The idea is to understand the interdependencies between channels of the images by explicitly
modeling on it and hence to make the network sensitive to informative features which is further
exploited in the next set of transformation.

* Squeeze(Global Information Embedding) operation converts feature maps into single value per channel.
* Excitation(Adaptive Recalibration) operation converts this single value into per-channel weight.

Squeeze turns (C x H x W) into (C x 1 x 1) using Global Average Pooling.
Excitation turns (C x 1 x 1) into (C x H x W) channel weights using 2 FC layer with activation function
inbetween, then which is expanded as same size as input.

Rescale the output from excitation operation into feature maps as earlier.

Based on the depth of the network, the role played by SE operation is differs. At early layers,
it excites shared low level representation irrespective of the classes. But in later stage, SE 
network responds differently based input class.

SE Block is simple and is added with existing CNN architecture to enhance the performance like 
ResNet or Inception V1 etc.

Reference: https://amaarora.github.io/2020/07/24/SeNet.html
"""
import torch.nn as nn
from ResNet import residualBlock

class SEBlock(nn.Module):
    def __init__(self, C, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(C, C//r, bias=False),
            nn.ReLU(),
            nn.Linear(C//r, C, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        bs, c, _, _ = x.shape
        s = self.squeeze(x).view(bs, c)
        e = self.excitation(s).view(bs, c, 1, 1)
        return x * e.expand_as(x)

class SENet(nn.Module):
    def __init__(self, input_channel, n_classes):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*[SEBlock(C=64)])
        self.b3 = nn.Sequential(*residualBlock(64, 64, 2, first_block=True))
        self.b4 = nn.Sequential(*[SEBlock(C=64)])
        self.b5 = nn.Sequential(*residualBlock(64, 128, 2))
        self.b6 = nn.Sequential(*[SEBlock(C=128)])
        self.b7 = nn.Sequential(*residualBlock(128, 256, 2))
        self.b8 = nn.Sequential(*[SEBlock(C=256)])
        self.b9 = nn.Sequential(*residualBlock(256, 512, 2))
        self.b10 = nn.Sequential(*[SEBlock(C=512)])
        self.finalLayer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, n_classes))

        self.b1.apply(self.init_weights)
        self.b2.apply(self.init_weights)
        self.b3.apply(self.init_weights)
        self.b4.apply(self.init_weights)
        self.b5.apply(self.init_weights)
        self.b6.apply(self.init_weights)
        self.b7.apply(self.init_weights)
        self.b8.apply(self.init_weights)
        self.b9.apply(self.init_weights)
        self.b10.apply(self.init_weights)
        self.finalLayer.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out')
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, std=1e-3)
        if type(layer) == nn.BatchNorm2d:
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)
    

    def forward(self, X):
        out = self.b1(X)
        out = self.b2(out)
        out = self.b3(out)
        out = self.b4(out)
        out = self.b5(out)
        out = self.b6(out)
        out = self.b7(out)
        out = self.b8(out)
        out = self.b9(out)
        out = self.finalLayer(out)

        return out