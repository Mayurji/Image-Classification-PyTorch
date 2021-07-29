"""
MobileNetV2

MobileNet architecture is built with the idea to make neural networks feasible on mobile devices.
MobileNet introduces the idea of depthwise separable convolution, which is depthwise conv followed
by pointwise conv.

What's New 

With MobileNetV2, the architecture introduces the concept of inverted residual, where the residual
connections are made between the bottleneck layers. The intermediate expansion layer uses lightweight 
depthwise convolutions to filter features as a source of non-linearity.

A traditional Residual Block has a wide -> narrow -> wide structure with the number of channels. The 
input has a high number of channels, which are compressed with a 1x1 convolution. The number of 
channels is then increased again with a 1x1 convolution so input and output can be added.

In contrast, an Inverted Residual Block follows a narrow -> wide -> narrow approach, hence the inversion. 
We first widen with a 1x1 convolution, then use a 3x3 depthwise convolution (which greatly reduces the 
number of parameters), then we use a 1x1 convolution to reduce the number of channels so input and output 
can be added.

"""
import torch.nn as nn
import numpy as np
import math

def conv3x3(input_channel, output_channel, stride):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, 3, stride, 1, bias=False),
        nn.BatchNorm2d(output_channel),
        nn.ReLU6(inplace=True)
    )

def conv1x1(input_channel, output_channel):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output_channel),
        nn.ReLU6(inplace=True)
    )

def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class InvertedResidual(nn.Module):
    def __init__(self, input_channel, out_channel, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2], 'Stride value is greater than 2'

        hidden_dimension = round(input_channel * expand_ratio)
        self.identity = stride == 1 and input_channel == out_channel

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                #depthwise convolution
                nn.Conv2d(hidden_dimension, hidden_dimension, 3, stride, 1, groups=hidden_dimension, bias=False),
                nn.BatchNorm2d(hidden_dimension),
                nn.ReLU6(inplace=True),

                #pointwise linear
                nn.Conv2d(hidden_dimension, out_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.conv = nn.Sequential(
                # pointwise conv
                nn.Conv2d(input_channel, hidden_dimension, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dimension),
                nn.ReLU6(inplace=True),
                
                # depthwise conv
                nn.Conv2d(hidden_dimension, hidden_dimension, 3, stride, 1, groups=hidden_dimension, bias=False),
                nn.BatchNorm2d(hidden_dimension),
                nn.ReLU6(inplace=True),
                
                # pointwise-linear
                nn.Conv2d(hidden_dimension, out_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, input_channel,  n_classes=10, width_multipler=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        first_channel = 32
        last_channel = 1280
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.last_channel = make_divisible(last_channel * width_multipler) if width_multipler > 1.0 else last_channel
        self.features = [conv3x3(input_channel, first_channel, 2)]

        for t, c, n, s in self.cfgs:
            output_channel = make_divisible(c * width_multipler) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(first_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(first_channel, output_channel, 1, expand_ratio=t))
                first_channel = output_channel
        # building last several layers
        self.features.append(conv1x1(first_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_classes)
        self._initialize_weights()


    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
