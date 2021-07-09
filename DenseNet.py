"""
In ResNet, we see how the skip connection added as identity function from the inputs
to interact with the Conv layers. But in DenseNet, we see instead of adding skip 
connection to Conv layers, we can append or concat the output of identity function
with output of Conv layers.

In ResNet, it is little tedious to make the dimensions to match for adding the skip
connection and Conv Layers, but it is much simpler in DenseNet, as we concat the 
both the X and Conv's output.

The key idea or the reason its called DenseNet is because the next layers not only get
the input from previous layer but also preceeding layers before the previous layer. So 
the next layer becomes dense as it loaded with output from previous layers.

Check Figure 7.7.2 from https://d2l.ai/chapter_convolutional-modern/densenet.html for 
why DenseNet is Dense?

Two blocks comprise DenseNet, one is DenseBlock for concat operation and other is 
transition layer for controlling channels meaning dimensions (recall 1x1 Conv).
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms as transforms

def convBlock(in_channel, out_channel):
    return nn.Sequential(
        nn.BatchNorm2d(in_channel), nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
    )

class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channel, out_channel):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(convBlock(out_channel*i + in_channel, out_channel))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)

            X = torch.cat((X, Y),dim=1) #Concat on Channel Dimension

        return X

def transitionBlock(in_channel, out_channel):
    return nn.Sequential(
        nn.BatchNorm2d(in_channel), nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )

class DenseNet(nn.Module):
    def __init__(self, in_channel, out_channel, growth_rate, numConvDenseBlock):
        super().__init__()
        b2 = []
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(out_channel), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        for i, num_convs in enumerate(numConvDenseBlock):
            b2.append(DenseBlock(num_convs=num_convs, in_channel=out_channel, 
                            out_channel=growth_rate))
            out_channel += num_convs * growth_rate

            if i != len(numConvDenseBlock)-1:
                b2.append(transitionBlock(out_channel, out_channel // 2))
                out_channel = out_channel // 2

        self.finalLayer = nn.Sequential(*b2, nn.BatchNorm2d(out_channel),
                        nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),
                        nn.Linear(out_channel, 10)
                        )
        self.b1.apply(self.init_weights)
        self.finalLayer.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, X):
        out = self.b1(X)
        out = self.finalLayer(out)

        return out