"""
SqueezeNet

This network is known for providing AlexNet-Level accuracy at 50 times fewer parameters.
This small architecture offers three major advantages, first, it requires less bandwidth
for exporting the model and then it requires less communication between server during 
distributed training and more feasible to deploy on FPGAs.

Archiecture creates Fire module containing a squeeze convolution layer (which has only 
1×1 filters), feeding into an expand layer that has a mix of 1×1 and 3×3 convolution filters.

To reduce the parameters the architecture follows design strategies

1. Using Conv1x1 over Conv3x3
2. Decreasing number of channels using Squeeze Layers
3. Downsample late in the network, such that convolution
layers have large activation maps.

1 and 2 helps in reducing the parameters, and 3 helps in higher classification accuracy
because of large activation maps.

Reference: https://towardsdatascience.com/review-squeezenet-image-classification-e7414825581a

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Fire(nn.Module):
    def __init__(self, in_channel, squeeze_channel, expand_channel):
        super().__init__()
        
        #squeeze conv1x1
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squeeze_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(squeeze_channel),
            nn.ReLU(inplace=True),
        )

        #expand conv1x1
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channel, expand_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(expand_channel)
            )
        
        #expand conv3x3
        self.expand3x3 = nn.Sequential(    
            nn.Conv2d(squeeze_channel, expand_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(expand_channel)
            )

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

    def forward(self, x):
        squeezeOut = self.squeeze(x)
        expandOut_1x1 = self.expand1x1(squeezeOut)
        expandOut_3x3 = self.expand3x3(squeezeOut)
        output = torch.cat([expandOut_1x1, expandOut_3x3], 1)
        output = F.relu(output)
        return output

class SqueezeNet(nn.Module):
    def __init__(self, input_channel, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, 96, kernel_size=3, stride=1, padding=1), # 32
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 16
        )
        self.Fire2 = Fire(96, 16, 64)
        self.Fire3 = Fire(128, 16, 64)
        self.Fire4 = Fire(128, 32, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
        self.Fire5 = Fire(256, 32, 128)
        self.Fire6 = Fire(256, 48, 192)
        self.Fire7 = Fire(384, 48, 192)
        self.Fire8 = Fire(384, 64, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
        self.Fire9 = Fire(512, 64, 256)
        self.conv2 = nn.Conv2d(512, n_classes, kernel_size=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.LogSoftmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv(x)
        x = self.Fire2(x)
        x = self.Fire3(x)
        x = self.Fire4(x)
        x = self.maxpool2(x)
        x = self.Fire5(x)
        x = self.Fire6(x)
        x = self.Fire7(x)
        x = self.Fire8(x)
        x = self.maxpool3(x)
        x = self.Fire9(x)
        x = F.dropout(x, 0.5)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.softmax(x)
        x = x.view(x.size(0), -1)
        return x
