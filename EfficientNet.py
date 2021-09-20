"""
EfficientNet

CNN models improves its ability to classify images by either increasing the depth of the network or 
by increasing the resolution of the images to capture finer details of the image or by increasing
width of the network by increasing the number of channels. For instance, ResNet-18 to ResNet-152 has 
been built around these ideas.

Now there is limit to each of these factors mentioned above and with increasing requirement of computational 
power. To overcome these challenges, researchers introducted the concept of compound scaling, which scales
all the three factors moderately leading us to build EfficientNet.

EfficientNet scales all the three factors i.e. depth, width and resolution but how to scale it? we can 
scale each factor equally but this wouldn't work if our task requires fine grained estimation and which 
requries more depth. 

Complex CNN architectures are built using multiple conv blocks and each block needs to be consistent with 
previous and next block, thus each layers in the block are scaled evenly.

EfficientNet-B0 Architecture

* Basic ConvNet Block (AlexNet)
* Inverted Residual (MobileNetV2)
* Squeeze and Excitation Block (Squeeze and Excitation Network)

EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all 
dimensions of depth/width/resolution using a compound coefficient. Unlike conventional practice that arbitrary 
scales these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution 
with a set of fixed scaling coefficients. For example, if we want to use 2^N times more computational resources, 
then we can simply increase the network depth by alpha^N, width by beta^N, and image size by gamma^N, where 
alpha, beta and gamma, are constant coefficients determined by a small grid search on the original small model. 
EfficientNet uses a compound coefficient phi to uniformly scales network width, depth, and resolution in a 
principled way.

The compound scaling method is justified by the intuition that if the input image is bigger, then the network 
needs more layers to increase the receptive field and more channels to capture more fine-grained patterns on 
the bigger image.

The base EfficientNet-B0 network is based on the inverted bottleneck residual blocks of MobileNetV2, in addition 
to squeeze-and-excitation blocks.

EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), 
and 3 other transfer learning datasets, with an order of magnitude fewer parameters.

Interesting Stuff:

Now, the most interesting part of EfficientNet-B0 is that the baseline architecture is designed by Neural 
Architecture Search(NAS). NAS is a wide topic and is not feasible to be discussed here. We can simply 
consider it as searching through the architecture space for underlying base architecture like ResNet or 
any other architecture for that matter. And on top of that, we can use grid search for finding the scale 
factor for Depth, Width and Resolution. Combining NAS and with compound scaling leads us to EfficientNet. 
Model is evaluated by comparing accuracy over the # of FLOPS(Floating point operations per second).

Recommended Reading for NAS: https://lilianweng.github.io/lil-log/2020/08/06/neural-architecture-search.html
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

def roundChannels(c, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    
    return new_c

def roundRepeats(r):
    return int(math.ceil(r))

def dropPath(x, drop_probability, training):
    if drop_probability > 0 and training:
        keep_probability = 1 - drop_probability
        if x.is_cuda:
            mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_probability))
        else:
            mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_probability))

        x.div_(keep_probability)
        x.mul_(mask)

    return x

def batchNorm(channels, eps=1e-3, momentum=0.01):
    return nn.BatchNorm2d(channels, eps=eps, momentum=momentum)

#CONV3x3
def conv3x3(in_channel, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channels, 3, stride, 1, bias=False),
        batchNorm(out_channels),
        Swish()
    )

#CONV1x1
def conv1x1(in_channel, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channels, 1, 1, 0, bias=False),
        batchNorm(out_channels),
        Swish()
    )

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel, squeeze_channel, se_ratio):
        super().__init__()
        squeeze_channel = squeeze_channel * se_ratio
        if not squeeze_channel.is_integer():
            raise ValueError('channels must be divisible by 1/se_ratio')

        squeeze_channel = int(squeeze_channel)
        self.se_reduce = nn.Conv2d(channel, squeeze_channel, 1, 1, 0, bias=True)
        self.non_linear1 = Swish()
        self.se_excite = nn.Conv2d(squeeze_channel, channel, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear1(self.se_excite(y))
        y = x * y
        return y

class MBConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, expand_ratio, se_ratio, drop_path_rate):
        super().__init__()
        expand = (expand_ratio != 1)
        expand_channel = in_channel * expand_ratio
        se = (se_ratio != 0)
        self.residual_connection = (stride == 1 and in_channel == out_channel)
        self.drop_path_rate = drop_path_rate

        conv=[]

        if expand:
            pw_expansion = nn.Sequential(
                nn.Conv2d(in_channel, expand_channel, 1, 1, 0, bias=False),
                batchNorm(expand_channel),
                Swish()
            )
            conv.append(pw_expansion)

        #depthwise convolution
        dw = nn.Sequential(
            nn.Conv2d(expand_channel, expand_channel, kernel_size, stride, kernel_size//2, groups=expand_channel, bias=False),
            batchNorm(expand_channel),
            Swish()
        )
        conv.append(dw)

        if se:
            squeeze_excite = SqueezeAndExcitation(expand_channel, in_channel, se_ratio)
            conv.append(squeeze_excite)
        
        pw_projection = nn.Sequential(
            nn.Conv2d(expand_channel, out_channel, 1, 1, 0, bias=False),
            batchNorm(out_channel)
        )
        conv.append(pw_projection)
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.residual_connection:
            return x + dropPath(self.conv(x), self.drop_path_rate, self.training)
        else:
            return self.conv(x)

class EfficientNet(nn.Module):
    cfg = [
        #(in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, repeats)
        [32,  16,  3, 1, 1, 0.25, 1],
        [16,  24,  3, 2, 6, 0.25, 2],
        [24,  40,  5, 2, 6, 0.25, 2],
        [40,  80,  3, 2, 6, 0.25, 3],
        [80,  112, 5, 1, 6, 0.25, 3],
        [112, 192, 5, 2, 6, 0.25, 4],
        [192, 320, 3, 1, 6, 0.25, 1]
    ]
    def __init__(self, input_channels, param, n_classes, stem_channels=32, feature_size=1280, drop_connect_rate=0.2):
        super().__init__()

        # scaling width 
        width_coefficient = param[0]
        if width_coefficient != 1.0:
            stem_channels = roundChannels(stem_channels * width_coefficient)
            for conf in self.cfg:
                conf[0] = roundChannels(conf[0] * width_coefficient)
                conf[1] = roundChannels(conf[1] * width_coefficient)

        # scaling depth
        depth_coefficient = param[1]
        if depth_coefficient != 1.0:
            for conf in self.cfg:
                conf[6] = roundRepeats(conf[6] * depth_coefficient)

        #scaling resolution
        input_size = param[2]

        self.stem_conv = conv3x3(input_channels, stem_channels, 2)

        #total blocks
        total_blocks = 0
        for conf in self.cfg:
            total_blocks += conf[6]

        blocks = []
        for in_channel, out_channel, kernel_size, stride, expand_ratio, se_ratio, repeats in self.cfg:
            
            drop_rate = drop_connect_rate * (len(blocks) /  total_blocks)
            blocks.append(MBConvBlock(in_channel, out_channel, kernel_size, stride, expand_ratio, se_ratio, drop_rate))
            for _ in range(repeats-1):
                drop_rate = drop_connect_rate * (len(blocks) / total_blocks)
                blocks.append(MBConvBlock(out_channel, out_channel, kernel_size, 1, expand_ratio, se_ratio, drop_rate))
        self.blocks = nn.Sequential(*blocks)

        self.head_conv = conv1x1(self.cfg[-1][1], feature_size)
        #self.avgpool = nn.AvgPool2d(input_size//32, stride=1)
        self.dropout = nn.Dropout(param[3])
        self.classifier = nn.Linear(feature_size, n_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        x = torch.mean(x, (2, 3))
        x = self.dropout(x)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
