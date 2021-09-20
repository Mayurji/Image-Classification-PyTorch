"""
Paper: EfficientNetV2: Smaller Models and Faster Training by Mingxing Tan, Quoc V. Le

Training efficiency has gained significant interests recently. For instance, 
NFNets aim to improve training efficiency by removing the expensive batch normalization; 
Several recent works focus on improving training speed by adding attention layers into 
convolutional networks (ConvNets); Vision Transformers improves training efficiency on 
large-scale datasets by using Transformer blocks. However, these methods often come with
significant overheads.

To develop these models, it uses a combination of training-aware neural search(NAS) and 
scaling, to jointly optimize training speed and parameter efficiency.

Drawbracks of previous version of EfficientNets

1. training with very large image sizes is slow. 
2. depthwise convolutions are slow in early layers.
3. equally scaling up every stage is sub-optimal. 

Whats New With EfficientNetV2

Based on the above observations, V2 is designed on a search space enriched with additional 
ops such as Fused-MBConv, and apply training-aware NAS and scaling to jointly optimize model 
accuracy, training speed, and parameter size. EfficientNetV2, train up to 4x faster than 
prior models, while being up to 6.8x smaller in parameter size.

To further increase the training speed, it uses progressive increase image size, previously
done by FixRes, Mix&Match. The only difference between the current approach from the previous 
approach is the use of adaptive regularization as the image size is increased.

"""
import torch
import torch.nn as nn
import math

def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    
    new_v = max(min_value, int(v + divisor/2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor

    return new_v

if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    class SiLU:
        def forward(self, x):
            return x * torch.sigmoid(x)

class SELayer(nn.Module):
    def __init__(self, input, output, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(output, make_divisible(input//reduction, 8)),
            SiLU(),
            nn.Linear(make_divisible(input//reduction, 8), output),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def conv3x3_bn(input, output, stride):
    return nn.Sequential(
        nn.Conv2d(input, output, 3, stride, 1, bias=False),
        nn.BatchNorm2d(output),
        SiLU()
    )

def conv1x1_bn(input, output):
    return nn.Sequential(
        nn.Conv2d(input, output, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output),
        SiLU()
    )

class MBConv(nn.Module):
    def __init__(self, input, output, stride, expand_ratio, use_se) -> None:
        super().__init__()
        assert stride in [1, 2]

        hidden_dimension = round(input * expand_ratio)
        self.identity = stride == 1 and input == output

        if use_se: #MBCONV
            self.conv = nn.Sequential(
                nn.Conv2d(input, hidden_dimension, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dimension),
                SiLU(),

                nn.Conv2d(hidden_dimension, hidden_dimension, 3, stride, 1, groups=hidden_dimension, bias=False),
                nn.BatchNorm2d(hidden_dimension),
                SiLU(),
                SELayer(input, hidden_dimension),
                nn.Conv2d(hidden_dimension, output, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output)
            )
        else: #Fused-MBConv
            self.conv = nn.Sequential(
                nn.Conv2d(input, hidden_dimension, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dimension),
                SiLU(),
                nn.Conv2d(hidden_dimension, output, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output)
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EfficientNetV2(nn.Module):
    def __init__(self, cfgs, in_channel, num_classes=10, width_multiplier=1.) -> None:
        super().__init__()
        self.cfgs = cfgs

        input_channel = make_divisible(24 * width_multiplier, 8)
        layers = [conv3x3_bn(in_channel, input_channel, 2)]

        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = make_divisible(c * width_multiplier, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i==0 else 1, t, use_se))
                input_channel = output_channel
        
        self.features = nn.Sequential(*layers)
        output_channel = make_divisible(1792 * width_multiplier, 8) if width_multiplier > 1.0 else 1792
        self.conv = conv1x1_bn(input_channel, output_channel)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self.initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def initialize_weights(self):
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
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()