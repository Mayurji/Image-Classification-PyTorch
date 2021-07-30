"""

"The Inception deep convolutional architecture was introduced as GoogLeNet, here named Inception-v1. 
Later the Inception architecture was refined in various ways, first by the introduction of batch 
normalization (Inception-v2). Later by additional factorization ideas in the third iteration 
which is referred as Inception-v3.”

Factorizing Convolution: Idea is to decrease the number of connections/parameters without reducing
the performance.

* Factorizing large kernel into two similar smaller kernels
    - Using 1 5x5 kernel, number of parameters is 5x5=25
    - Using 2 3x3 kernel instead of one 5x5, gives 3x3 + 3x3 = 18 parameters.
    - Number of parameter is reduced by 28%.

* Factorizing large kernel into two assimilar smaller kernels
    - By using 3×3 filter, number of parameters = 3×3=9
    - By using 3×1 and 1×3 filters, number of parameters = 3×1+1×3=6
    - Number of parameters is reduced by 33%

* If we looking into InceptionV1 i.e. GoogLeNet, we have inception block which uses 5x5 kernel and 3x3 
kernel, this technique can reduce the number of parameters.

Other Changes:

From InceptionV1, we bring in Auxillary classifier which acts as regularizer. We also see, efficient
grid size reduction using factorization instead of standard pooling which expensive and greedy operation.
Label smoothing, to prevent a particular label from dominating all other class.

Reference: 

* https://sh-tsang.medium.com/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c
* https://paperswithcode.com/method/auxiliary-classifier#
* https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/inceptionv3.py
"""
import torch
import torch.nn as nn

class BasicConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, bias=False, **kwargs),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return self.conv(X)

#Inception Block from InceptionV1
class InceptionA(nn.Module):
    def __init__(self, input_channel, pool_features):
        super().__init__()
        self.branch1x1 = BasicConvBlock(input_channel, 64, kernel_size=1)

        self.branch5x5 = nn.Sequential(
            BasicConvBlock(input_channel, 48, kernel_size=1),
            BasicConvBlock(48, 64, kernel_size=5, padding=2)
        )

        self.branch3x3 = nn.Sequential(
            BasicConvBlock(input_channel, 64, kernel_size=1),
            BasicConvBlock(64, 96, kernel_size=3, padding=1),
            BasicConvBlock(96, 96, kernel_size=3, padding=1)
        )

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConvBlock(input_channel, pool_features, kernel_size=3, padding=1)
        )

    def forward(self, X):
         #x -> 1x1(same)
        branch1x1 = self.branch1x1(X)

        #x -> 1x1 -> 5x5(same)
        branch5x5 = self.branch5x5(X)
        #branch5x5 = self.branch5x5_2(branch5x5)

        #x -> 1x1 -> 3x3 -> 3x3(same)
        branch3x3 = self.branch3x3(X)

        #x -> pool -> 1x1(same)
        branchpool = self.branchpool(X)

        outputs = [branch1x1, branch5x5, branch3x3, branchpool]

        return torch.cat(outputs, 1)

#Factorization
class InceptionB(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.branch3x3 = BasicConvBlock(input_channels, 384, kernel_size=3, stride=2)
        self.branch3x3stack = nn.Sequential(
            BasicConvBlock(input_channels, 64, kernel_size=1),
            BasicConvBlock(64, 96, kernel_size=3, padding=1),
            BasicConvBlock(96, 96, kernel_size=3, stride=2)
        )
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, X):
        #x - > 3x3(downsample)
        branch3x3 = self.branch3x3(X)

        #x -> 3x3 -> 3x3(downsample)
        branch3x3stack = self.branch3x3stack(X)

        #x -> avgpool(downsample)
        branchpool = self.branchpool(X)

        """We can use two parallel stride 2 blocks: P and C. P is a pooling
        #layer (either average or maximum pooling) the activation, both of
        #them are stride 2 the filter banks of which are concatenated as in
        #figure 10."""
        outputs = [branch3x3, branch3x3stack, branchpool]

        return torch.cat(outputs, 1)

#Factorizing Convolutions with Large Filter Size
class InceptionC(nn.Module):
    def __init__(self, input_channels, channels_7x7):
        super().__init__()
        self.branch1x1 = BasicConvBlock(input_channels, 192, kernel_size=1)

        c7 = channels_7x7

        #In theory, we could go even further and argue that one can replace any n × n
        #convolution by a 1 × n convolution followed by a n × 1 convolution and the
        #computational cost saving increases dramatically as n grows (see figure 6).
        self.branch7x7 = nn.Sequential(
            BasicConvBlock(input_channels, c7, kernel_size=1),
            BasicConvBlock(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConvBlock(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch7x7stack = nn.Sequential(
            BasicConvBlock(input_channels, c7, kernel_size=1),
            BasicConvBlock(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConvBlock(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            BasicConvBlock(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConvBlock(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConvBlock(input_channels, 192, kernel_size=1),
        )

    def forward(self, x):

        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        #x -> 1layer 1*7 and 7*1 (same)
        branch7x7 = self.branch7x7(x)

        #x-> 2layer 1*7 and 7*1(same)
        branch7x7stack = self.branch7x7stack(x)

        #x-> avgpool (same)
        branchpool = self.branch_pool(x)

        outputs = [branch1x1, branch7x7, branch7x7stack, branchpool]

        return torch.cat(outputs, 1)

class InceptionD(nn.Module):

    def __init__(self, input_channels):
        super().__init__()

        self.branch3x3 = nn.Sequential(
            BasicConvBlock(input_channels, 192, kernel_size=1),
            BasicConvBlock(192, 320, kernel_size=3, stride=2)
        )

        self.branch7x7 = nn.Sequential(
            BasicConvBlock(input_channels, 192, kernel_size=1),
            BasicConvBlock(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConvBlock(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            BasicConvBlock(192, 192, kernel_size=3, stride=2)
        )

        self.branchpool = nn.AvgPool2d(kernel_size=3, stride=2)

    def forward(self, x):

        #x -> 1x1 -> 3x3(downsample)
        branch3x3 = self.branch3x3(x)

        #x -> 1x1 -> 1x7 -> 7x1 -> 3x3 (downsample)
        branch7x7 = self.branch7x7(x)

        #x -> avgpool (downsample)
        branchpool = self.branchpool(x)

        outputs = [branch3x3, branch7x7, branchpool]

        return torch.cat(outputs, 1)


#same
class InceptionE(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.branch1x1 = BasicConvBlock(input_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConvBlock(input_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConvBlock(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConvBlock(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3stack_1 = BasicConvBlock(input_channels, 448, kernel_size=1)
        self.branch3x3stack_2 = BasicConvBlock(448, 384, kernel_size=3, padding=1)
        self.branch3x3stack_3a = BasicConvBlock(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3stack_3b = BasicConvBlock(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConvBlock(input_channels, 192, kernel_size=1)
        )

    def forward(self, x):

        #x -> 1x1 (same)
        branch1x1 = self.branch1x1(x)

        # x -> 1x1 -> 3x1
        # x -> 1x1 -> 1x3
        # concatenate(3x1, 1x3)
        #"""7. Inception modules with expanded the filter bank outputs.
        #This architecture is used on the coarsest (8 × 8) grids to promote
        #high dimensional representations, as suggested by principle
        #2 of Section 2."""
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3)
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        # x -> 1x1 -> 3x3 -> 1x3
        # x -> 1x1 -> 3x3 -> 3x1
        #concatenate(1x3, 3x1)
        branch3x3stack = self.branch3x3stack_1(x)
        branch3x3stack = self.branch3x3stack_2(branch3x3stack)
        branch3x3stack = [
            self.branch3x3stack_3a(branch3x3stack),
            self.branch3x3stack_3b(branch3x3stack)
        ]
        branch3x3stack = torch.cat(branch3x3stack, 1)

        branchpool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch3x3stack, branchpool]

        return torch.cat(outputs, 1)

class InceptionV3(nn.Module):

    def __init__(self, input_channel, n_classes=10):
        super().__init__()
        self.Conv2d_1a_3x3 = BasicConvBlock(input_channel, 32, kernel_size=3, padding=1)
        self.Conv2d_2a_3x3 = BasicConvBlock(32, 32, kernel_size=3, padding=1)
        self.Conv2d_2b_3x3 = BasicConvBlock(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConvBlock(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConvBlock(80, 192, kernel_size=3)

        #naive inception module
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        #downsample
        self.Mixed_6a = InceptionB(288)

        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        #downsample
        self.Mixed_7a = InceptionD(768)

        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        #6*6 feature size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d()
        self.linear = nn.Linear(2048, n_classes)

    def forward(self, x):

        #32 -> 30
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)

        #30 -> 30
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        #30 -> 14
        #Efficient Grid Size Reduction to avoid representation
        #bottleneck
        x = self.Mixed_6a(x)

        #14 -> 14
        #"""In practice, we have found that employing this factorization does not
        #work well on early layers, but it gives very good results on medium
        #grid-sizes (On m × m feature maps, where m ranges between 12 and 20).
        #On that level, very good results can be achieved by using 1 × 7 convolutions
        #followed by 7 × 1 convolutions."""
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        #14 -> 6
        #Efficient Grid Size Reduction
        x = self.Mixed_7a(x)

        #6 -> 6
        #We are using this solution only on the coarsest grid,
        #since that is the place where producing high dimensional
        #sparse representation is the most critical as the ratio of
        #local processing (by 1 × 1 convolutions) is increased compared
        #to the spatial aggregation."""
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)

        #6 -> 1
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
