"""
A convolutional neural network with large number of layers is expensive, both interms of memory and the 
hardware requirement for inference and thus deploying such models in mobile devices is not feasible.

To overcome the above challenge, a group of researchers from Google built a neural network model 
optimized for mobile devices referred as MobileNet. Underlying idea of mobilenet is depthwise
seperable convolutions consisting of depthwise and a pointwise convolution to build lighter models.

MobileNet introduces two hyperparameters

* Width Multiplier

Width muliplier (denoted by α) is a global hyperparameter that is used to construct smaller and less 
computionally expensive models.Its value lies between 0 and 1.For a given layer and value of α, the 
number of input channels 'M' becomes α * M and the number of output channels 'N' becomes α * N hence 
reducing the cost of computation and size of the model at the cost of performance.The computation cost 
and number of parameters decrease roughly by a factor of α2.Some commonly used values of α are 1,0.75,0.5,0.25.

* Resolution Multiplier

The second parameter introduced in MobileNets is called resolution multiplier and is denoted by ρ.This 
hyperparameter is used to decrease the resolution of the input image and this subsequently reduces the 
input to every layer by the same factor. For a given value of ρ the resolution of the input image becomes 
224 * ρ. This reduces the computational cost by a factor of ρ2.

The above parameters helps in trade-off between latency (speed of inference) and accuracy.

MobileNet is 28 layers neural net represented by both the depthwise convolution and pointwise convolution.

"""
import torch.nn as nn

class MobileNetV1(nn.Module):
    def __init__(self, input_channel, n_classes):
        super().__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(input_channel, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7)
        )
        self.fc = nn.Linear(1024, n_classes)

        self.model.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out')
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, std=1e-3)
        if type(layer) == nn.BatchNorm2d:
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
