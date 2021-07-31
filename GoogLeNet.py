"""
In AlexNet, we've used 11x11 Conv, in NIN, we used 1x1 Conv. And in this paper, we identify
among different kernel, which sized convolutional kernels are best. It is the version 1 of Inception
model. 

GoogLeNet introduces the concept of parallel concatenation of networks. We bulid Inception block and 
which is repeated in the architecture.

Some intution on the architecture, since the various different sized filters are at work, different spatial
relations are extracted by different filters efficiently. It also allocates different amt of parameters
across different filters.

* 1×1 convolutions reduce channel dimensionality on a per-pixel level. Maximum pooling reduces the resolution.
* If you're wondering how these dimensions were decided, it is based on trial and error & based on ImageNet 
Dataset

"""
import torch
import torch.nn as nn
from torch.nn import functional as F

class Inception(nn.Module):
	def __init__(self, in_channel, out_channel_1, out_channel_2, out_channel_3, out_channel_4, **kwargs):
		#Four output channel for each parallel block of network
		super().__init__()

		self.p1_1 = nn.Conv2d(in_channel, out_channel_1, kernel_size=1) #1x1Conv

		self.p2_1 = nn.Conv2d(in_channel, out_channel_2[0], kernel_size=1) #1x1Conv
		self.p2_2 = nn.Conv2d(out_channel_2[0], out_channel_2[1], kernel_size=3, padding=1) #3x3Conv

		self.p3_1 = nn.Conv2d(in_channel, out_channel_3[0], kernel_size=1) #1x1Conv
		self.p3_2 = nn.Conv2d(out_channel_3[0], out_channel_3[1], kernel_size=5, padding=2) #5x5Conv

		self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) #3x3 MaxPool
		self.p4_2 = nn.Conv2d(in_channel, out_channel_4, kernel_size=1) #1x1 Conv
			
	def forward(self, x):
		p1 = F.relu(self.p1_1(x))
		p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
		p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
		p4 = F.relu(self.p4_2(self.p4_1(x)))

		return torch.cat((p1, p2, p3, p4), dim=1) 
		#Finally, the outputs along each path are concatenated along the channel dimension and comprise the block’s
	    # output.

class GoogLeNet(nn.Module):
	def __init__(self, input_channel, n_classes):
		super().__init__()
		self.b1 = nn.Sequential(
				nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

		self.b2 = nn.Sequential(
				nn.Conv2d(64, 64, kernel_size=1),
				nn.ReLU(),
				nn.Conv2d(64, 192, kernel_size=3, padding=1),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
		"""
		Explaining Only two Inception block, there are totally 9 Inception Block.

		First Inception Module in b3, 

		in_channel=192, 
		out_channels = {	
			self.p1_1: 64, 
			self.p2_1: 96, self.p2_2: 128, 
			self.p3_1: 16, self.p3_2: 32,
			self.p4_1: 32
			}
		Add all out_channel => 64 + 128 + 32 + 32 = 256, which is our input to next Inception module.

		For each parallel block, the input is 192, and we can see in second and third path we reduce in=192 to out=96 
		and in=192 to out=16 respectively.
		
		Second Inception Module

		in_channel=256, 
		out_channels = {	
			self.p1_1: 128, 
			self.p2_1: 128, self.p2_2: 192, 
			self.p3_1: 32, self.p3_2: 96,
			self.p4_1: 64
			}
		Add all out_channel => 128 + 192 + 96 + 64 = 480, which is our input to next Inception module.

		For each parallel block, the input is 256, and we can see in second and third path we reduce in=256 to out=128 
		and in=256 to out=32 respectively.
		"""
		self.b3 = nn.Sequential(
				Inception(192, 64, (96, 128), (16, 32), 32), 
				Inception(256, 128, (128, 192), (32, 96), 64),
				nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

		self.b4 = nn.Sequential(
				Inception(480, 192, (96, 208), (16, 48), 64),
                Inception(512, 160, (112, 224), (24, 64), 64),
				Inception(512, 128, (128, 256), (24, 64), 64),
                Inception(512, 112, (144, 288), (32, 64), 64),
                Inception(528, 256, (160, 320), (32, 128), 128),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
				)

		self.b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                Inception(832, 384, (192, 384), (48, 128), 128),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

		self.fc = nn.Linear(1024, n_classes)

		self.b1.apply(self.init_weights)
		self.b2.apply(self.init_weights)
		self.b3.apply(self.init_weights)
		self.b4.apply(self.init_weights)
		self.b5.apply(self.init_weights)
		self.fc.apply(self.init_weights)

	def init_weights(self, layer):
		if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
			nn.init.xavier_uniform_(layer.weight)

	def forward(self, x):
		out = self.b1(x)
		out = self.b2(out)
		out = self.b3(out)
		out = self.b4(out)
		out = self.b5(out)
		out = self.fc(out)
		
		return out