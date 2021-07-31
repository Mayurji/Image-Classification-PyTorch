"""
NIN - Network In Network

**To use an MLP on the channels for each pixel separately.**

The idea behind NiN is to apply a fully-connected layer at each pixel location (for each height and width). 
If we tie the weights across each spatial location, we could think of this as a 1×1 convolutional layer 
or as a fully-connected layer acting independently on each pixel location. Another way to view this is to think
of each element in the spatial dimension (height and width) as equivalent to an example and a channel as equivalent
to a feature.

NIN has introducted the 1x1 Convolution.

Smaller batch size results in better performance even though it is slow.
"""
import torch.nn as nn

class NIN(nn.Module):
	def __init__(self, input_channel, n_classes):
		super().__init__()

		def NINBlock(input_channel, out_channel, kernel_size, strides, padding):
			return nn.Sequential(
				nn.Conv2d(input_channel, out_channel, kernel_size=kernel_size, stride=strides, padding=padding),
				nn.ReLU(),
				nn.Conv2d(out_channel, out_channel, kernel_size=1),
				nn.ReLU(),
				nn.Conv2d(out_channel, out_channel, kernel_size=1),
				nn.ReLU())

		self.layers = nn.Sequential(
			NINBlock(input_channel, 96, kernel_size=11, strides=4, padding=0),
			nn.MaxPool2d(3, stride=2),
			NINBlock(96, 256, kernel_size=5, strides=1, padding=2),
			nn.MaxPool2d(3, stride=2),
			NINBlock(256, 384, kernel_size=3, strides=1, padding=1),
			nn.MaxPool2d(3, stride=2),nn.Dropout(0.5),
			NINBlock(384, n_classes, kernel_size=3, strides=1, padding=1),
			nn.AdaptiveAvgPool2d((1,1)),nn.Flatten())
		self.layers.apply(self.init_weights)

	def init_weights(self, layer):
		if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
			nn.init.xavier_uniform_(layer.weight)

	def forward(self, x):
		out = self.layers(x)
		return out