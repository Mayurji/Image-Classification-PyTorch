"""
BatchNorm - https://en.wikipedia.org/wiki/Batch_normalization

It is a simple CNN architecture with combination of ideas from different researches
with batchNorm, activation function, kernels etc.

"""
import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class CNN(nn.Module):
	def __init__(self, in_channel):
		super().__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channel, 16, kernel_size=5, stride=1, padding=2), # Transforming 28*28 into 16 kernels of size(28 * 28) since padding=2 and stride=1.
			nn.BatchNorm2d(16), # Reduce Internal Covariate Shift and acts as regularizer.
			nn.ReLU(), # max(0, +ve)
			nn.MaxPool2d(kernel_size=2, stride=2)) # Reducing the image dimension by half. Because there is no padding and stride=2.

		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), # Transforming 14*14 into 32 kernels of size(14 * 14) since padding=2 and stride=1.
			nn.BatchNorm2d(32), # Reduce Internal Covariate Shift and acts as regularizer.
			nn.ReLU(), # max(0, +ve)
			nn.MaxPool2d(kernel_size=2, stride=2)) # Reducing the image dimension by half, to (32 * 7 * 7). Because there is no padding and stride=2.

		self.fc = nn.Linear(7*7*32, 10) # Tranforming 7 * 7 * 32 into n_classes(10).

	def forward(self, x):
		h1 = self.layer1(x)
		h2 = self.layer2(h1)
		out = h2.reshape(h2.size(0), -1)
		out = self.fc(out)
		return out