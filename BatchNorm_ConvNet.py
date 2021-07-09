"""
Batch Normalization:

https://e2eml.school/batch_normalization.html

* BatchNorm is used for faster convergence.
* In Machine Learning, we perform standardization to bring the features to similar scale 
to avoid few parameters to dominate over others. On similar stance, we use BatchNorm in 
Deep Learning.
* During forward propagation while passing through deeper layers from input to output, 
the distributions of inputs varies in different magnitude thus to avoid this drift in 
distribution we use BatchNorm.


Batch normalization is applied to individual layers (optionally, to all of them) and works as follows:
    In each training iteration, we first normalize the inputs (of batch normalization) by subtracting 
    their mean and dividing by their standard deviation, where both are estimated based on the 
    statistics of the current minibatch. Next, we apply a scale coefficient and a scale offset.
    It is precisely due to this normalization based on batch statistics that batch normalization 
    derives its name.

Batch normalization layers function differently in training mode (normalizing by minibatch statistics) 
and in prediction mode (normalizing by dataset statistics).

In Original paper, For fully-connected layer, BatchNorm is inserted between affine transform and 
Non-Linear activation function. For Convolution layers, batch norm is inserted after conv operation
and non-linear activation function.

Criticism: Authors offered the success of batchnorm is because of reducing the internal covariate shift
as mentioned before interms of normalization, but wide ml researcher disagreed and still debate exists on
why BatchNorm works?

"""
import torch
from torch.nn.modules import batchnorm
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
num_epochs = 10
inputs = 28*28
batch_size = 200
learning_rate = 1.5

def batchNormalization(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    """
    X - dataset
    gamma - scale parameter
    beta - shift parameter
    moving_mean - used during inference 
    moving_var - used during inference
    """
    # Checking if not training mode
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)

    else:
        assert len(X.shape) in (2, 4)
        #Feed-Forward layer
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean)**2).mean(dim=0)

        else:
            #Convolutional Layer
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean)**2).mean(dim=(0, 2, 3), keepdim=True)

        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var

    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)

        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)

        Y, self.moving_mean, self.moving_var = batchNormalization(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var,
            eps=1e-5, momentum=0.9)

        return Y

class LeNet_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, bias=False),
            BatchNorm(6, num_dims=4),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, bias=False),
            BatchNorm(16, num_dims=4),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten())

        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120, bias=False),
            BatchNorm(120, num_dims=2),
            nn.Sigmoid(),
            nn.Linear(120, 84, bias=False),
            BatchNorm(84, num_dims=2),
            nn.Sigmoid(),
            nn.Linear(84, 10, bias=False))

    def forward(self, X):
        out = self.conv(X)
        out = self.fc(out)

        return out


#datasets

path = "/home/mayur/Desktop/Pytorch/data"

train_dataset = torchvision.datasets.MNIST(root=path, train=True,
											transform = transforms.ToTensor(),
											download=False)

test_dataset = torchvision.datasets.MNIST(root=path, train=False,
										transform = transforms.ToTensor(),
										download=False)

train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset,
												batch_size=batch_size,
												shuffle=True)

test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset,
												batch_size=batch_size,
												shuffle=True)

model = LeNet_BN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_dataloader)
Loss  = []
start_time = time.time()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            Loss.append(loss.cpu().detach().numpy())

model.eval()
with torch.no_grad():
	correct = 0
	total = 0
	for images, labels in test_dataloader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	print(f'Accuracy: {(correct*100)/total}')
print(f'Total Number of Parameters of LeNet with BatchNorm is {sum(p.numel() for p in model.parameters())}') #44878
print(f'Total time taken: {time.time() - start_time}')

plt.scatter(range(len(Loss)), Loss, color='blue', label='Loss')
plt.title("Loss Over Iterations")
plt.legend()
plt.show()
