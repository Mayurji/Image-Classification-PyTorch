import torch
import torchvision
from augmentations import augmentation, ContrastiveAugmentation
import torchvision.transforms as transforms

class initialize_dataset:
    def __init__(self, image_resolution=224, batch_size=128, MNIST=True):
        self.image_resolution= image_resolution
        self.batch_size=batch_size
        self.MNIST = MNIST

    def load_dataset(self, transform=False):
        path = "/home/mayur/Desktop/Pytorch/data"
        #path = './data'
        if transform:
            transform = augmentation(image_resolution=self.image_resolution)
        elif self.MNIST:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.image_resolution, self.image_resolution)),
                                            transforms.Normalize((0.1307,), (0.3081,))])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.image_resolution, self.image_resolution)),
                        transforms.RandomHorizontalFlip(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        if self.MNIST:
            train_dataset = torchvision.datasets.MNIST(root=path, train=True,
                                                        transform = transform,
                                                        download=True)
            test_dataset = torchvision.datasets.MNIST(root=path, train=False,
                                                    transform = transform,
                                                    download=True)
        else:
            train_dataset = torchvision.datasets.CIFAR10(root=path, train=True,
                                                        transform = transform,
                                                        download=True)
            test_dataset = torchvision.datasets.CIFAR10(root=path, train=False,
                                                    transform = transform,
                                                    download=True)

        train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=True)

        return train_dataloader, test_dataloader
