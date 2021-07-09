import torch
import torchvision
import torchvision.transforms as transforms

class initialize_dataset:
    def __init__(self, image_resolution=224, batch_size=128, grayscale=True):
        self.image_resolution= image_resolution
        self.batch_size=batch_size
        self.grayscale=grayscale
        
    def load_dataset(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.image_resolution, self.image_resolution))])
        path = "/home/mayur/Desktop/Pytorch/data"

        if self.grayscale:
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