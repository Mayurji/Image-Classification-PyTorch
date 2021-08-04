import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, CyclicLR

def optim(model_name, model, lr):
    """
    The script tries to replicates the optimizer used in the paper, sometimes it resulted
    in Nan loss and I've modified the learning rate scheduler to make it work.
    """
    if model_name == 'resnet':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer, T_max=200)
        return optimizer, scheduler
    if model_name == 'alexnet':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005, momentum=0.9)
        scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular")
        return optimizer, scheduler
    if model_name == 'vggnet':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005, momentum=0.9)
        scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular")
        return optimizer, scheduler