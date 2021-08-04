import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

def optim(model_name, model, lr):
    if model_name == 'resnet':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer, T_max=200)
        return optimizer, scheduler
    