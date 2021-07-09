import argparse

import torch
import yaml

from convNet import CNN
from AlexNet import AlexNet
from DenseNet import DenseNet
from GoogLeNet import GoogLeNet
from ResNet import ResNet
from SENet import SENet
from VGG import VGG11
from NiN import NIN

from dataset import initialize_dataset
from train_test import testing, training

try:
    stream = open("config.yaml", 'r')
    config = yaml.safe_load(stream)
except FileNotFoundError:
    print("Config file missing")

"""Device Selection"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" Initialize model based on command line argument """
model_parser = argparse.ArgumentParser(description='Select the Model or Models', usage='[option] model_name')
model_parser.add_argument('--model', type=str, required=True)
args = model_parser.parse_args()

"""Dataset Initialization"""
data_initialization = initialize_dataset(image_resolution=config['image_resolution'], batch_size=config['batch_size'], 
                      grayscale=config['grayscale'])
train_dataloader, test_dataloader = data_initialization.load_dataset()

if config['grayscale']:
    input_channel = 1
else:
    input_channel = 3

"""Model Initialization"""
if args.model == 'vgg':
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    model = VGG11(VGGArchitecture=conv_arch, input_channel=input_channel, 
            image_resolution=config['image_resolution']).to(device)
elif args.model == 'alexnet':
    model = AlexNet(input_channel=input_channel).to(device)
elif args.model == 'senet':
    model = SENet(in_channel=input_channel).to(device)
elif args.model == 'resnet':
    model = ResNet(input_channel=input_channel).to(device)
elif args.model == 'densenet':
    model = DenseNet(input_channel=input_channel).to(device)
elif args.model == 'googlenet':
    model = GoogLeNet(input_channel=input_channel).to(device)
elif args.model == 'nin':
    model = NIN(input_channel=input_channel).to(device)
elif args.model == 'cnn':
    model = CNN(input_channel=input_channel).to(device)


print(f'Total Number of Parameters of {args.model.capitalize()} is {round((sum(p.numel() for p in model.parameters()))/1000000, 2)}M')

trainer = training(model=model, optimizer='sgd', learning_rate=config['learning_rate'],train_dataloader=train_dataloader, num_epochs=config['num_epochs'])
trainer.train()

evaluating = testing(model=model, test_dataloader=test_dataloader)
evaluating.test()