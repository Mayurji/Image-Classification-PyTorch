import argparse

import torch
import yaml
from torchstat import stat

from convNet import CNN
from AlexNet import AlexNet
from DenseNet import DenseNet
from GoogLeNet import GoogLeNet
from ResNet import ResNet
from SENet import SENet
from VGG import VGG11
from NiN import NIN
from MLPMixer import MLPMixer
from MobileNetV1 import MobileNetV1
from InceptionV3 import InceptionV3
from Xception import Xception
from ResNeXt import ResNeXt29_2x64d
from ViT import ViT
from MobileNetV2 import MobileNetV2
from Darknet53 import Darknet53
from SqueezeNet import SqueezeNet
from ShuffleNet import ShuffleNet
from EfficientNet import EfficientNet
from ResMLP import ResMLP

from dataset import initialize_dataset
from train_test import training

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
                      MNIST=config['MNIST'])
train_dataloader, test_dataloader = data_initialization.load_dataset()

input_channel = next(iter(train_dataloader))[0].shape[1]
#n_classes = len(torch.unique(next(iter(train_dataloader))[1]))
n_classes = config['n_classes']

"""Model Initialization"""

if args.model == 'vggnet':
    model = VGG11(input_channel=input_channel, n_classes=n_classes,
            image_resolution=config['image_resolution']).to(device)

elif args.model == 'alexnet':
    model = AlexNet(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'senet':
    model = SENet(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'resnet':
    model = ResNet(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'densenet':
    model = DenseNet(input_channel=input_channel, n_classes=n_classes, 
            growthRate=12, depth=100, reduction=0.5, bottleneck=True).to(device)

elif args.model == 'nin':
    model = NIN(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'googlenet':
    model = GoogLeNet(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'cnn':
    model = CNN(input_channel=input_channel).to(device)

elif args.model == 'mobilenetv1':
    model = MobileNetV1(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'inceptionv3':
    model = InceptionV3(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'xception':
    model = Xception(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'resnext':
    model = ResNeXt29_2x64d(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'vit':
    model = ViT(image_size=config['image_resolution'], patch_size=32, dim=1024, depth=6, heads=16, 
            input_channel=input_channel, n_classes=n_classes,  mlp_dim=2048, dropout=0.1, emb_dropout=0.1).to(device)

elif args.model == 'mobilenetv2':
    model = MobileNetV2(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'darknet':
    model = Darknet53(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'squeezenet':
    model = SqueezeNet(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'shufflenet':
    cfg = {'out': [200,400,800], 'n_blocks': [4,8,4], 'groups': 2}
    model = ShuffleNet(cfg=cfg, input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'efficientnetb0':
    param = {
        # 'efficientnet type': (width_coef, depth_coef, resolution, dropout_rate)
        'efficientnetb0': (1.0, 1.0, 224, 0.2), 'efficientnetb1': (1.0, 1.1, 240, 0.2),
        'efficientnetb2': (1.1, 1.2, 260, 0.3), 'efficientnetb3': (1.2, 1.4, 300, 0.3),
        'efficientnetb4': (1.4, 1.8, 380, 0.4), 'efficientnetb5': (1.6, 2.2, 456, 0.4),
        'efficientnetb6': (1.8, 2.6, 528, 0.5), 'efficientnetb7': (2.0, 3.1, 600, 0.5)
    }
    model = EfficientNet(input_channels=input_channel, param=param[args.model], n_classes=n_classes).to(device)

elif args.model == 'mlpmixer':
    model = MLPMixer(image_size = config['image_resolution'], input_channels = input_channel,
    patch_size = 16, dim = 512, depth = 12, n_classes = n_classes, token_dim=128, channel_dim=1024).to(device)

elif args.model == 'resmlp':
    model = ResMLP(in_channels=input_channel, image_size=config['image_resolution'], patch_size=16, n_classes=n_classes,
                     dim=384, depth=12, mlp_dim=384*4).to(device)

print(model)
print(f'Total Number of Parameters of {args.model.capitalize()} is {round((sum(p.numel() for p in model.parameters()))/1000000, 2)}M')

trainer = training(model=model, optimizer=config['optimizer'], learning_rate=config['learning_rate'],train_dataloader=train_dataloader, 
          num_epochs=config['num_epochs'],test_dataloader=test_dataloader)
trainer.train()

# Calculate FLops and Memory Usage.
# model.to('cpu')
# dummy_input = (input_channel, config["image_resolution"], config["image_resolution"])
# print(stat(model, dummy_input))