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
from gMLP import gMLPForImageClassification
from EfficientNetV2 import EfficientNetV2

from dataset import initialize_dataset
from train_test import Training
from trainAndTestWithSAM import TrainingWithSAM

"""Device Selection"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" Initialize model based on command line argument """
model_parser = argparse.ArgumentParser(description='Image Classification Using PyTorch', usage='[option] model_name')
model_parser.add_argument('--model', type=str, required=True)
model_parser.add_argument('--model_save', type=bool, required=False)
model_parser.add_argument('--checkpoint', type=bool, required=False)
model_parser.add_argument('--sam', type=bool, required=False)
args = model_parser.parse_args()

"""Loading Config File"""
try:
    stream = open("config.yaml", 'r')
    config = yaml.safe_load(stream)
except FileNotFoundError:
    print("Config file missing")

"""Dataset Initialization"""
data_initialization = initialize_dataset(image_resolution=config['parameters']['image_resolution'], batch_size=config['parameters']['batch_size'], 
                      MNIST=config['parameters']['MNIST'])
train_dataloader, test_dataloader = data_initialization.load_dataset(transform=True)

input_channel = next(iter(train_dataloader))[0].shape[1]
#n_classes = len(torch.unique(next(iter(train_dataloader))[1]))
n_classes = config['parameters']['n_classes']

"""Model Initialization"""

if args.model == 'vggnet':
    model = VGG11(input_channel=input_channel, n_classes=n_classes,
            image_resolution=config['parameters']['image_resolution']).to(device)

elif args.model == 'alexnet':
    model = AlexNet(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'senet':
    model = SENet(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'resnet':
    model = ResNet(input_channel=input_channel, n_classes=n_classes).to(device)

elif args.model == 'densenet':
    model = DenseNet(input_channel=input_channel, n_classes=n_classes, 
            growthRate=12, depth=40, reduction=0.5, bottleneck=True).to(device)

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
    model = ViT(image_size=config['parameters']['image_resolution'], patch_size=32, dim=1024, depth=6, heads=16, 
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

elif args.model in ['efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 
                    'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7']:
    param = {
        # 'efficientnet type': (width_coef, depth_coef, resolution, dropout_rate)
        'efficientnetb0': (1.0, 1.0, 224, 0.2), 'efficientnetb1': (1.0, 1.1, 240, 0.2),
        'efficientnetb2': (1.1, 1.2, 260, 0.3), 'efficientnetb3': (1.2, 1.4, 300, 0.3),
        'efficientnetb4': (1.4, 1.8, 380, 0.4), 'efficientnetb5': (1.6, 2.2, 456, 0.4),
        'efficientnetb6': (1.8, 2.6, 528, 0.5), 'efficientnetb7': (2.0, 3.1, 600, 0.5)
    }
    model = EfficientNet(input_channels=input_channel, param=param[args.model], n_classes=n_classes).to(device)

elif args.model == 'mlpmixer':
    model = MLPMixer(image_size = config['parameters']['image_resolution'], input_channels = input_channel,
    patch_size = 16, dim = 512, depth = 12, n_classes = n_classes, token_dim=128, channel_dim=1024).to(device)

elif args.model == 'resmlp':
    model = ResMLP(in_channels=input_channel, image_size=config['parameters']['image_resolution'], patch_size=16, 
            n_classes=n_classes, dim=384, depth=12, mlp_dim=384*4).to(device)

elif args.model == 'gmlp':
    model = gMLPForImageClassification(in_channels=input_channel, n_classes=n_classes, 
                                        image_size=config['parameters']['image_resolution'], patch_size=16).to(device)

elif args.model in ['efficientnetv2']:
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],]

    model = EfficientNetV2(cfgs=cfgs, in_channel=input_channel, num_classes=n_classes).to(device)


#print(device)

print(f'Total Number of Parameters of {args.model.capitalize()} is {round((sum(p.numel() for p in model.parameters()))/1000000, 2)}M')
if not args.sam:
    trainer = Training(model=model, optimizer=config['parameters']['optimizer'], learning_rate=config['parameters']['learning_rate'], 
                train_dataloader=train_dataloader, num_epochs=config['parameters']['num_epochs'],test_dataloader=test_dataloader,
                model_name=args.model, model_save=args.model_save, checkpoint=args.checkpoint)
    trainer.runner()
else:
    trainer = TrainingWithSAM(model=model, optimizer=config['parameters']['optimizer'], learning_rate=config['parameters']['learning_rate'], 
                train_dataloader=train_dataloader, num_epochs=config['parameters']['num_epochs'],test_dataloader=test_dataloader,
                model_name=args.model, model_save=args.model_save, checkpoint=args.checkpoint)
    trainer.runner()
    
# Calculate FLops and Memory Usage.
# model.to('cpu')
# dummy_input = (input_channel, config['parameters']["image_resolution"], config['parameters']["image_resolution"])
# print(stat(model, dummy_input))
