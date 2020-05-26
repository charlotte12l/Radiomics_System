import sys
assert sys.version_info >= (3,5), 'requires python3.5 or higher'

from .unet import UNet, UNets, UNet4, UNet5, UNet6, UNet7, UNet8, UNet9
from .vnet import VNet, VNet7
from .dilatedVnet import DilatedVNet

import torchvision
import torchvision.models.densenet
import torchvision.models.inception
import torchvision.models.densenet
import torchvision.models.resnet

from torch import nn

def inception_v3(classes_num=2):
    model = torchvision.models.inception.inception_v3( \
            pretrained=True, aux_logits=True)
    model.aux_logits = False
    del model.AuxLogits
    #for param in model.parameters():
    #    param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, classes_num)
    return model

def DenseNet121(classes_num=2):
    model = torchvision.models.densenet.densenet121(pretrained=True)
    # reset fc to fit different number of classes
    if classes_num != 1000:
        model.classifier = nn.Linear(\
                model.classifier.in_features, classes_num)
    return model

def ResNet18(classes_num=2):
    model = torchvision.models.resnet.resnet18(pretrained=True)
    # reset fc to fit different number of classes
    if classes_num != 1000:
        model.fc = nn.Linear(\
                model.fc.in_features, classes_num)
        model.avgpool = nn.AvgPool2d(14, stride=1)
    return model

def ResNet34(classes_num=2):
    model = torchvision.models.resnet.resnet34(pretrained=True)
    # reset fc to fit different number of classes
    if classes_num != 1000:
        model.fc = nn.Linear(\
                model.fc.in_features, classes_num)
        model.avgpool = nn.AvgPool2d(14, stride=1)
    return model

def ResNet50(classes_num=2):
    model = torchvision.models.resnet.resnet50(pretrained=True)
    # reset fc to fit different number of classes
    if classes_num != 1000:
        model.fc = nn.Linear(\
                model.fc.in_features, classes_num)
        model.avgpool = nn.AvgPool2d(14, stride=1)
    return model

def ResNet101(classes_num=2):
    model = torchvision.models.resnet.resnet101(pretrained=True)
    # reset fc to fit different number of classes
    if classes_num != 1000:
        model.fc = nn.Linear(\
                model.fc.in_features, classes_num)
        model.avgpool = nn.AvgPool2d(14, stride=1)
    return model

models={}
models['unet'] = UNet
models['unet4'] = UNet4
models['unet5'] = UNet5
models['unet6'] = UNet6
models['unet7'] = UNet7
models['unet8'] = UNet8
models['unet9'] = UNet9
models['unets'] = UNets
models['vnet'] = VNet
models['vnet7'] = VNet7
models['dilatedvnet'] = DilatedVNet
models['inception3'] = inception_v3
models['densenet121'] = DenseNet121
models['resnet18'] = ResNet18
models['resnet34'] = ResNet34
models['resnet50'] = ResNet50
models['resnet101'] = ResNet101
