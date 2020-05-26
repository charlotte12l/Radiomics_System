import torch
from .DiceLoss import DiceLoss, Dice
from .Top1 import Top1

losses={}
losses['CrossEntropyLoss'] = torch.nn.CrossEntropyLoss
losses['DiceLoss'] = DiceLoss
losses['Dice'] = Dice
losses['Top1'] = Top1
