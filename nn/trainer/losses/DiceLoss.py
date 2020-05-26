#!/usr/bin/python3
import os
import sys

import numpy as np

import torch
import torch.nn
import torch.nn.modules


class DiceLoss(torch.nn.Module):
    """1 - dice as loss
    averaged over batch and class
    """
    def __init__(self, eps = 1e-5):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, prediction, target):
        assert target.requires_grad == False
        prediction = prediction.view(*prediction.shape[0:2],-1)
        target = target.view(target.shape[0], 1 ,-1)
        target_onehot = torch.zeros_like(prediction, dtype=prediction.dtype)
        target_onehot.scatter_(1, target, 1)

        eps = self.eps
        dice = (2* torch.sum(prediction*target_onehot, dim=2) + eps) /\
                (torch.sum(prediction ** 2, dim=2) + \
                torch.sum(target_onehot ** 2, dim=2) + eps)
        return 1 - dice.mean()

class Dice(torch.nn.Module):
    """calculate dice
    """
    def __init__(self, eps = 1e-5, ifaverage = True):
        super(Dice, self).__init__()
        self.eps = eps
        self.ifaverage = ifaverage

    def forward(self, input, target):
        ifgpu = input.is_cuda
        input = input.cpu().detach()
        target = target.cpu().detach()
        ishape = input.size()
        tshape = target.size()
        input = input.view(*ishape[0:2],-1)
        target = target.view(tshape[0],-1)
        _,input = torch.max(input, dim=-2)
        if ifgpu:
            dice = torch.cuda.FloatTensor(tshape[0], ishape[1]-1)
        else:
            dice = torch.FloatTensor(tshape[0], ishape[1]-1)
        eps = self.eps
        for index in torch.arange(1, ishape[1]):
            index = int(index)
            targeti = target == index
            inputi = input == index
            for batch_id in torch.arange(0, ishape[0]):
                batch_id = int(batch_id)
                # output of torch.sum(tensor) is float
                intersection = \
                        torch.sum(targeti[batch_id] * inputi[batch_id])
                union = \
                        torch.sum(targeti[batch_id]) +\
                        torch.sum(inputi[batch_id])
                intersection = intersection.item()
                union = union.item()
                dice[batch_id,index-1] =\
                        (2. * intersection + self.eps)/\
                        (union + self.eps)
        if self.ifaverage:
            diceout = torch.mean(dice, dim=0)
            diceout = torch.mean(diceout, dim=0, keepdim=True)
        else:
            diceout = dice

        return diceout

    def backward(self, input):
        raise NotImplementedError(
                "Dice overlap is not differentiable as it is not consistent")

