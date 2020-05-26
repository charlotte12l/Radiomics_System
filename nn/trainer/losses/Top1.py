#!env python3

import torch

class Top1(torch.nn.Module):
    def __init__(self):
        super(Top1, self).__init__()

    def forward(self, input, target):
        ifgpu = input.is_cuda
        input = input.cpu().detach()
        target = target.cpu().detach()
        correct = torch.ones(input.shape[0],2)
        _, predicted = torch.max(input, 1)
        for index in torch.arange(0, target.shape[0]).type(torch.LongTensor):
            correct[index, :] = torch.FloatTensor((target[index],)*2)
            if predicted[index] != target[index]:
                correct[index, 1] = -1 - correct[index, 1]
        if ifgpu:
            return correct.cuda()
        else:
            return correct

    def backward(self, input):
        raise NotImplementedError(
                "Top1 accuracy is not differentiable as it is not consistent")
