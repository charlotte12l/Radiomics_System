import torch
import torch.optim

optimizers = {}
optimizers['SGD'] = torch.optim.SGD
optimizers['Adam'] = torch.optim.Adam
