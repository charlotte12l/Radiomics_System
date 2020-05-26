import torch
import torch.utils
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    def __init__(transform=None):
        super(Dataset, self).__init__()
        raise NotImplementedError

    def __call__(self, index):
        raise NotImplementedError
        return transform(image)

class NpDataset2D(Dataset):
    pass

class NpDataset2DSeq(Dataset):
    pass
