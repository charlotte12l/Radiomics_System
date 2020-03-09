#!/usr/bin/python3

ConfigFilePath='./config.json'

import os.path
import json
import SimpleITK as sitk
import numpy as np
import torch
import cv2


from trainer.trainer import Trainer
#from .trainer import Trainer
from segmentation.utils.utils_data import calM_2D


class classificationGrade(object):
    def __init__(self):
        super(classificationGrade, self).__init__()
        configPath = os.path.join(\
                os.path.dirname(__file__), ConfigFilePath)
        # load config
        with open(configPath, 'r') as f:
            config = json.load(f)
        self.spacing = config['voxelSpacing']
        self.crop = config['crop']
        # load saved parameters
        self.net = Trainer(os.path.join(\
                os.path.dirname(configPath), config['trainedModel']))
        self.hmTemplate = sitk.ReadImage(os.path.join(\
                os.path.dirname(configPath), config['hmTemplate']))

    def __call__(self, *args, **kw):
        return self.evaluate(*args, **kw)

    def evaluate(self, image):
        self.net.model.eval()
        image = preProcess(image, self.hmTemplate)
        inputs = image2batch(image, crop=self.crop, spacing=self.spacing)
        inputs = inputs.cuda()
        outputs = self.net.model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)
        #return (probs.cpu().data.numpy(), predictions.cpu().data.numpy())
        #return probs.cpu().data.numpy().tolist()
        #return predictions.cpu().data.numpy().tolist()
        return probs.cpu().data.numpy()

def preProcess(image, template):
    template = sitk.Cast(template, sitk.sitkFloat32)
    image = sitk.Cast(image, sitk.sitkFloat32)
    imageOut = sitk.HistogramMatching(image, template, \
            numberOfHistogramLevels=1024, numberOfMatchPoints=7)
    return imageOut


def image2batch(image, crop=None, spacing=None):
    if spacing is None:
        scale = np.array((1,1), dtype=np.float32)
    else:
        scale = np.array(image.GetSpacing()[1::-1], dtype=np.float32)/ \
                np.array(spacing, dtype=np.float32)
    if crop is None:
        crop = np.array(image.GetSize()[1::-1], dtype=np.int64)
    else:
        crop = np.array(crop, dtype=np.int64)
    array = sitk.GetArrayFromImage(image)
    # Nifty image converted via dcm2niix has array arranged as [P,S,-L]
    # and image read from dicom via simpleitk has arrary arrange as [P,-S,-L].
    # What's more, image[P,-S,-L] = array[-L,-S,P].
    # We use nii images converted from dicom via dcm2niix to train the model,
    # so y axis have to be flipped.
    array = array[:,::-1,:] # flip (superior <-> inferior)
    # preprocess and convert to mini-batch
    array = array.astype(np.float64)
    M = calM_2D( \
            sizein = np.array(image.GetSize()[1::-1], dtype=np.int64), \
            sizeout = crop, scale = scale, \
            translate = None, rotate = None)
    batch = torch.FloatTensor(image.GetSize()[2], 3, *(crop.tolist())) # 3 chs
    for i in range(image.GetSize()[2]):
        img = cv2.warpAffine(array[i], M, tuple(crop.tolist()[::-1]), \
                flags=cv2.INTER_CUBIC)
        if img.std() != 0:
            img = (img - img.mean())/img.std()
        else:
            img = (img - img.mean())
        img = torch.from_numpy(img.astype(np.float32))
        batch[i,0,:,:] = img
        batch[i,1,:,:] = img
        batch[i,2,:,:] = img
    return batch

