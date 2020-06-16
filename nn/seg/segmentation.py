#!/usr/bin/python3

import os.path
import json
import SimpleITK as sitk
import numpy as np
import torch
import cv2



class segmentation(object):
    def __init__(self, ConfigFilePath):
        super(segmentation, self).__init__()
        configPath = os.path.join(\
                os.path.dirname(__file__), ConfigFilePath)
        # please load config, model and saved parameters
        with open(configPath, 'r') as f:
            config = json.load(f)
        self.spacing = config['voxelSpacing']
        self.crop = config['crop']
        # load saved parameters
        self.net = YourNet

    def __call__(self, *args, **kw):
        return self.evaluate(*args, **kw)

    def evaluate(self, image):
        self.net.model.eval()
        image = preProcess(image, self.hmTemplate)
        inputs, M = image2batch(image, crop=self.crop, spacing=self.spacing)
        # or slice by slice
        array = torch.LongTensor()
        for i in range(inputs.shape[0]):
            output = self.net.model(inputs[i:i+1,:,:,:].cuda())
            prob = torch.nn.functional.softmax(output, dim=1)
            _, prediction = torch.max(prob, 1)
            array = torch.cat([array, prediction.cpu().detach()])
        array = array.numpy()
        # end of slice by slice
        out = np.zeros(image.GetSize()[::-1], dtype=np.uint8)
        invM = cv2.invertAffineTransform(M)
        for i in range(out.shape[0]):
            prediction = array[i,:,:]
            out[i,:,:] = cv2.warpAffine(prediction, invM, \
                    tuple(image.GetSize()[0:2]), flags=cv2.INTER_NEAREST)
        out = out[:,::-1,:]
        out = sitk.GetImageFromArray(out)
        out.CopyInformation(image)
        return out

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
    batch = torch.FloatTensor(image.GetSize()[2], 1, *(crop.tolist())) # 3 chs
    for i in range(image.GetSize()[2]):
        img = cv2.warpAffine(array[i], M, tuple(crop.tolist()[::-1]), \
                flags=cv2.INTER_CUBIC)
        if img.std() != 0:
            img = (img - img.mean())/img.std()
        else:
            img = (img - img.mean())
        img = torch.from_numpy(img.astype(np.float32))
        batch[i,0,:,:] = img
    return (batch, M)


class segmentationJoint(segmentation):
    def __init__(self):
        super(segmentationJoint, self).__init__( \
                ConfigFilePath='./configJoint.json')

