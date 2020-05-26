#!/usr/bin/python3

import os.path
import random
from functools import reduce

import numpy as np
#import cv2
#cv2.setNumThreads(0)
import torch
import torch.utils
import torch.utils.data
import SimpleITK as sitk
#sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(0)
import scipy
import scipy.ndimage


class SlicesOfSubject(torch.utils.data.Dataset):
    '''Dataset of slices of a subject
    You can concatenate datasets to a torch.ConcatDataset afterwards.
    Available slices are include_slices 
    Slice indice start from 0.
    Function preprocess should be thread-safe as there are multiple workers.
    '''
    suitableJobs = ['seg', 'cla']
    def __init__(self, image, label, classes, job, \
            spacing=None, crop=None, ratio=None, rotate=None, \
            include_slices=None):
        assert job in self.suitableJobs, 'not suitable jobs'
        self.job = job
        if job == 'seg':
            assert classes[0] == 0
        if job == 'cla':
            assert len(classes) > 1
        assert len(image.GetSize()) == 3
        self.classes = classes
        self.size = image.GetSize()
        self.origin = image.GetOrigin()
        self.direction = image.GetDirection()
        self.spacing = image.GetSpacing()
        # slices
        if include_slices is None:
            self.slice_indice = list(range(image.GetSize()[-1]))
        else:
            assert len(include_slices) > 0
            self.slice_indice = include_slices
        # extract slices
        array = sitk.GetArrayFromImage(image)
        self.slices = []
        for index in self.slice_indice:
            self.slices.append((array[index,:,:]).copy())
        self.labels = []
        # if eval?
        self.label=None
        if label is None:
            self.labels = None
            self.if_eval = True
            self.label_array = None
        else:
            self.if_eval = False
            if job == 'seg':
                assert image.GetSize() == label.GetSize()
                array = sitk.GetArrayFromImage(label)
                for index in self.slice_indice:
                    self.labels.append((array[index,:,:]).copy())
            if job == 'cla':
                assert label >= 0
                self.labels = len(self.slice_indice)*[label]
        # data augmentation
        if ratio is None or ratio[0] is None:
            ratio = np.array((0,0), dtype=np.float32)
        assert (ratio[0] < 0.5 ) and (ratio[0] >=0) and \
                (ratio[1] < 0.5 ) and (ratio[1] >=0), \
                "border_ratio range [0,0.5)"
        if rotate is None:
            rotate = 0
        if spacing is None or spacing[0] is None:
            scale = np.array((1,1), dtype=np.float32)
        else:
            scale = np.array(image.GetSpacing()[1::-1], dtype=np.float32) /\
                    spacing
        if crop is None or crop[0] is None:
            crop = np.array(image.GetSize()[1::-1], dtype=np.int64)
        self.ratio = ratio
        self.scale = scale
        self.crop = crop
        self.rotate = rotate

    def __len__(self):
        return len(self.slice_indice)

    def __getitem__(self, index):
        # image
        image = self.slices[index].copy()
        image = image.astype(np.float32)
        # data augmentation
        crop = self.crop
        if_eval = self.if_eval
        size = np.array(image.shape)
        scale = self.scale
        translate = crop*self.ratio
        rotate = self.rotate
        if if_eval:
            translate = np.array([0,0]);
            rotate = 0;
        M = calM_2D(sizein = size, \
                sizeout = crop, \
                scale = scale, \
                translate = np.array((translate[0]*2*(np.random.random()-0.5),\
                    translate[1]*2*(np.random.random()-0.5))),\
                rotate = np.array((rotate*(np.random.random()-0.5),)))
        # there's a slight difference between opencv and scipy, don't know why
        #image1 = cv2.warpAffine(image, M[0:2,:], tuple(crop.tolist()[::-1]), \
        #        flags=cv2.INTER_CUBIC)
        image = scipy.ndimage.affine_transform(image, np.linalg.inv(M), \
                output_shape=crop, output=np.float32, \
                order=3, mode='constant', cval=0)
        eps = 10e-5
        image = (image - image.mean())/(image.std()+eps)
        # one channel image
        image = image.reshape((1,) + image.shape)
        image = torch.from_numpy(image.astype(np.float32))
        if self.job == 'cla':
            # three channel image if classification
            image = torch.cat((image,image,image), dim=0)
        # label
        if self.if_eval:
            if self.job == 'seg':
                label = image #pytorch cannot make None as batch
            else:
                label = -2
        else:
            if self.job == 'seg':
            # change label
            # background is always zero
                label = self.labels[index]
                label = label.copy().astype(np.uint8)
                # new class id is index of classes
                #label = cv2.warpAffine(label, invM, \
                #        tuple(crop.tolist()[::-1]), \
                #        flags=cv2.INTER_NEAREST)
                label = scipy.ndimage.affine_transform(label, \
                        np.linalg.inv(M), output_shape=crop, output=np.uint8, \
                        order=0, mode='constant', cval=0)
                label_new = np.zeros(label.shape, dtype = np.int64)
                for index, oneclass in enumerate(self.classes):
                    label_new = np.where(label == oneclass, index, label_new)
                label_new.reshape((1,)+label_new.shape)
                label = torch.from_numpy(label_new)
            if self.job == 'cla':
                label = self.labels[index]
                label = self.classes.index(label)
                #label = torch.LongTensor([label])
        return (image, label)



    def setPrediction(self, indice, predictions):
        '''set slice predict
        indice (mini-batch tensor) are slice available outside, \
                range [0, len(Datatset) -1]
        predict is mini-batch tensor
        '''
        assert self.if_eval, "setting prediction is no allowed in non-eval job"
        indice = list(map(lambda x, self=self: self.slice_indice[x], indice))
        if self.job == 'cla':
            if self.label is None:
                self.label = np.zeros( \
                        self.size[2], dtype=np.int64)
            for i, index in enumerate(indice):
                prediction = predictions[i]
                self.label[index] = self.classes[prediction]
        if self.job == 'seg':
            if self.label is None:
                self.label = np.zeros( \
                        self.size[::-1], dtype=np.int8)
            sizein = np.array(self.size[1::-1])
            sizeout = self.crop
            scale = self.scale
            M = calM_2D(sizein, sizeout=sizeout, scale=scale)
            #M = np.linalg.inv(M)
            lut = np.array(self.classes, dtype=np.int64)
            for i, index in enumerate(indice):
                prediction = predictions[i,:,:].numpy()
                prediction = prediction.astype(np.int64)
                #prediction = cv2.warpAffine(prediction, M, \
                #        tuple(sizein.tolist()[::-1]), flags=cv2.INTER_NEAREST)
                prediction = scipy.ndimage.affine_transform(prediction, \
                        M, output_shape=sizein, \
                        order=0, mode='constant', cval=0)
                prediction = lut[prediction]
                ####!!!!!!!!!!maybe self.label[index,:,:] = prediction??
                self.label[self.slice_indice[index],:,:] = prediction

    def getPrediction(self):
        '''get 3D predict
        '''
        assert self.if_eval, "getting prediction is no allowed in non-eval job"
        if self.job == 'cla':
            return self.label
        if self.job == 'seg':
            label = sitk.GetImageFromArray(self.label)
            label.SetOrigin(self.origin)
            label.SetDirection(self.direction)
            label.SetSpacing(self.spacing)
            return label


def calM_2D(sizein, sizeout=None, scale=None, translate=None, rotate=None):
    """center to origin -> rotate -> scale -> translation -> center back
    unit of sizein/out is pixel, (y, x) vector, i.e. (cols, rows)
    unit of scale is none (ratio), (y, x) vector
    unit of translate is pixel, (y, x) vector
    unit of rotation is rad, scalar
    """
    # ATTENTION !
    # for opencv , array.shape = (y/row_length,x/col_length)
    # check in/out image size
    assert sizein.shape == (2,), "sizein should be (2,) array"
    if sizeout is None:
        sizeout = sizein.copy()
    else:
        assert sizeout.shape == (2,), "sizein should be (2,) array"
    # identity transform
    M_I = np.array(( \
            (1,0,0), \
            (0,1,0), \
            (0,0,1)), dtype=np.float64)
    # move image center to origin
    M_m1 = np.array(( \
            (1,0,-float(sizein[1])/2), \
            (0,1,-float(sizein[0])/2), \
            (0,0,1)), dtype=np.float64)
    # scale
    if scale is None:
        M_s = M_I.copy()
    else:
        assert scale.shape == (2,), "scale should be (2,) array"
        M_s = np.array(( \
                (scale[1],0,0), \
                (0,scale[0],0), \
                (0,0,1)), dtype=np.float64)
    # rotate
    if rotate is None:
        M_r = M_I.copy()
    else:
        assert rotate.shape == (1,), "rotate shoud be (1,) array"
        M_r = np.array(( \
                (np.cos(rotate), np.sin(rotate),0),\
                (-np.sin(rotate), np.cos(rotate), 0),\
                (0, 0, 1)), dtype=np.float64)
    # translate
    if translate is None:
        M_t = M_I.copy()
    else:
        assert translate.shape == (2,), "translate shoud be (2,) array"
        M_t = np.array(( \
                (1,0,translate[1]),
                (0,1,translate[0]),
                (0,0,1)), dtype=np.float64)
    # move image center to output space center
    M_m2 = np.array(( \
            (1,0,float(sizeout[1])/2), \
            (0,1,float(sizeout[0])/2), \
            (0,0,1)), dtype=np.float64)
    M = reduce(np.dot, [M_m2, M_t, M_r, M_s, M_m1])
    return M[0:2,:]




if __name__ == '__main__':
    from utils_view import view_tensor_data, view_tensor_image
    row = ['2', \
            '/home/kylexuan/workspace/cartilage/ski10/hm/image-005.nii.gz', \
            '/home/kylexuan/workspace/cartilage/ski10/label/labels-005.nii.gz',\
            None]
    classes = [0,1,2,3,4]
    subject = SlicesOfSubject( \
            sitk.ReadImage(row[1]), sitk.ReadImage(row[2]),\
            classes = classes, job = 'seg', \
            spacing=np.array((0.333333333, 0.33333333)), \
            crop=np.array((512, 512)), \
            include_slices=row[3])
    #for i in range(len(subject)):
    #   view_tensor_data(subject[i], [0,1,2,3,4])

    subjectT = SlicesOfSubject( \
            sitk.ReadImage(row[1]), None, \
            classes = classes, job = 'seg', \
            spacing=np.array((0.333333333, 0.33333333)), \
            crop=np.array((512, 512)), \
            ratio=np.array((0.025, 0.025)), \
            rotate=3.14159/180*5, \
            include_slices=row[3])

    indice = list(map(lambda x: x*2, list(range(50))))
    subjectT.setPrediction(indice, torch.stack( \
            list(map(lambda x: subject[x][1], indice)), dim=0))

    subject = SlicesOfSubject( \
            sitk.ReadImage(row[1]), subjectT.getPrediction(),\
            classes = classes, job = 'seg', \
            spacing=np.array((0.333333333, 0.33333333)), \
            crop=np.array((512, 512)), \
            ratio=np.array((0.028, 0.028)), \
            rotate=3.14159/180*5, \
            include_slices=None)

    datasets = [\
            SlicesOfSubject( \
            sitk.ReadImage('/home/kylexuan/workspace/cartilage/ski10/hm/image-{:03}.nii.gz'.format(i+1)), \
            sitk.ReadImage('/home/kylexuan/workspace/cartilage/ski10/label/labels-{:03}.nii.gz'.format(i+1)), \
            classes = [0,1,2,3,4,5,6], job = 'seg', \
            spacing=np.array((1/3, 1/3)), \
            crop=np.array((512, 512)), \
            ratio=np.array((5.0/180, 5.0/180)), \
            rotate=5*3.14159265359/180, \
            include_slices=None) \
            for i in range(100)]

    dataset = torch.utils.data.dataset.ConcatDataset(datasets)
    training_loader = torch.utils.data.DataLoader( \
            dataset, batch_size=8, shuffle=False, \
            num_workers=8, pin_memory=False, drop_last=True)


    for i, (image, label) in enumerate(training_loader):
        #view_tensor_data(subject[i], [0,1,2,4], str(i))
        #img, label = subject[i]
        print(i)
