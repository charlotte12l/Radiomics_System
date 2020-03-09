#!/usr/bin/python3

import sys

import SimpleITK as sitk
import numpy as np
from classification import classificationGrade

if __name__ == '__main__':
    if len(sys.argv) > 1:
        image = sitk.ReadImage(sys.argv[1])
    else:
        image = sitk.ReadImage( \
                '/mnt/workspace/cartilage/PDslp1/hm/a103513699.nii.gz')
    # already histogram matched
    array = sitk.GetArrayFromImage(image)
    array = array[:,::-1,:]
    # Switch superor and inferior to simulate image read via simpleITK + Dicom
    # instead of dcm2niix.
    imageS2I = sitk.GetImageFromArray(array) # image superior to inferior
    imageS2I.CopyInformation(image)
    c = classificationGrade()
    print('evaluating...')
    print(list(prob.index(max(prob)) for prob in c.evaluate(imageS2I).tolist()))
    #print(c.evaluate(imageS2I))

