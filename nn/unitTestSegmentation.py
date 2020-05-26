#!/usr/bin/python3

import sys
sys.path.append('/home/liuxy/Radiomics_System')
import SimpleITK as sitk
import numpy as np
from seg import segmentationJoint


if __name__ == '__main__':
    if len(sys.argv) > 1:
        image = sitk.ReadImage(sys.argv[1])
    else:
        image = sitk.ReadImage( \
                '/home/liuxy/Radiomics_System/nn/segmentation/test.nii')
    # already histogram matched
    array = sitk.GetArrayFromImage(image)
    array = array[:,::-1,:]
    # Switch superor and inferior to simulate image read via simpleITK + Dicom
    # instead of dcm2niix.
    imageS2I = sitk.GetImageFromArray(array) # image superior to inferior
    imageS2I.CopyInformation(image)
    c = segmentationJoint()
    out = c.evaluate(imageS2I)
    sitk.WriteImage(out, 'test_out.nii')
