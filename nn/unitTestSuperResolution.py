#!/usr/bin/python3

import sys

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from superResolution import superResolution

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
    c = superResolution()
    print('evaluating...')
    out = c(imageS2I)
    outArray = sitk.GetArrayFromImage(out)
    print(outArray.max(), outArray.min())
    n, bins, patches = plt.hist(outArray.ravel(), 100, range=[-1,1])
    #,density=False)
    #plt.axis([-1, 1, 0, 1])
    plt.ylabel('pixels (log)')
    plt.xlabel('residual')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    #print(c.evaluate(imageS2I))

