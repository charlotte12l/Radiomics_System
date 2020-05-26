#!/usr/bin/python3
import sys
import os

import SimpleITK as sitk
import numpy as np
import scipy
import scipy.ndimage
import scipy.ndimage.measurements

# label 0 (background) is not included
labelbone = [1,3,5]
labelcartilage = [2,4,6]
minmaxratiocartilage = 0.3
minmaxratiobone = 0.8
minsize = 5

def eliminateNoise(label, labels, minmaxratio=0.5, minsize=0):
    for labelid in labels:
        labelin = label == labelid
        labelout = np.zeros_like(labelin, dtype=np.uint8)
        comp, num = scipy.ndimage.measurements.label(labelin)
        # present labelid not found
        if num <= 0:
            continue
        hist,_ = np.histogram(comp, bins=num+1, range=(-0.5, num+0.5))
        # final object label start with 1
        threshold = minmaxratio * hist[1:].max()
        valid_comp_id = []
        for compid, count in enumerate(hist):
            if compid == 0:
                continue
            if count >= threshold and count >= minsize:
                valid_comp_id.append(compid)
        for comp_id in valid_comp_id:
            labelout += (comp == comp_id)
        # same? 1, diff? 0
        labelsame = labelin == labelout
        # keep uneliminated elements
        label *= labelsame
    return label


if len(sys.argv) != 3:
    print('usage: '+sys.argv[0]+' labelin labelout')
    sys.exit(1)
labelpath = sys.argv[1]
labelpathout = sys.argv[2]

label = sitk.ReadImage(labelpath)
#array generated from GetArrayFromImage have reversal order from GetSize

labelout = sitk.GetArrayFromImage(label)
# for cartilage, eliminate noise slice by slice
for i in range(labelout.shape[0]):
    labelout[i,:,:] = eliminateNoise( \
            labelout[i,:,:].copy(), labelcartilage, \
            minmaxratiocartilage, minsize)
# for bone, eliminate noise slice by volume
labelout = eliminateNoise( \
        labelout.copy(), labelbone, minmaxratiobone, minsize)

labelout = sitk.GetImageFromArray(labelout)
labelout.SetSpacing(label.GetSpacing())
labelout.SetOrigin(label.GetOrigin())
labelout.SetDirection(label.GetDirection())

# fill holes ( not useful?)
for labelid in labelbone+labelcartilage:
    labelout = sitk.BinaryFillhole(labelout, foregroundValue=labelid)
sitk.WriteImage(labelout, labelpathout) 

