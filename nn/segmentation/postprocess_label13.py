#!/usr/bin/python3

#combine all labels, and find connected components(A1,A2,A3...)
#for each label(L1,L2,L3...), find the valuable components(B1,B2,B3...) of it
#for each valuable component(Bi),
#   find the corresponding corresponding component(Aj), and mark it as L
#Bug: if two valuable component(Bi1,Bi2) share
#   the same corresponding component(Aj),
#   component(Aj) will be labels as both Li1 and Li2

# solution: find valid comps(A1,A2,A3...) and use pp_label13_alt.py to label Ai

import sys
import os

import SimpleITK as sitk
import numpy as np

# label 0 (background) is not included
labels = [1,3]
minmaxratio = 0.9


if len(sys.argv) != 3:
    print('usage: '+sys.argv[0]+' labelin labelout')
    sys.exit(1)
labelpath = sys.argv[1]
labelpathout = sys.argv[2]

label = sitk.ReadImage(labelpath)
# ignore not included labels
newlabel = label*0
for label_id in labels:
    newlabel += label_id*(label_id == label)
label = newlabel
del newlabel

#array generated from GetArrayFromImage have reversal order from GetSize
comps = sitk.ConnectedComponent(label)
valid_comps_id = set()
for labelid in labels:
    comp_sitk = sitk.ConnectedComponent(label == labelid)
    comp = sitk.GetArrayFromImage(comp_sitk)
    hist,_ = np.histogram(\
            comp, bins=comp.max()+1, range=(-0.5, comp.max()+0.5))
    # present labelid not found
    if len(hist) <= 1:
        continue
    # object label start with 1, 0 stands for background
    threshold = minmaxratio * hist[1:].max() # unit: number of pixels
    valid_labelcomp_id = []
    for compid, count in enumerate(hist):
        if compid == 0:
            # background, ingnored
            continue
        if count >= threshold:
            valid_labelcomp_id.append(compid)
    for comp_id in valid_labelcomp_id:
        mask = comp_sitk == comp_id
        onecomp = sitk.Mask(comps, mask)
        minmax = sitk.MinimumMaximumImageFilter()
        minmax.Execute(onecomp)
        comps_id = minmax.GetMaximum()
        valid_comps_id.add(comps_id)

labelout = label*0
comp_sitk = comps
comp = sitk.GetArrayFromImage(comp_sitk)
hist,_ = np.histogram(\
        comp, bins=comp.max()+1, range=(-0.5, comp.max()+0.5))
threshold = minmaxratio * hist[1:].max() # unit: number of pixels
for comp_id in valid_comps_id:
    mask = comp_sitk == comp_id
    onecomp = sitk.Mask(label, mask)
    comp_hist,_ = np.histogram( \
        sitk.GetArrayFromImage(onecomp), \
        bins=max(labels)+1, range=(-0.5, max(labels)+0.5))
    comp_hist_nobg = comp_hist[1:] # hist without background
    onelabel = comp_hist_nobg.argmax()+1
    labelout += int(onelabel)*mask


# fill holes ( not useful?)
for labelid in labels:
    labelout = sitk.BinaryFillhole(labelout, foregroundValue=labelid)
sitk.WriteImage(labelout, labelpathout) 

