import SimpleITK as sitk
import six
import sys, os
import radiomics
import numpy as np
import csv


def startWith(*startstring):
    starts = startstring

    def run(s):
        f = map(s.startswith, starts)
        if True in f: return s

    return run


def endWith(*endstring):
    ends = endstring

    def run(s):
        f = map(s.endswith, ends)
        if True in f: return s

    return run



def write_csv(dict_1, file_1):
    with open(file_1, 'w+') as f:
        for key, value in list(dict_1.items())[12:]:
            f.write(str(key) + ',')
            f.write(str(value) + '\n')


def fuse_mask(image):
    array = sitk.GetArrayFromImage(image)
    array[array > 0] = 1
    after_image = sitk.GetImageFromArray(array)
    after_image.CopyInformation(image)
    return after_image


def select_file(file_address, selected):
    listed_file = os.listdir(file_address)
    selector = startWith(selected) #('_p2.nii')
    file_name = list(filter(selector, listed_file))
    if not len(file_name) == 0:
        file_selected = file_address + file_name[-1]
    else:
        file_selected = ''
    return file_selected


def Normalize(image, scale):

    image_arr = sitk.GetArrayFromImage(image).astype('float')

    min_value = np.percentile(image_arr, 0.1)
    max_value = np.percentile(image_arr, 99.9)
    image_arr[image_arr > max_value] = max_value
    image_arr[image_arr < min_value] = min_value   #-outliers

    new_image_arr = (image_arr-min_value)/(max_value-min_value)*scale
    new_image = sitk.GetImageFromArray(new_image_arr)
    new_image.CopyInformation(image)
    return new_image



def single_mask(image, label):
    array = sitk.GetArrayFromImage(image)
    array[array != label] = 0
    array[array == label] = 1
    after_image = sitk.GetImageFromArray(array)
    after_image.CopyInformation(image)
    return after_image



def fuse_mask(image):
    array = sitk.GetArrayFromImage(image)
    array[array > 0] = 1
    after_image = sitk.GetImageFromArray(array)
    after_image.CopyInformation(image)
    return after_image
