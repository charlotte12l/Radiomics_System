import SimpleITK as sitk
import six
import sys, os
import radiomics
# from ToolsFunc import select_file, write_csv, Normalize, fuse_mask
import numpy as np
import SimpleITK as sitk
import numpy as np
from python.ToolsFunc import *
from python.func_cuRadiomics import func_cuRadiomics
import time
import radiomics

def stack(dir):
    # if input is not stacked, please use this function
    subjects = os.listdir(dir)
    size  = np.squeeze(sitk.ReadImage(subjects[0])).shape
    if len(size)!=2:
        return False
    arr = np.zeros(len(subjects),size[0],size[1])
    for i in subjects:
        arr[i,:,:] = sitk.GetArrayFromImage(sitk.ReadImage(i))

    return arr

def func_pyRadiomics(yaml_addr, image, mask):
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(yaml_addr)
    # init = extractor.execute(img_normed[0], mask[0])
    all = []
    for i in range(200):
    # calculate features on prostate mask
        result_Of_prostate = extractor.execute(image[:,:,i], mask[:,:,i])
        # result_Of_prostate = extractor.execute(image[i, :, :], mask[i, :, :])
        all.append(result_Of_prostate)
    d = {}
    for k in result_Of_prostate.keys():
        d[k] = tuple(d[k] for d in all)
    return d
    # result_Of_prostate = extractor.execute(image, mask)
    # return result_Of_prostate

arr_img = np.random.randint(-256, high=256, size=(200, 240, 240), dtype='l')
arr_img[arr_img <= 0] = -1
arr_msk = np.zeros(arr_img.shape).astype('int')
arr_msk[:,80:180,80:180] = 1

py_yaml = './python/py_params.yaml'
yaml_addr = './python/params.yaml'
#
time_start = time.clock()

features = func_cuRadiomics(yaml_addr, arr_img, arr_msk)
time_end = time.clock()
gpu_t = time_end - time_start
print('GPU:' + str(gpu_t))

arr_img = sitk.GetImageFromArray(arr_img)
arr_msk = sitk.GetImageFromArray(arr_msk)

# arr_img = sitk.ReadImage('B2_CESAG.dcm.nii')
# arr_msk = sitk.ReadImage('B2_Label.nii')

# arr_img = Normalize(arr_img,255)
# arr_msk = fuse_mask(arr_msk)


time_start = time.clock()
features = func_pyRadiomics(py_yaml, arr_img, arr_msk)
time_end = time.clock()
cpu_t = time_end - time_start
print('CPU:' + str(cpu_t))
print('GPU:' + str(gpu_t))
print('ratio:',str(cpu_t/gpu_t))

