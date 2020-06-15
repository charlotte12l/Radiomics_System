import SimpleITK as sitk
import six
import sys, os
import radiomics
from analysis.ToolsFunc import select_file, write_csv, Normalize, fuse_mask
import numpy as np

Param = './Params.yaml'

# def FeatureExtraction(Param, addr, selector, sav_folder, mask_value):
class FeatureExt(object):
    def __init__(self):
        super(FeatureExt, self).__init__()

    def __call__(self, *args, **kw):
        return self.evaluate(*args, **kw)

    def evaluate(self, img, mask_init):

        img_normed = Normalize(img, 255)
        mask = fuse_mask(mask_init)

        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(Param)

        # calculate features on prostate mask

        result_Of_prostate = extractor.execute(img_normed, mask)
        #filename_of_prostate = 'tmp.csv'
        #write_csv(result_Of_prostate, filename_of_prostate)
        return result_Of_prostate

class cuFeatureExt(object):
    def __init__(self):
        super(cuFeatureExt, self).__init__()

    def __call__(self, *args, **kw):
        return self.evaluate(*args, **kw)

    def evaluate(self, img, mask_init):

        img_normed = Normalize(img, 255)
        mask = fuse_mask(mask_init)

        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(Param)

        # calculate features on prostate mask

        result_Of_prostate = extractor.execute(img_normed, mask)
        #filename_of_prostate = 'tmp.csv'
        #write_csv(result_Of_prostate, filename_of_prostate)
        return result_Of_prostate


if __name__ == '__main__':
    #import pandas

    img = sitk.ReadImage('..\B2_CESAG.dcm.nii')
    # img_array = sitk.GetArrayFromImage(img)

    label = sitk.ReadImage('..\B2_Label.nii')
    # label_array = sitk.GetArrayFromImage(label)
    ext = FeatureExt()
    arry = ext(img,label)

    #d3 = {k: v for k, v in arry.items() if v.dtype }


    #write_csv(arry, 'tmp.csv')

    # list = sorted(arry.items())
    # name,value = zip(*list) # unpack a list of pairs into two tuples
    #
    # #print(name)
    # print(len(name))
    # print(name[50])
    # #print(value)
    # print(len(value))
    # print(value[50])


    #print(np.shape(arry))
    #print(arry)
