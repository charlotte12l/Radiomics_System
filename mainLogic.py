#!/usr/bin/python3

import os,sys
from ReadSagittalPD import ReadSagittalPDs
from simpleThickness import getCartilageThickness
from analysis.FeatureExtraction import  FeatureExt
from analysis.FeatureSelection import  FeatureSel

import SimpleITK as sitk
import numpy as np

class mainLogic(object):
    def __init__(self):
        super(mainLogic, self).__init__()
        #self.gradeNN = classificationGrade()
        #self.segmentationJointNN = segmentationJoint()
        #self.superResolutionNN = superResolution()
        self.featureExt = FeatureExt()
        self.featureSel = FeatureSel()
        self.__image = None
        self.__ROI = None
        self.__feature_extracted = None
        # self.__seg = None
        # self.__grade = None
        # self.__superResolution = None
        # self.__thickness = None
    
    def getImage(self):
        return self.__image

    def setImage(self, image):
        if image is None:
            return False
        self.__image = image
        self.__feature_extracted = None
        # self.__seg = None
        # self.__grade = None
        # self.__superResolution = None
        # self.__thickness = None
        return True

    def setROI(self, ROI):
        if ROI is None:
            return False
        self.__ROI = ROI
        self.__feature_extracted = None
        # self.__grade = None
        # self.__superResolution = None
        # self.__thickness = None
        return True

    def getFeature(self):
        assert self.__image is not None, 'No image loaded'
        assert self.__ROI is not None, 'No ROI loaded'
        if self.__feature_extracted is not None:
            return self.__feature_extracted
        feature_extracted = self.featureExt(self.__image,self.__ROI)
        self.__feature_extracted = feature_extracted
        return feature_extracted

    def getSeg(self):
        assert self.__image is not None, 'No image loaded'
        if self.__seg is not None:
            return self.__seg
        seg = self.segmentationJointNN(self.__image)
        self.__seg = seg
        return self.__seg

    def getGrade(self):
        assert self.__image is not None, 'No image loaded'
        if self.__grade is not None:
            return self.__grade
        grade = self.gradeNN(self.__image)
        self.__grade = grade
        return grade

    def getSuperResolution(self):
        assert self.__image is not None, 'No image loaded'
        if self.__superResolution is not None: return self.__superResolution
        superResolution = self.superResolutionNN(self.__image)
        self.__superResolution = superResolution
        #self.__superResolution = self.__image
        return self.__superResolution

    def getThickness(self):
        assert self.__image is not None, 'No image loaded'
        assert self.__seg is not None, 'Run segmentation first'
        if self.__thickness is not None: return self.__thickness
        f = getCartilageThickness(self.__seg, cartilage=2, bone=1) - 0.2
        t = getCartilageThickness(self.__seg, cartilage=4, bone=3) - 0.2
        p = getCartilageThickness(self.__seg, cartilage=6, bone=5) - 0.6
        self.__thickness = (f,t,p)
        return self.__thickness

if __name__ == '__main__':
    import time
    import torch
    images = ReadSagittalPDs('/mnt/workspace/repo/privateData/cartilage_origin/CartilageData_FenJin/SixthPeopleHospitalPD_OriDicom/A103327625')
    logic = mainLogic()
    logic.setImage(images[0])
    logic.getGrade()
    logic.getSeg()
    logic.getThickness()
    logic.getSuperResolution()
