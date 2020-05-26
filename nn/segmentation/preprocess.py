#!/usr/bin/python3

import sys, os
import logging

import SimpleITK as sitk
import numpy as np




def _bias_correction(imgin):
    statistics = sitk.StatisticsImageFilter()
    statistics.Execute(imgin)
    mask = imgin > statistics.GetMinimum()
    #statistics.Execute(imgin)
    #if statistics.GetMean() : # log warning if foreground>0.8, for example
    #    logging.warning('')
    imgin = sitk.Cast(imgin, sitk.sitkFloat32)
    return sitk.N4BiasFieldCorrection(imgin, mask)

def _normalization(imgin):
    return sitk.Normalize(imgin)

def _histogram_matching(imgin, template):
    return sitk.HistogramMatching(imgin, template,
            numberOfHistogramLevels=1024, numberOfMatchPoints=7)

def _winsorizing(imgin, quantile):
    # reference: (ANTs soruce code)
    #   function PreprocessImage, Examples/itkantsRegistrationHelper.hxx
    # sitk.IntensityWindowing(imgin,
    raise NotImplementedError

def preprocesstemplate(templatein):
    """preprocess template before being used in preprocess
    BiasCorrection

    return: processed template, a simpleitk image
    """
    #return _bias_correction(templatein)
    return templatein

def preprocess(imgin, template = None, quantile = None):
    """Prepare images.
    BiasCorrection->HistogramMatching->Normalization
    
    return: prepared data. A simpleitk image of sitkFloat32 type.

    imgin: input image, a simpleitk image
    template: simpleitk template image for histogram matching.
    quantile: quantile for winsorizing, for example [0.005, 0.995]
        If not specified, winsorizing will be skipped.
    Note: you'd better do bias correction on template image first.
    """
    #if quantile != None
    #    imgin = _winsorizing(imgin, qunatile)
    #imgin = _bias_correction(imgin)
    if template != None:
        imgin = _histogram_matching(imgin, template)
    #imgin = _normalization(imgin)
    return imgin


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if len(sys.argv) != 4:
        logging.error('usage: '+sys.argv[0]+' input output template')
        sys.exit(1)
    inpath = sys.argv[1]
    outpath = sys.argv[2]
    templateimagepath = sys.argv[3]
    logging.info("...preprocessing template image: " + templateimagepath)
    templateimage = sitk.ReadImage(templateimagepath)
    templateimage = sitk.Cast(templateimage, sitk.sitkFloat32)
    templateimage = preprocesstemplate(templateimage)
    logging.info("...preprocessing image: "+inpath)
    image = sitk.ReadImage(inpath)
    image = sitk.Cast(image, sitk.sitkFloat32)
    image = preprocess(image, templateimage)
    sitk.WriteImage(image, outpath)
    logging.info("...writing preprocessed image: " + outpath)

