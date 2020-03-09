#!/usr/bin/python3

import os, sys
from functools import reduce
import operator
import SimpleITK as sitk
import numpy as np

dicomTags = { \
        'AccessionNumber': '0008|0050', \
        'SeriesDescription': '0008|103e', \
        'PatientName' : '0010|0010', \
        'PatientID' : '0010|0020', \
        'PatientBirthData': '0010|0030', \
        'PatientSex': '0010|0040', \
        'PatientAge': '0010|1010', \
        'PatientWeight' : '0010|1030', \
        'SeriesDate' : '0008|0021', \
        'SeriesTime' : '0008|0031', \
        'Modality' : '0008|0060', \
        'Manufacturer' : '0008|0070', \
        'SeriesInstanceUID' : '0020|000e', \
        'Laterality' : '0020|0060', \
        'SliceOrientation' : '2001|100b'}

'''
I gave up the alternative way as sitk.JoinSeries throw errors:
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/kylexuan/.local/lib/python3.6/site-packages/SimpleITK/SimpleITK.py", line 38851, in JoinSeries
    return _SimpleITK.JoinSeries(*args)
RuntimeError: Exception thrown in SimpleITK JoinSeries: /tmp/SimpleITK-build/ITK-prefix/include/ITK-4.11/itkImageToImageFilter.hxx:250:
itk::ERROR: JoinSeriesImageFilter(0x405fac0): Inputs do not occupy the same physical space!
InputImage Origin: [1.2256139e+02, -1.3847505e+02], InputImage_1 Origin: [1.1858115e+02, -1.3808055e+02]
        Tolerance: 3.1250000e-07

Now the metadata is read from a single slices and the copied to 3D image
'''
def ReadDICOMSeriesRecursively(dicom):
    '''Read dicom series recursively
    return -> a list of sitk.Image, metadata included
    '''
    allDirs = [dirs for dirs, _, _ in os.walk(dicom)]
    dicomSeries = reduce(operator.add, map(ReadDICOMSeries, allDirs))
    return dicomSeries

def ReadDICOMSeries(dicomPath):
    '''Read dicom series
    return -> a list of sitk.Image, metadata included
    '''
    dicomSeriesList = []
    seriesIDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicomPath)
    for seriesID in seriesIDs:
        seriesDicomFiles = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(\
                dicomPath, seriesID) #recursive=True does not work??
        # => sitk.ImageSeriesReader.MetaDataDictionaryArrayUpdateOn
        # will be implemented in SimpleITK > 1.0.1
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(seriesDicomFiles)
        #reader.MetaDataDictionaryArrayUpdateOn()
        #reader.LoadPrivateTagsOn()
        if len(seriesDicomFiles) != 1:
            image = reader.Execute() # image 3D
        # an alternative way
        # reader = sitk.ImageFileReader()
        # reader.LoadPrivateTagsOn()
        # imageList = []
        # for seriesDicomFile in seriesDicomFiles:
        #     reader.SetFileName(seriesDicomFile)
        #     imageList.append(reader.Execute())
        # imageList2D = [image[:,:,0] for image in imageList]
        # image = sitk.JoinSeries(imageList2D, \
        #         imageList[0].GetOrigin()[2], \
        #         imageList[1].GetOrigin()[2] - imageList[0].GetOrigin()[2])
        # image.SetDirection(imageList[0].GetDirection())
        # end of an alternative way
        # get metadata using ImageFileReader
        reader = sitk.ImageFileReader()
        reader.LoadPrivateTagsOn()
        reader.SetFileName(seriesDicomFiles[0])
        imageslice = reader.Execute()
        if len(seriesDicomFiles) == 1:
            dicomSeriesList.append(imageslice)
            continue
        for key in imageslice.GetMetaDataKeys():
            value = imageslice.GetMetaData(key)
            if isinstance(value, str):
                value = autoDecode(value)
            image.SetMetaData(key, value)
        dicomSeriesList.append(image)
    return dicomSeriesList

def autoDecode(s):
    b = s.encode('utf-8', errors='surrogateescape')
    for encodingName in ['utf-8', 'gb18030', 'latin-1']:
        try:
            s = b.decode(encodingName)
            return s
        except (UnicodeDecodeError,):
            #continue
            pass
    return 'UNKNOWN ENCODING bytes: '+str(b)
    
def IfSagittalPD_SIEMENS(image):
    SeriesDescription = image.GetMetaData(dicomTags['SeriesDescription'])
    SeriesDescription = SeriesDescription.strip()
    descriptionList = SeriesDescription.split('_')
    if 'pd' in descriptionList and 'sag' in descriptionList:
        return True
    return False

def IfSagittalPD_PHILIPS(image):
    descriptions=['PDW_SPAIR', 'PDW_SPAIR NEW', 'T1W_aTSE', 'SURVEY', \
            'WATER IMAGE', 'T2W_SPAIR', '3D_mFFE']
    orientations=['SAGITTAL', 'CORONAL', 'TRANSVERSAL']
    SeriesDescription = image.GetMetaData(dicomTags['SeriesDescription'])
    SeriesDescription = SeriesDescription.strip()
    SliceOrientation = image.GetMetaData(dicomTags['SliceOrientation'])
    SliceOrientation = SliceOrientation.strip()
    assert SeriesDescription in descriptions, \
            'UNKNOWN SeriesDescription ' + SeriesDescription
    assert SliceOrientation in orientations, \
            'UNKNOWN Orientation ' + Orientation
    if SeriesDescription in ['PDW_SPAIR', 'PDW_SPAIR NEW']:
        if SliceOrientation == 'SAGITTAL':
            return True
    return False

def IfSagittal(image):
    unitZ = np.array([0, 0, 1], dtype=np.float64)
    unitL = np.array([1, 0, 0], dtype=np.float64)

    D = np.array(image.GetDirection(), dtype=np.float64).reshape(3,3)
    S = np.diag(np.array(image.GetSpacing(), dtype=np.float64))
    v = unitZ
    o = np.array(image.GetOrigin(), dtype=np.float64)
    #directionZ = D.dot(D).dot(v) + o
    directionZ = D.dot(v)
    #print(directionZ)
    theta = np.arccos(directionZ.dot(unitL))*180/np.pi
    #print(theta)
    if theta < 30 or theta > 150:
        return True
    else:
        return False


def IfSagittalPD(image):
    if image.GetDimension() != 3:
        return False
    if image.GetDepth() <= 1:
        return False
    if image.GetMetaData(dicomTags['Modality']) != 'MR':
        return False
    seriesDescription = image.GetMetaData(dicomTags['SeriesDescription'])
    if 'PD' not in seriesDescription and 'pd' not in seriesDescription:
        return False
    return IfSagittal(image)

def ReadImage(path):
    print(path)
    if os.path.isdir(path):
        imageList = ReadDICOMSeriesRecursively(path)
        assert len(imageList) >= 1, 'No DICOM image found'
    elif os.path.isfile(path):
        imageList = sitk.ReadImage(path)
        #imageList = sitk.GetArrayFromImage(imageList)
    else:
        print("it's a special file(socket,FIFO,device file)")
    return imageList

def ReadROI(path):

    ROI = sitk.ReadImage(path)
    return ROI

def ReadSagittalPDs(dicomPath):
    # read dicom series from dicom directory, and return Sagittal PD image.
    # if get multiple Sagittal PD images, only return the last one
    # if found no Sagittal PD, raise an exception
    imageList = ReadDICOMSeriesRecursively(dicomPath)
    saggitalPDList = imageList
    #saggitalPDList = list(filter(IfSagittalPD, imageList))
    assert len(saggitalPDList) >= 1, 'No Sagittal PD image found'
    # only return the last one
    #return saggitalPDList[-1]
    return saggitalPDList
    #timeList=[]
    #for image in saggitalPDList:
    #    sd = float(image.GetMetaData(dicomTags['SeriesDate']))
    #    st = float(image.GetMetaData(dicomTags['SeriesTime']))
    #    timeList.append(sd[
    #print(seriesDateList)
    #print(seriesTimeList)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dicomPath = sys.argv[1]
    else:
        dicomPath = \
            '/home/kylexuan/data/data/cartilage_origin/FromXuhua/PD/A102817260'
    dicomSeriesList = ReadDICOMSeriesRecursively(dicomPath)
    for image in dicomSeriesList:
        size = image.GetSize()
        print(image.GetMetaData('0008|0050'), image.GetMetaData('0008|103e'))
    for image in list(ReadSagittalPDs(dicomPath)):
        print('PD Sagittal', image.GetMetaData(dicomTags['SeriesInstanceUID']))
        print(image.GetMetaData('0008|0050'), image.GetMetaData('0008|103e'))
