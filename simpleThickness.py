#!/usr/bin/python3
import numpy as np
import SimpleITK as sitk

def ApplySliceBySlice(func, image):
    assert image.GetDimension() == 3
    results = []
    for i in range(image.GetDepth()):
        extractSize = list(image.GetSize())
        extractSize[-1] = 0
        extractIndex = [0]*(image.GetDimension()-1) + [i]
        #frame = sitk.RegionOfInterest(mask, extractSize, extractIndex)
        #don't care about direction
        frame = sitk.Extract(image, extractSize, extractIndex, \
                sitk.ExtractImageFilter.DIRECTIONCOLLAPSETOIDENTITY)
        results.append(func(frame))
    #boundaries = sitk.Tile(boundaries, (1,1,0))
    array3D = np.stack(list(map(sitk.GetArrayFromImage, results)), axis=0)
    result = sitk.GetImageFromArray(array3D)
    result.CopyInformation(image)
    return result

def getDirection(maskTarget, maskRef):
    '''Extract the inner and outer boundary of femur cartilage.
    Inner and outer boundary will be labeled as 1 and 2 seperately.
    For input (mask), femur and femur cartilage area is expected to be labeled
    as 1 and 3 seperately
    '''
    assert maskTarget.GetSize() == maskRef.GetSize()

    sigma = 1.0/3*4

    distanceRef = sitk.SignedMaurerDistanceMap( \
            maskRef, squaredDistance=False, useImageSpacing=True)
    distanceRef = sitk.DiscreteGaussian( \
            distanceRef, sigma, 32, 0.01, True)
    gradientRef = sitk.Gradient(distanceRef, \
            useImageSpacing=True, useImageDirection=True)

    distanceTarget = sitk.SignedMaurerDistanceMap( \
            maskTarget, squaredDistance=False, useImageSpacing=True)
    distanceTarget = sitk.DiscreteGaussian( \
            distanceTarget, sigma, 32, 0.01, True)
    gradientTarget = sitk.Gradient(distanceTarget, \
            useImageSpacing=True, useImageDirection=True)

    # inner product
    direction = sitk.Image(gradientTarget.GetSize(), sitk.sitkFloat32, 1)
    direction.CopyInformation(gradientTarget)
    for i in range(gradientTarget.GetNumberOfComponentsPerPixel()):
        compRef = sitk.VectorIndexSelectionCast( \
                gradientRef, i, sitk.sitkFloat32)
        compTarget = sitk.VectorIndexSelectionCast( \
                gradientTarget, i, sitk.sitkFloat32)
        direction += compRef * compTarget
    #result = gradientFemur * gradientCartilage

    return direction

def getSurface(mask, radius=1):
    #surfaceCartilage = sitk.BinaryContour(maskCartilage)
    assert radius==1 or radius==2, 'radius of 1 or 2 is supported'
    if radius==1:
        surface = sitk.BinaryDilate(mask, 1, sitk.sitkBall) - mask
    elif radius==2:
        surface = sitk.BinaryDilate(mask, 1, sitk.sitkBall) - \
                sitk.BinaryErode(mask, 1, sitk.sitkBall)
    return surface


def getCartilageThickness(mask, cartilage=2, bone=1):
    eps = 1e-6
    labelFemur = bone
    labelCartilage = cartilage
    #erode bone/cart for better reference when diffing inner/outer boundary
    erosionB = 20
    dilationC = 3
    cartilage = mask == labelCartilage
    nonCartilage = mask != labelCartilage
    femur = mask == labelFemur
    surface = ApplySliceBySlice(getSurface, cartilage)
    direction = ApplySliceBySlice( \
            lambda img: getDirection( \
            sitk.BinaryDilate( \
            img == labelCartilage, dilationC, sitk.sitkBall),
            sitk.BinaryErode( \
            img == labelFemur, erosionB, sitk.sitkBall)),
            mask)
    tmp = sitk.Cast(nonCartilage, sitk.sitkFloat32) * direction
    innerBoundary = surface*(tmp < -eps)
    outerBoundary = surface*(tmp > eps)
    distance = ApplySliceBySlice( \
            lambda x: sitk.SignedMaurerDistanceMap(x, \
            squaredDistance=False, useImageSpacing=True), \
            outerBoundary)
    distance = sitk.Cast(distance < 100.0, sitk.sitkFloat32)*distance
    thickness = sitk.Cast(innerBoundary, sitk.sitkFloat32)*distance
    thickness = sitk.GetArrayFromImage(thickness)
    thickness = thickness[np.nonzero(thickness)]
    return thickness.mean()


if __name__ == '__main__':
    import sys
    mask = sitk.ReadImage(sys.argv[1])
    f = getCartilageThickness(mask, cartilage=2, bone=1)
    t = getCartilageThickness(mask, cartilage=4, bone=3)
    p = getCartilageThickness(mask, cartilage=6, bone=5)
    print(str(sys.argv[2])+','+str(f)+','+str(t)+','+str(p))

