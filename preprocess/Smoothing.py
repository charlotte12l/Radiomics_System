from scipy.ndimage import gaussian_filter
import numpy as np
import cv2
from scipy.signal import medfilt2d


def guassianSmooth(image, level):
    new = np.zeros(np.shape(image))
    for i in range(np.shape(image)[0]):
        new[i,:,:] = cv2.GaussianBlur(image[i,:,:],(level,level),0)
    return new

def meanSmooth(image, level):
    new = np.zeros(np.shape(image))
    for i in range(np.shape(image)[0]):
        new[i,:,:] = cv2.blur(image[i,:,:],(level,level))
    return new

def medianSmooth(image, level):
    new = np.zeros(np.shape(image))
    for i in range(np.shape(image)[0]):
        new[i,:,:] = medfilt2d(image[i,:,:],(level,level))
    return new

'''
class MeanSmooth(object):
    def __init__(self):
        super(Denoise, self).__init__()

    def __call__(self, *args, **kw):
        return self.evaluate(*args, **kw)

    def evaluate(self, image, level):
        return denoise_wavelet(image, multichannel=False, rescale_sigma=True)

class MedianSmooth(object):
    def __init__(self):
        super(Denoise, self).__init__()

    def __call__(self, *args, **kw):
        return self.evaluate(*args, **kw)

    def evaluate(self, image, level):
        return denoise_wavelet(image, multichannel=False, rescale_sigma=True)
'''