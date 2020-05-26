#!/usr/bin/python3

from functools import reduce
import numpy as np
import scipy
import scipy.ndimage

class Transform(object):
    r'''base type of a transformation
    '''
    def __init__(self):
        pass

    def __call__(self, array):
        pass


class Affine(Transform):
    def __init__(self, TransformMatrix=((1,0,0),(0,1,0),(0,0,1))):
        super(TransformAffine, self).__init__()
        TransformMatrix = np.array(TransformMatrix, dtype=np.float64)
        assert TransformMatrix.shape == (3,3), \
                'Affine transformation matrix shoud be 3x3'
        self.matrix = TransformMatrix

    def __call__(self, array, output_shape=None, **kw)
        if output_shape is None:
            output_shape = array.shape
        else:
            output_shape = np.array(output_shape)
        assert array.ndim == 2, 'for 2D array only'
        assert output_shape.shape == (2,), 'output_shape shoud be (2,)'
        assert issubclass(output_shape.dtype, np.integer), \
                'output_shape shoud be array of interger'
        # move image center to origin
        center2origin = np.array(( \
                (1, 0, -array.shape[1]/2), \
                (0, 1, -array.shape[0]/2),
                (0, 0, 1)), dtype=np.float64)
        # restore image center from origin
        origin2center = np.array(( \
                (1, 0, output_shape[1]/2), \
                (0, 1, output_shape[0]/2),
                (0, 0, 1)), dtype=np.float64)
        matrix = reduce(np.dot, [origin2center, self.matrix, center2origin])
        # other default options of scipy.ndimage.affine_transform:
        # output=None, order=3, mode='constant', cval=0
        return scipy.ndimage.affine_transform( \
                array, matrix, output_shape, **kw)
    
    def inv(self):
        return TransformAffine(np.linalg.inv(self.matrix))

class Translation(TransformAffine):
    pass

class Rotation(TransformAffine):
    pass

class Scale(TransformAffine):
    pass

class Reflection(TransformAffine):
    pass

class Shear(TransformAffine):
    pass

class Elastic(Transform):
    r'''scipy.ndimage.map_coordinates
    reference: https://github.com/DLTK/DLTK/blob/master/dltk/io/augmentation.py
    '''

    pass


class TransformList(Transform):
    r'''Hold transform
    all transformations will be concatenated together thus
    interpolation will only take place once
    '''
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __setitem__(self, idx, transform):
        pass

    def __delitem__(self, idex):
        pass

    def __len__(self):
        pass

    def __iter__(self):
        pass

    def __iadd__(transform):
        pass

    def __call__(self, array):
        pass

class RandomAffine():
    pass

class RandomElastic():
    pass

class RandomNoise():
    pass

class RandomBias():
    pass

class ToTensor():
    pass

class Normalize():
    pass
