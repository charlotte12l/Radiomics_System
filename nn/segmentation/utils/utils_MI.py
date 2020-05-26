#!/usr/bin/python3
# implementation of mutual information

import sys
import numpy as np
import nibabel as nib

def entropy(pdf, eps=1e-8):
    H = np.multiply(pdf+eps, np.log2(pdf+eps))
    return -H.sum()

def MI(X, Y, bins=256):
    X = X.ravel()
    Y = Y.ravel()

    H_XY, _, _ = np.histogram2d(X, Y, bins, normed=True)
    H_XY = entropy(H_XY)

    H_X, _ = np.histogram(X, bins, density=True)
    H_X = entropy(H_X)

    H_Y, _ = np.histogram(Y, bins, density=True)
    H_Y = entropy(H_Y)

    return H_X + H_Y - H_XY

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: ', sys.argv[0], ' fixed, float0, float1, ...')
        sys.exit(1)
    fixedimg = nib.load(sys.argv[1]).get_data()
    shape = fixedimg.shape
    floatimg=[]
    score=[]
    for path in sys.argv[2:]:
        floatimg.append(nib.load(path).get_data())
        assert shape == floatimg[-1].shape, \
                sys.argv[1]+":"+str(shape)+" and "+\
                path+":"+str(floatimg[-1].shape)+" shoud have same size"
    for img in floatimg:
        score.append(MI(fixedimg, img))
    i = score.index(max(score))
    print(score[i], ',', sys.argv[2+i])

