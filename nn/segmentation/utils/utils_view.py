#!/usr/bin/python3

import cv2
cv2.setNumThreads(0)
import numpy as np
import torch

def view_tensor_label(label, classes, title='label'):
    label = label.numpy()
    assert len(classes) >= 2
    label = 255.0*label/max(classes)
    label = label.astype(np.uint8)
    label = cv2.applyColorMap(label, cv2.COLORMAP_JET)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, label)
    cv2.waitKey(0)

def view_tensor_image(image, title='image'):
    image = image.numpy()[0,:,:]
    if image.max() != image.min():
        image = (image - image.min())*255.0/(image.max() - image.min())
    image = image.astype(np.uint8)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, image)
    cv2.waitKey(0)

def view_tensor_P(image, output, classes, title='P'):
    '''view image & probability(network output)
    '''
    raise NotImplementedError
    label = data[1].numpy()
    assert len(classes) >= 2
    label = 255.0*label/max(classes)
    label = label.astype(np.uint8)
    label = cv2.applyColorMap(label, cv2.COLORMAP_JET)
    image = data[0].numpy()[0,:,:]
    image = (image - image.min())*255.0/(image.max() - image.min())
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = image.astype(np.uint8)
    #cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title, image)
    cv2.createTrackbar('alpha', title, 0, 255, \
            lambda v: cv2.imshow( \
            title,cv2.addWeighted( \
            image, 1 - cv2.getTrackbarPos('alpha', title)/255.0, \
            label, cv2.getTrackbarPos('alpha', title)/255.0, 1)))
    cv2.waitKey(0)
    pass

def view_tensor_data(data, classes, title='data'):
    '''view image & label
    '''
    assert len(classes) >= 2
    image = data[0].numpy()[0,:,:]
    image = (image - image.min())*255.0/(image.max() - image.min())
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = image.astype(np.uint8)
    label = data[1].numpy()
    label = 255.0*label/max(classes)
    label = label.astype(np.uint8)
    label = cv2.applyColorMap(label, cv2.COLORMAP_JET)
    #cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title, image)
    cv2.createTrackbar('alpha', title, 0, 255, \
            lambda v: cv2.imshow( \
            title,cv2.addWeighted( \
            image, 1 - cv2.getTrackbarPos('alpha', title)/255.0, \
            label, cv2.getTrackbarPos('alpha', title)/255.0, 1)))
    cv2.waitKey(0)

if __name__ == '__main__':
    label = torch.ones(100,100)
    image = torch.randn(1,100,100)
    view_tensor_data((image, label), [0,1])
