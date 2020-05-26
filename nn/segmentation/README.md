# MRI Image Segmentation

## Dependency

  1. Python3

  1. Pytorch (tested under 0.2)

  1. nibabel

  1. SimpleITK

  1. Tensorflow (TensorBoard for visualizing training)

## Steps

### Preprocess

  1. Bias correction.

  1. Histogram matching.

  1. Intensity cliping.(Not Implemented)

    Deal with outlier voxels.
    Sort voxel intensities and cliping intensity out of the
    range, for example, [0.005, 0.995].
    This helps eliminate bad voxel intensities of an image.

### Prepare training/testing data.
  1. Data augmentation.

    1. Interpolation.

    1. Rotation.
      * Rotate different degrees along x, y, z seperately.
      
        For example: random degrees among
        [-\pi/20, +\pi/20] along x, y, z axises.

    1. Deformation.

### Train& testing.

  * Training V-net for segmentation.
