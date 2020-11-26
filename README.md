# Radiomics System

## Introduction
This is a radiomics-computing system which supports 3D medical image visualization, annotation, feature extraction and analysis.
Video [demo](https://youtu.be/mqR0WP9Neh8) and figures can be found on the [website](https://sites.google.com/view/radiomics-computing-platform/homepage).

## Environment
This package is based on Python 3.6, and uses external packages like PyQt5, to reproduce the environment on your computer, please do:
```bash
conda env create -f environment.yml
```
If you do not need to use the CUDA-based feature extraction method, then either Win10 and Ubuntu 16.04 can work well for you.

However,if you hope to use the CUDA-based feature extraction method, you must make sure you are using Ubuntu and you can compile the [cuRadiomics](https://github.com/charlotte12l/Radiomics_System/cuRadiomics) codes correctly.
## How to use
After setting up the environment, please first activate the environment and then run the app.
```bash
conda activate bs
python app.py
```
 A GUI interface would appear:
 
 ![GUI](https://github.com/charlotte12l/Radiomics_System/blob/master/fig/MainWindow.png) 
 
 Here is a [demo](https://youtu.be/mqR0WP9Neh8) on how to use the package. You can find more on the  [website](https://sites.google.com/view/radiomics-computing-platform/homepage).
