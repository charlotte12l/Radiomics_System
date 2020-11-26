# Radiomics System

## Introduction
This is a radiomics-computing system which supports 3D medical image visualization, annotation, feature extraction and analysis.
Video demos and figures can be seen on the [website](https://sites.google.com/view/radiomics-computing-platform/homepage).

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
 A GUI interface would appear. Here is a demo on how to use the package.

<video src="https://youtu.be/mqR0WP9Neh8" controls="controls" width="500" height="300">Demo</video>

