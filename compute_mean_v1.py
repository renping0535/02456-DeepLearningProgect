#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 19:50:25 2019

@author: renping

Usage: check the Mean and Stnadard Division for augmentation in torchvision.transforms.Normalize()
"""

import os
import numpy as np
from scipy.misc import imread
 

def Compute_Mean_Std(path):

    R_channel = 0
    G_channel = 0
    B_channel = 0
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = imread(os.path.join(filepath, filename)) / 255.0
        R_channel = R_channel + np.sum(img[:, :, 0])
        G_channel = G_channel + np.sum(img[:, :, 1])
        B_channel = B_channel + np.sum(img[:, :, 2])
    
    num = len(pathDir) * 512 * 512  # img_height, img_weight
    R_mean = R_channel / num
    G_mean = G_channel / num
    B_mean = B_channel / num
    
    R_channel = 0
    G_channel = 0
    B_channel = 0
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = imread(os.path.join(filepath, filename)) / 255.0
        R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
        G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
        B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)
    
    R_std = np.sqrt(R_channel / num)
    G_std = np.sqrt(G_channel / num)
    B_std = np.sqrt(B_channel / num)

    mean = [R_mean, G_mean, B_mean]
    std = [R_std, G_std, B_std]
    return mean, std



filepath = "/home/renping/02456 DeepLearning Project/train/cropped/raw"  # the path of train data
pathDir = os.listdir(filepath)

Norm_mean, Norm_std = Compute_Mean_Std(pathDir)
print("mean={}".format(Norm_mean))
print("std={}".format(Norm_std))
# print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
# print("R_std is %f, G_std is %f, B_std is %f" % (R_std, G_std, B_std))