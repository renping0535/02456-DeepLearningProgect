#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 19:50:25 2019

@author: renping

Usage: The functionality of this script is to augment our train data since we have limited train images.
"""

import torch as t
from torch.utils import data
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
from torchvision import transforms as T
from compute_mean_v1 import Compute_Mean_Std
import torchvision
import matplotlib.pyplot as plt

# the path of your train data
path_train = '/home/renping/02456 DeepLearning Project/train/cropped/raw'
# the path of your corresponding annotated data
path_anno = '/home/renping/02456 DeepLearning Project/train/cropped/anno'

normMean, normStd = Compute_Mean_Std(path_train)
# normMean = [0.5,0.5,0.5]
# normStd = [0.5,0.5,0.5]
transform = [T.Compose([
    T.RandomCrop(256, padding=4),
    T.RandomRotation(30),
    T.ToTensor(),
    T.Normalize(normMean, normStd, inplace=True)
]), T.Compose([
    T.RandomCrop(256, padding=4),
    T.RandomRotation(30),
    T.ToTensor(),
])]

class Custom_Data(data.Dataset):
    def __init__(self, path_train, path_anno, transforms = None):
        raw_img = os.listdir(path_train) # raw images
        anno_img = os.listdir(path_anno) # anno images
        self.raw_img = [os.path.join(path_train, img) for img in raw_img]
        self.anno_img = [os.path.join(path_anno, img) for img in anno_img]
        self.transforms = transforms
   
    def __getitem__(self, index):
        raw_img_path = self.raw_img[index]
        pil_raw_img = Image.open(raw_img_path)
        anno_img_path = self.anno_img[index]
        pil_anno_img = Image.open(anno_img_path)
        if self.transforms:
            raw_data = self.transforms[0](pil_raw_img) # transform[0] for raw image
            anno_data = self.transforms[1](pil_anno_img) # transform[1] for annotated image
        return (raw_data, anno_data)

    def __len__(self):
        return len(self.raw_img)



train_data = Custom_Data(path_train, path_anno, transforms=transform)
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)

data_iterator = iter(train_loader)
raw_data, anno_data = data_iterator.next() # no transform form CPU to GPU
print("raw data set size = {}.".format(raw_data.size()))
print("anno data set size = {}.".format(anno_data.size()))

print(raw_data.type())
print(anno_data.type())

"""
code block to check a given idx raw image and annotated image

print(train_data.__len__())
idx = 15
print(train_data.__getitem__(idx)[0].size())
random_img = train_data.__getitem__(idx)[0] # raw
random_img = random_img.permute(1,2,0) # raw
random_anno = train_data.__getitem__(idx)[1] # corresponding anno
random_anno = random_anno.permute(1,2,0) # corresponding anno
print(random_img.size())
plt.figure()
plt.subplot(1,2,1)
plt.imshow(random_img)
plt.subplot(1,2,2)
plt.imshow(random_anno)
plt.show()
"""

# for i, data in enumerate(data_iterator, 0):
#     inputs, labels = data
#     print(inputs.size())
#     print(labels.size())
#     print(i)
#     print()
# for i, (raw_img, anno_img) in enumerate(data_iterator):
#     print("batch {}: raw image shape = {}, anno image shape = {}.".format(i, raw_img.size(), anno_img.size()))