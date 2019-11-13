#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 19:47:03 2019

@author: renping

Usage: check the RGB value of every pixel from a random image
"""

from PIL import Image
import glob
import os
import random
import matplotlib.pyplot as plt

path_to_annotations = '/home/renping/02456 DeepLearning Project/train/cropped/anno'
anno_list = glob.glob(path_to_annotations + os.sep + "*.png")
anno_list = sorted(anno_list)
rand_int = random.randint(0, len(anno_list))
#rand_int = 90
path_to_images = '/home/renping/02456 DeepLearning Project/train/cropped/raw'
img_list = glob.glob(path_to_images + os.sep + "*.png")
img_list = sorted(img_list)


im_anno = Image.open(anno_list[rand_int])
im_raw = Image.open(img_list[rand_int])


pix = im_anno.load()
width = im_anno.size[0]
height = im_anno.size[1]
for x in range(width):
    for y in range(height):
        r, g, b = pix[x, y]
        print("({},{}) pixel RGB is ({},{},{}) of the {} image.".format(x,y,r,g,b,anno_list[rand_int]))

plt.figure()
plt.suptitle(anno_list[rand_int])
plt.subplot(1,2,1)
plt.imshow(im_anno)
plt.title("annotaion image")
plt.subplot(1,2,2)
plt.imshow(im_raw)
plt.title("raw image")
plt.show()