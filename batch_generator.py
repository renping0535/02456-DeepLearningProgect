import torch as t
from torch.utils import data
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from time import time

# ============================================================================
# Parameters
path_train = '/Users/cmiao/Cellari-Deeplearning-Course/train/cropped/raw'
path_anno = '/Users/cmiao/Cellari-Deeplearning-Course/train/cropped/anno'

batches = 5
crop_size = 256

# ============================================================================
# Functions to convert rgb segmaps to 2d

# Convert rgb array to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Mapping for grayscale original segmap
mapping = {
    0: 0,
    76: 1,
    149: 2,
    225: 3
}

# Mapping for grayscale imgaug segmap
mapping2 = {
    0: 0,
    91: 1,
    132: 2,
    211: 3
}

# Convert original segmap to 2d with classes
def create_anno(anno):
    anno_reshaped = rgb2gray(anno).astype(int)
    # print(np.unique(anno_reshaped))
    for k in mapping:
        anno_reshaped[anno_reshaped==k] = mapping[k]
    # print(np.unique(anno_reshaped))
    return anno_reshaped

# Convert imgaug segmap to 2d with classes
def seg_to_anno(seg):
    anno = rgb2gray(seg).astype(int)
    # print(np.unique(anno))
    for k in mapping2:
        anno[anno==k] = mapping2[k]
    # print(np.unique(anno))
    return anno

# ============================================================================
# Image Augmentors
seq = iaa.Sequential([
    iaa.HorizontalFlip(0.5),
    iaa.Affine(rotate=(-180, 180)),
    # iaa.Dropout(p=(0, 0.1)),
    # iaa.Sharpen((0.0, 1.0)),
    # iaa.ElasticTransformation(alpha=50, sigma=5),
    iaa.CropToFixedSize(width=256, height=256)
], random_order=False)

# Class to load + process data
class Custom_Data(data.Dataset):
    def __init__(self, path_train, path_anno):

        # Get all raw + annotated images
        raw_img = os.listdir(path_train)
        anno_img = os.listdir(path_anno)
        raw_img.sort()
        anno_img.sort()
        self.raw_img = [os.path.join(path_train, img) for img in raw_img]
        self.anno_img = [os.path.join(path_anno, img) for img in anno_img]

    def __getitem__(self, index, plots=False):

        # Get raw img + segmap at index
        raw_img_path = self.raw_img[index]
        raw_img = np.array(Image.open(raw_img_path))
        anno_img_path = self.anno_img[index]
        anno_img = np.array(Image.open(anno_img_path))

        # Format segmap
        anno_img = create_anno(anno_img).astype('int32')
        seg_map = SegmentationMapsOnImage(anno_img, shape=anno_img.shape)

        # print()
        # print(raw_img_path)
        # print(anno_img_path)

        # Perform data augmentations
        raw_aug, seg_aug = seq(image=raw_img, segmentation_maps=seg_map)
        anno_aug = seg_aug.draw()[0]
        anno_aug = seg_to_anno(anno_aug)

        if plots:
            plt.figure()
            plt.subplot(2,2,1)
            plt.imshow(raw_img)
            plt.subplot(2,2,2)
            plt.imshow(anno_img)
            plt.subplot(2,2,3)
            plt.imshow(raw_aug)
            plt.subplot(2,2,4)
            plt.imshow(anno_aug)
            plt.show()

        return (raw_aug, anno_aug)

    def __len__(self):
        return len(self.raw_img)

# ============================================================================
# Testing

# Initialize
train_data = Custom_Data(path_train, path_anno)
data_len = train_data.__len__()

# Test with data loader
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
start = time()
for raw_aug, anno_aug in train_loader:
    break
end = time()
print(end-start)

print(raw_aug.shape)
print(anno_aug.shape)

# Test with manual loop
start = time()
imgs = np.zeros((data_len,crop_size,crop_size,3))
annos = np.zeros((data_len,crop_size,crop_size))
for i in range(5):
    (img, anno) = train_data.__getitem__(index=i)
    imgs[i] = img
    annos[i] = anno
end = time()
print(end-start)

print(imgs.shape)
print(annos.shape)

# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(imgs[0])
# plt.subplot(1,2,2)
# plt.imshow(annos[0])
# plt.show()
