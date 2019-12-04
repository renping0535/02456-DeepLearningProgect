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
path_train = '/home/renping/02456 DeepLearning Project/train/cropped/raw'
path_anno = '/home/renping/02456 DeepLearning Project/train/cropped/anno'


batches = 10
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
    for k in mapping:
        anno_reshaped[anno_reshaped==k] = mapping[k]
    return anno_reshaped

# Convert imgaug segmap to 2d with classes
def seg_to_anno(seg):
    anno = rgb2gray(seg).astype(int)
    for k in mapping2:
        anno[anno==k] = mapping2[k]
    return anno

# ============================================================================
# Image Augmentors
seq = iaa.Sequential([
    iaa.HorizontalFlip(0.5),
    iaa.Affine(rotate=(-180, 180)),
    # iaa.Dropout(p=(0, 0.1)),
    # iaa.Sharpen((0.0, 1.0)),
    # iaa.ElasticTransformation(alpha=50, sigma=5),
    iaa.CropToFixedSize(width=crop_size, height=crop_size)
], random_order=False)

# Class to load + process data
class Custom_Data(data.Dataset):
    def __init__(self, path_train, path_anno):

        # Get all raw + annotated images
        raw_img = os.listdir(path_train)
        anno_img = os.listdir(path_anno)
        raw_img.sort()
        anno_img.sort()
        raw_imgs = [os.path.join(path_train, img) for img in raw_img]
        anno_imgs = [os.path.join(path_anno, img) for img in anno_img]

        # Select only images with at least 3 classes represented
        raw_new, anno_new = [], []
        thresh = 512 * 512 / 2
        for i in range(len(anno_imgs)):
            segmap = create_anno(np.array(Image.open(anno_imgs[i])))
            if (len(np.unique(segmap))>=3 and len(segmap[segmap == 0])<thresh):
                raw_new = np.append(np.append(raw_new, raw_imgs[i]), raw_imgs[i])
                anno_new = np.append(np.append(anno_new, anno_imgs[i]), anno_imgs[i])

        # print(len(raw_imgs), len(raw_new))
        self.raw_img = raw_new
        self.anno_img = anno_new

    def __getitem__(self, index, plots=False):

        # Get raw img + segmap at index
        raw_img_path = self.raw_img[index]
        raw_img = np.array(Image.open(raw_img_path))
        anno_img_path = self.anno_img[index]
        anno_img = np.array(Image.open(anno_img_path))

        # Format segmap
        anno_img = create_anno(anno_img).astype('int32')
        seg_map = SegmentationMapsOnImage(anno_img, shape=anno_img.shape)

        # print(raw_img_path, anno_img_path)

        # Perform data augmentations to generate 2 sets of augmented data
        raw_aug, seg_aug = seq(image=raw_img, segmentation_maps=seg_map)
        anno_aug = seg_aug.draw()[0]
        anno_aug = seg_to_anno(anno_aug)

        # Plot images to compare
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

# # ============================================================================
# # Testing

# # Initialize
train_data = Custom_Data(path_train, path_anno)
# data_len = train_data.__len__()

# # Test with data loader
train_loader = DataLoader(train_data, batch_size=batches, shuffle=True)


# train_iter = iter(train_loader)
# train_input, train_target = train_iter.next()
# print("train_input size = {}.".format(train_input.size()))
# print("train_target size = {}.".format(train_target.size()))














# start = time()
# for (raw_aug, anno_aug) in train_loader:
#     break
# end = time()
# print(end-start)

# print(raw_aug.shape)
# print(anno_aug.shape)

# # Test with manual loop
# start = time()
# imgs = np.zeros((batches,crop_size,crop_size,3))
# annos = np.zeros((batches,crop_size,crop_size))
# for i in range(batches):
#     (img, anno) = train_data.__getitem__(index=i)
#     imgs[i] = img
#     annos[i] = anno
# end = time()
# print(end-start)

# print(imgs.shape)
# print(annos.shape)

# # plt.figure()
# # plt.subplot(1,2,1)
# # plt.imshow(imgs[0])
# # plt.subplot(1,2,2)
# # plt.imshow(annos[0])
# # plt.show()