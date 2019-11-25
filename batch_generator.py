import torch as t
from torch.utils import data
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
from compute_mean_v1 import Compute_Mean_Std
import matplotlib.pyplot as plt

import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

path_train = '/Users/cmiao/Cellari-Deeplearning-Course/train/cropped/raw'
path_anno = '/Users/cmiao/Cellari-Deeplearning-Course/train/cropped/anno'

normMean, normStd = Compute_Mean_Std(path_train)

def create_anno(anno):
    anno_reshaped = np.zeros((anno.shape[0], anno.shape[1]))
    for x in range(anno.shape[0]):
        for y in range(anno.shape[1]):
            if (anno[x,y,0] > 0 and anno[x,y,1] > 0):
                anno_reshaped[x,y] = 1
            elif (anno[x,y,0] > 0):
                anno_reshaped[x,y] = 2
            elif (anno[x,y,1] > 0):
                anno_reshaped[x,y] = 3
    return anno_reshaped

def seg_to_anno(seg):

    seg_2d = np.zeros((seg.shape[0], seg.shape[1]))
    anno = np.zeros((seg.shape[0], seg.shape[1]))

    for x in range(seg.shape[0]):
        for y in range(seg.shape[1]):
            val = int(str(seg[x,y,0]) + str(seg[x,y,1]) + str(seg[x,y,1]))
            seg_2d[x,y] = val

    cols = np.unique(seg_2d)

    for x in range(seg_2d.shape[0]):
        for y in range(seg_2d.shape[1]):
            for i in range(len(cols)):
                if seg_2d[x,y] == cols[i]:
                    anno[x,y] = i
    return anno


seq = iaa.Sequential([
    iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
    iaa.Sharpen((0.0, 1.0)),       # sharpen the image
    iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
    iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
], random_order=True)

class Custom_Data(data.Dataset):
    def __init__(self, path_train, path_anno):
        raw_img = os.listdir(path_train)
        anno_img = os.listdir(path_anno)
        raw_img.sort()
        anno_img.sort()
        self.raw_img = [os.path.join(path_train, img) for img in raw_img]
        self.anno_img = [os.path.join(path_anno, img) for img in anno_img]

    def __getitem__(self, index):
        raw_img_path = self.raw_img[index]
        raw_img = np.array(Image.open(raw_img_path))
        anno_img_path = self.anno_img[index]
        anno_img = np.array(Image.open(anno_img_path))
        anno_img = create_anno(anno_img).astype('int32')
        seg_map = SegmentationMapsOnImage(anno_img, shape=anno_img.shape)

        print()
        print(raw_img_path)
        print(anno_img_path)

        # Perform data augmentations
        raw_aug, seg_aug = seq(image=raw_img, segmentation_maps=seg_map)
        anno_aug = seg_aug.draw()[0]

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

train_data = Custom_Data(path_train, path_anno)
data_len = train_data.__len__()

train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
data_iterator = iter(train_loader)
raw_data, anno_data = data_iterator.next() # no transform form CPU to GPU

idx = 0
(random_img, random_anno) = train_data.__getitem__(index=idx)
random_anno = seg_to_anno(random_anno)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(random_img)
plt.subplot(1,2,2)
plt.imshow(random_anno)
plt.show()


# for i, data in enumerate(data_iterator, 0):
#     inputs, labels = data
#     print(inputs.size())
#     print(labels.size())
#     print(i)
#     print()
# for i, (raw_img, anno_img) in enumerate(data_iterator):
#     print("batch {}: raw image shape = {}, anno image shape = {}.".format(i, raw_img.size(), anno_img.size()))
