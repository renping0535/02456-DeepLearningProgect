import torch
import os
import random
import glob
from PIL import Image
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
from SegNet import net


# mapping = {
#     0: 0, # no data
#     76: 1, # soil
#     149: 2, # crops
#     225: 3 # weeds
# }

net.load_state_dict(torch.load("/home/renping/02456 DeepLearning Project/Archive8/net_trained/net_trained.pt"))
net.eval()
use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Running on GPU!")
    net = net.cuda()

path_to_images = "/home/renping/02456 DeepLearning Project/test/cropped/raw/"
path_to_annotations = "/home/renping/02456 DeepLearning Project/test/cropped/anno/"

img_list = glob.glob(path_to_images+os.sep+"*.png")
img_list = sorted(img_list)
anno_list = glob.glob(path_to_annotations+os.sep+"*.png")
anno_list = sorted(anno_list)


preds = []
for rand_int in range(len(img_list)):
    # rand_int = random.randint(0, len(img_list)-1)
    # rand_int = random.randint(0, len(img_list)-1)

    raw_img = np.array(Image.open(img_list[rand_int]))
    show_anno = np.array(Image.open(anno_list[rand_int]).convert("L"))


    show_raw = raw_img

    img = torch.from_numpy(raw_img).type(torch.FloatTensor).cuda()
    img = img.permute(2,0,1)
    # print(img.shape)

    img = img.unsqueeze(0)

    # print(img.shape)

    pred = net(img)

    # print(pred.shape)

    pred = pred.squeeze()
    # print(pred.shape)

    pred_img = torch.argmax(pred, dim=0).cpu().numpy()
    print("{}th image:{} ".format(rand_int+1, np.unique(pred_img)))
    preds.append(pred_img)

plt.imshow(preds[16])
plt.show()

# # print(pred_img.shape)


# mapping_rev = {
#     0:0, # 0 no data
#     1:76, # 1 soil
#     2:149, # 2 crop
#     3:225 # 3 weed
# }
# for item in mapping_rev:
#     pred_img[pred_img==item] = mapping_rev[item]

# print(np.unique(pred_img))

# num_class0 = np.sum(pred_img==0)
# num_class1 = np.sum(pred_img==76)
# num_class2 = np.sum(pred_img==149)
# num_class3 = np.sum(pred_img==225)
# total = 512*512
# print("%_class0 = {}".format(num_class0/total))
# print("%_class1 = {}".format(num_class1/total))
# print("%_class2 = {}".format(num_class2/total))
# print("%_class3 = {}".format(num_class3/total))


# plt.figure(figsize=(15,15))
# plt.subplot(1,3,1)
# plt.imshow(show_raw)
# plt.title("raw")
# # plt.colorbar()
# plt.subplot(1,3,2)
# plt.imshow(show_anno)
# plt.title("anno")
# plt.colorbar()
# plt.subplot(1,3,3)
# plt.imshow(pred_img)
# plt.title("pred")
# plt.colorbar()
# plt.show()