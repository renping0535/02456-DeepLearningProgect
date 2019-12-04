import torch
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as T
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Custom_Data(data.Dataset):
    def __init__(self, path_test_input, path_test_target):
        input = os.listdir(path_test_input)
        self.input = sorted([os.path.join(path_test_input, a) for a in input])

        target = os.listdir(path_test_target)
        self.target = sorted([os.path.join(path_test_target, a) for a in target])

        self.mapping = {
            0:0,
            76:1,
            149:2,
            225:3
        }

    def mapping_to_class(self, target):
        for k in self.mapping:
            target[target==k] = self.mapping[k]
        return target

    def __getitem__(self, index):
        # test input data
        test_input_path = self.input[index]
        test_input = Image.open(test_input_path)
        # test_input = T.RandomCrop(256)(test_input)
        test_input = T.ToTensor()(test_input)

        # test target data
        test_target_path = self.target[index]
        test_target = Image.open(test_target_path)
        # test_target = T.RandomCrop(256)(test_target)
        test_target = T.ToTensor()(test_target)
        test_target_RGB = test_target

        # mapping target to class index
        test_target = T.ToPILImage()(test_target).convert("L")
        test_target_grey = torch.from_numpy(np.array(test_target))
        test_target = self.mapping_to_class(test_target_grey)
    
        return test_input, test_target, test_target_RGB, test_target_grey, test_input_path, test_target_path

    def __len__(self):
        return len(self.input)

    


# the path of your test data
path_test_input = '/home/renping/02456 DeepLearning Project/test/cropped/raw'
# the path of your corresponding annotated data
path_test_target = '/home/renping/02456 DeepLearning Project/test/cropped/anno'



num_batch = 7
test_data = Custom_Data(path_test_input, path_test_target)
test_loader = DataLoader(test_data, batch_size=num_batch, shuffle=True)

# test_iter = iter(test_loader)
# test_input, test_target, test_target_RGB, test_target_grey, test_input_path, test_target_path = test_iter.next()
# print("test_input size = {}.".format(test_input.size()))
# print("test_target size = {}.".format(test_target.size()))
# print("test_target_grey size = {}.".format(test_target_grey.size()))

# print("image path of selected samples from batch_generator_test")
# print(np.array(test_input_path).reshape(-1,1))
# # print(test_input_path[2][59:])
# # print(len(test_input_path[0]))
# # print(len("/home/renping/02456 DeepLearning Project/test/cropped/raw/"))
# print(np.array(test_target_path).reshape(-1,1))
# print(torch.unique(test_target))


# # split test input
# test_input = test_input.permute(0,2,3,1)
# test_input_array = np.array(test_input)
# print(type(test_input_array))
# print(np.shape(test_input_array))

# # split test target RBG
# test_target_RGB = test_target_RGB.permute(0,2,3,1)
# test_target_RGB_array = np.array(test_target_RGB)
# print(type(test_target_RGB_array))
# print(np.shape(test_target_RGB_array))

# # split test target
# test_target_grey_array = np.array(test_target_grey)
# print(type(test_target_grey_array))
# print(np.shape(test_target_grey_array))


# plt.figure()
# plt.suptitle("test raw")
# for column in range(0, num_batch):
#         plt.subplot(1, num_batch, column+1)
#         plt.title(test_input_path[column][58:])
#         plt.imshow(test_input_array[column])

# plt.figure()
# plt.suptitle("test anno RGB")
# for column in range(0, num_batch):
#         plt.subplot(1, num_batch, column+1)
#         plt.title(test_target_path[column][59:])
#         plt.imshow(test_target_RGB_array[column])

# plt.figure()
# plt.suptitle("test anno grey")
# for column in range(0, num_batch):
#         plt.subplot(1, num_batch, column+1)
#         plt.title(test_target_path[column][59:])
#         plt.imshow(test_target_grey_array[column])
# plt.show()