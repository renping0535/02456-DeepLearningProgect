import torch.nn.functional as F
import torch.nn as nn
from torchsummary import summary
import torch

class Encode_conv_bn_x2(nn.Module):
    def __init__(self, in_, out):
        super(Encode_conv_bn_x2,self).__init__()

        batchNorm_momentum = 0.1
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_, out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out, momentum=batchNorm_momentum)
        self.conv2 = nn.Conv2d(out, out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out, momentum=batchNorm_momentum)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class Encode_conv_bn_x3(nn.Module):
    def __init__(self, in_, out):
        super(Encode_conv_bn_x3, self).__init__()

        batchNorm_momentum = 0.1
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_, out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out, momentum=batchNorm_momentum)
        self.conv2 = nn.Conv2d(out, out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out, momentum=batchNorm_momentum)
        self.conv3 = nn.Conv2d(out, out, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out, momentum=batchNorm_momentum)


    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x

class Decode_conv_bn_x1(nn.Module):
    def __init__(self, in_, out):
        super(Decode_conv_bn_x1,self).__init__()
        batchNorm_momentum = 0.1
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_, in_, kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(in_, momentum=batchNorm_momentum)
        self.conv2 = nn.Conv2d(in_,out, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)

        return x

class Decode_conv_bn_x2(nn.Module):
    def __init__(self, in_, out):
        super(Decode_conv_bn_x2,self).__init__()
        batchNorm_momentum = 0.1
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_, in_, kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(in_, momentum=batchNorm_momentum)

        self.conv2 = nn.Conv2d(in_,out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out, momentum=batchNorm_momentum)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class Decode_conv_bn_x3(nn.Module):
    def __init__(self, in_, out):
        super(Decode_conv_bn_x3,self).__init__()
        batchNorm_momentum = 0.1
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_, in_, kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(in_, momentum=batchNorm_momentum)

        self.conv2 = nn.Conv2d(in_,in_, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_, momentum=batchNorm_momentum)


        self.conv3 = nn.Conv2d(in_,out, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out, momentum=batchNorm_momentum)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x

class SegNet(nn.Module):
    def __init__(self, input_nbr, label_nbr):
        super(SegNet, self).__init__()
        batchNorm_momentum = 0.1
        
        self.encode1 = Encode_conv_bn_x2(input_nbr, 64)
        self.encode2 = Encode_conv_bn_x2(64, 128)
        self.encode3 = Encode_conv_bn_x3(128, 256)
        self.encode4 = Encode_conv_bn_x3(256, 512)
        self.encode5 = Encode_conv_bn_x3(512, 512)

        self.decode5 = Decode_conv_bn_x3(512, 512)
        self.decode4 = Decode_conv_bn_x3(512, 256)
        self.decode3 = Decode_conv_bn_x3(256, 128)
        self.decode2 = Decode_conv_bn_x2(128, 64)
        self.decode1 = Decode_conv_bn_x1(64, label_nbr)

    def forward(self, x):
        # Stage 1 Encode
        x1 = F.relu(self.encode1(x))
        self.x1p, self.id1 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)
        # Stage 2 Encode
        x2 = F.relu(self.encode2(self.x1p))
        self.x2p, self.id2 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)
        # Stage 3 Encode
        x3 = F.relu(self.encode3(self.x2p))
        self.x3p, self.id3 = F.max_pool2d(x3, kernel_size=2, stride=2,return_indices=True)
        # Stage 4 Encode
        x4 = F.relu(self.encode4(self.x3p))
        self.x4p, self.id4 = F.max_pool2d(x4, kernel_size=2, stride=2,return_indices=True)
        # Stage 5 Encode
        x5 = F.relu(self.encode5(self.x4p))
        self.x5p, self.id5 = F.max_pool2d(x5, kernel_size=2, stride=2,return_indices=True)

        # Stage 5 Decode
        x5 = F.max_unpool2d(self.x5p, self.id5, kernel_size=2, stride=2)
        x5 = F.relu(self.decode5(x5))

        # Stage 4 Decode
        x4 = F.max_unpool2d(x5, self.id4, kernel_size=2, stride=2)
        x4 = F.relu(self.decode4(x4))

        # Stage 3 Decode
        x3 = F.max_unpool2d(x4, self.id3, kernel_size=2, stride=2)
        x3 = F.relu(self.decode3(x3))

        # Stage 2 Decode
        x2 = F.max_unpool2d(x3, self.id2, kernel_size=2, stride=2)
        x2 = F.relu(self.decode2(x2))

        # Stage 5 Decode
        x1 = F.max_unpool2d(x2, self.id1, kernel_size=2, stride=2)
        x1 = F.softmax(self.decode1(x1), dim=1) # the dimension of softmax ?????

        return x1

segnet = SegNet(3,3)
use_cuda = torch.cuda.is_available()
print("Running GPU.") if use_cuda else print("No GPU available.")
if use_cuda:
    segnet.cuda()
# print(segnet)
print("******************************************")
print("******************************************")
print("******************************************")
print("******************************************")
print("******************************************")
summary(segnet, (3,512,512))