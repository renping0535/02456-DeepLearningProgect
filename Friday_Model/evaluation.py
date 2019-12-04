import torch
import os
import random
import glob
from PIL import Image
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
from SegNet import net
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sbn


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
    # print("{}th image:{} ".format(rand_int+1, np.unique(pred_img)))
    preds.append(pred_img)

# Utilities
def color_mapping(img):
    mapping_rev = {
    0:0, # no data
    76:1, # soil
    149:2, # crops
    225:3 # weeds
    }
    for item in mapping_rev:
        img[img==item] = mapping_rev[item]
    return img

annotations_path = '/home/renping/02456 DeepLearning Project/test/cropped/anno'
annotations_list = sorted(glob.glob(annotations_path+os.sep+'*.png'))

annos = []
for file in annotations_list:
    anno = np.array(Image.open(file).convert('L'))
    anno = color_mapping(anno)
    annos.append(anno)
annos = np.array(annos)
print('Annotations shape: {}'.format(annos.shape))


classes = ['No data', 'Soil', 'Crops', 'Weeds']
colors = [(1.0, 1.0, 0.8980392156862745, 1.0), # no data
         (0.7359477124183007, 0.8915032679738563, 0.5843137254901961, 1.0), # soil
         (0.21568627450980393, 0.6196078431372549, 0.330718954248366, 1.0), # crops
         (0.0, 0.27058823529411763, 0.1607843137254902, 1.0) # weeds
         ]
patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=classes[i])) for i in range(len(classes))]


fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ex1 = 16
ex2 = 10

# Anno ex 1
ax[0][0].imshow(annos[ex1], cmap=plt.cm.YlGn, vmin = 0, vmax = 3)
ax[0][0].set_title('Annotation example 1')

# Anno ex 2
ax[1][0].imshow(annos[ex2], cmap=plt.cm.YlGn, vmin = 0, vmax = 3)
ax[1][0].set_title('Annotation example 2')

# Pred ex 1
ax[0][1].imshow(preds[ex1], cmap=plt.cm.YlGn, vmin = 0, vmax = 3)
ax[0][1].set_title('Prediction example 1')

# Pred ex 2
ax[1][1].imshow(preds[ex2], cmap=plt.cm.YlGn, vmin = 0, vmax = 3)
ax[1][1].set_title('Prediction example 2')

fig.legend(handles=patches, loc='right')
# plt.show()

preds = np.array(preds)

cm_preds = np.reshape(preds, preds.shape[0]*preds.shape[1]*preds.shape[2])
cm_annos = np.reshape(annos, annos.shape[0]*annos.shape[1]*annos.shape[2])
from sklearn.utils.class_weight import compute_sample_weight
weights = compute_sample_weight(class_weight='balanced', y=cm_preds)
cm = confusion_matrix(cm_annos, cm_preds, sample_weight=weights)


plt.figure()
sbn.heatmap(cm, xticklabels=classes, yticklabels=classes) # annot=True prints the actual numbers in the heatmap matrix
plt.xlabel('Annotation')
plt.ylabel('Prediction')

plt.savefig("confusion matrix")
plt.show()

f1 = 0
for idx in range(len(annos)):
     for classes in np.unique(annos[idx]):
         if classes == 0:
             continue
     y_pred = (preds[idx]==classes)*1
     y_true = (annos[idx]==classes)*1
     f1 +=f1_score(y_true.reshape(-1), y_pred.reshape(-1), average='macro')

f1 /= len(annos)
print("F1 score: {}".format(f1))