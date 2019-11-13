import torch.optim as optim
import torch
import torch.nn as nn
from SegNet_v1 import segnet
from batch_generator_v1 import data_iterator
# criterion and optimizer
criterion = nn.BCELoss().cuda()
# BCELoss accepts only inputs that have all elements in range [0; 1]
optimizer = optim.SGD(segnet.parameters(), lr = 0.001, momentum=0.9)

# train the neural network

num_epoch = 2

for epoch in range(1, num_epoch+1):
    train_loss = 0.0
    for i, data in enumerate(data_iterator, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda().float()

        optimizer.zero_grad()
        outputs = segnet.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()



print("Finished Training!!")