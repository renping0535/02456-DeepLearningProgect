import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from SegNet import net

from batch_generator_train import train_loader
from batch_generator_test import test_loader

# mapping = {
#     0: 0, # no data
#     76: 1, # soil
#     149: 2, # crops
#     225: 3 # weeds
# }



# initialize the net, loss, optimizer
use_cuda = torch.cuda.is_available()
# use_cuda = False
if use_cuda:
    print("Running on GPU!")
    net = net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
else:
    print("Running on CPU!")
    net = net
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.0001)
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)


num_epoch = 100
print("Start training!")
print("Total training epoch: {}.".format(num_epoch))

total_loss = []
total_accuracy = []

total_loss_test = []
total_accuracy_test = []

for epoch in tqdm(range(0, num_epoch)):
    
    # # train
    # total_train = 0.0 # number of pixels
    # current_train = 0.0 # number of correct pixels
    # train_accuracy = 0.0
    
    # # test
    # total_test = 0.0 # number of pixels
    # current_test = 0.0 # number of correct pixels
    # test_accuracy = 0.0

    train_epoch_loss = 0.0
    net.train()
    for i, train_data in enumerate(train_loader):
        inputs, targets = train_data
        inputs = inputs.permute(0,3,1,2)
        inputs = inputs.type(torch.FloatTensor)
        targets = targets.type(torch.LongTensor)
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        # print("inpusts size = {}".format(inputs.size()))
        # print("targets size = {}".format(targets.size()))
        optimizer.zero_grad()
        # forward
        outputs = net.forward(inputs)
        # print("outputs size = {}".format(outputs.size()))
        loss = criterion(outputs, targets)

        # backward
        loss.backward()
        optimizer.step()

        train_epoch_loss = train_epoch_loss + loss.item()
        
        # # train accuracy
        # pred = torch.argmax(outputs, dim=1)
        # total_train += targets.nelement()
        # current_train += pred.eq(targets).sum().item()
        # train_accuracy += 100*current_train/total_train
  
    total_loss.append(train_epoch_loss/88)

    test_epoch_loss = 0.0
    net.eval()
    for j, test_data in enumerate(test_loader):
        test_input, test_target, _, _, _, _ = test_data
        test_input = test_input.type(torch.FloatTensor)
        test_target = test_target.type(torch.LongTensor)
        if use_cuda:
            test_input = test_input.cuda()
            test_target = test_target.cuda()
        with torch.no_grad():
            test_output = net.forward(test_input)
            loss_test = criterion(test_output, test_target)
        
        test_epoch_loss = test_epoch_loss + loss_test.item()
        
        # # test accuracy
        # test_pred = torch.argmax(test_output, dim=1)
        # total_test += test_target.nelement()
        # current_test += test_pred.eq(test_target).sum().item()
        # test_accuracy += 100*current_test/total_test

    total_loss_test.append(test_epoch_loss/35)

    # print train epoch loss every epoch
    print("epoch {}/{}, train loss {}".format(epoch+1, num_epoch, train_epoch_loss/88))

    # print test epoch loss every epoch
    print("epoch {}/{}, test loss {}".format(epoch+1, num_epoch, test_epoch_loss/35))

    # # print train accuracy statics every epoch
    # epoch_average_acc = train_accuracy/len(train_loader)
    # total_accuracy.append(epoch_average_acc)
    # print("epoch {}, average train accuracy {}".format(epoch+1, epoch_average_acc))
    
    # # print test accuracy statics every epoch
    # epoch_average_acc_test = test_accuracy/len(test_loader)
    # total_accuracy_test.append(epoch_average_acc_test)
    # print("epoch {}, average test accuracy {}".format(epoch+1, epoch_average_acc_test))


torch.save(net.state_dict(), '/home/renping/02456 DeepLearning Project/net_trained/net_trained.pt')

print("Trained Model Saved")
print("Training Finished")



plt.figure()
plt.title("loss")
plt.plot(total_loss, label="train loss")
plt.plot(total_loss_test, label= "test loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(loc=1)
plt.grid()


plt.savefig("loss")
plt.show()