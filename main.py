import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import time

from model import *
from config import DefaultConfig

######################################
# Initializing
######################################

opt = DefaultConfig()
 
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                          shuffle=True, num_workers=opt.num_workers)
 
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size,
                                         shuffle=False, num_workers=opt.num_workers)
 
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''
 Remember to update this!
'''
net = resnet18()
if opt.use_gpu == True:
    net.cuda()

########################################
# Training
########################################

print('Start training')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr,
             momentum=opt.momentum, weight_decay=opt.weight_decay)

for epoch in range(opt.max_epoch):
 
    running_loss = 0.

    batch_size = opt.batch_size
    
    for i, data in enumerate(
            torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=opt.num_workers), 0):
        
        inputs, labels = data
        if opt.use_gpu == True:
            inputs, labels = inputs.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if i % opt.print_freq == 0:
	        print('[%d, %5d] loss: %.4f' %(epoch + 1, i, loss.item()))
 
print('Finished Training')

########################################
# Save the model
########################################

prefix = './checkpoints/'
current_time = time.strftime("%m%d%H%M", time.localtime())
filename = prefix + current_time + '.pth'
torch.save(net, filename)

log = open('./log.txt', 'a')
message = input('INPUT LOG MESSAGE:\n')
print('\n\ntest ' + current_time, file = log)
print(message, file = log)

########################################
# Testing
########################################

net = torch.load(filename)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if opt.use_gpu == True:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total), file = log)
 
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if opt.use_gpu == True:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
 
 
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]),
        file = log)