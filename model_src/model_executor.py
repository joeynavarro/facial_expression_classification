#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


from __future__ import print_function

import os
import csv
import h5py
import numpy as np
import skimage.io
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.autograd import Variable



#custom libraries
import model_backend.transformers as transforms
from model_backend import progress_bar
from model_backend.create_confusion_matrix import plot_confusion_matrix
from model_backend.create_train_data import CK
from built_models import *

from sklearn.metrics import confusion_matrix


# ## Data Extraction and Label Creation for CK+ Dataset

# ### Data Dictionary

# Expressions: 0 = anger 1 = disgust, 2 = fear, 3 = happy, 4 = sadness, 5 = surprise, 6 = contempt \
# Dataset holds: 135, 177, 75, 207, 84, 249, 54 images for each expression respectively
# 

# In[2]:


#this code will create an .h5 file that the model will call upon when searching for inputs
emotions = {'anger' : 'anger_path' , 'disgust' : 'disgust_path', 'fear' : 'fear_path', 
            'happy' : 'happy_path' , 'sadness': 'sadness_path', 'surprise' : 'surprise_path' , 
            'contempt' : 'contempt_path'}

# path to image directory
ck_path = '../data/ck+'


# instantiate lists and counter to store data and label information
data_x = []
data_y = []
count = 0

datapath = os.path.join('../data/ck+','ck_data.h5')
if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))

for emo_key, emo_val in emotions.items():
    emo_val = os.path.join(ck_path, emo_key)
    
    files = os.listdir(emo_val)
    files.sort()
    
    for filename in files:
        I = skimage.io.imread(os.path.join(emo_val, filename))
        data_x.append(I.tolist())
        data_y.append(count)
    count += 1
    
print(f'The pixel data shape is: {np.shape(data_x)}')
print(f'The label data shape is: {np.shape(data_y)}')  


#save the pixel and labels in .h5 file in the ck+ data folder for the model to call upon
datafile = h5py.File(datapath, 'w')
datafile.create_dataset("data_pixel", dtype = 'uint8', data=data_x)
datafile.create_dataset("data_label", dtype = 'int64', data=data_y)
datafile.close()

print("Oh happy day!, the image data has been compiled without a hitch!")


# ## Instantiating Command Line Interface Arguements 

# In[3]:


parser = argparse.ArgumentParser(description='PyTorch CK+ CNN Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='CK+', help='dataset')
parser.add_argument('--fold', default=1, type=int, help='k fold number')
parser.add_argument('--bs', default=128, type=int, help='batch_size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
print(parser)


# ## Training Function Definition

# In[4]:


# instantiating global lists to collect data for visualizations
# the lists are the reason I did not load in the functions from a seperate script

train_acc_list_vgg = []
train_loss_list_vgg = []

train_acc_list_rn = []
train_loss_list_rn = []

train_all_pred = []
train_all_targ = []

def train(epoch):
    print('This Is Training Epoch: %d' % epoch )
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    all_target = []

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        progress_bar.set_lr(optimizer, current_lr)  # set the decayed rate
        
    else:
        current_lr = opt.lr
   
    print(' ')    
    print('Learning Rate: %s' % str(current_lr))
    print(' ')

    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        # apply optimizer
        optimizer.zero_grad()
        
        outputs = net(inputs)
        
        # apply crossentropyloss
        loss = criterion(outputs, targets)
        loss.backward()
        progress_bar.clip_gradient(optimizer, 0.1)
        optimizer.step()

        train_loss += loss.item()
        
        # make prediction
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        #draw progress bar
        progress_bar.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Accuracy: %.3f%% (%d/%d)'
            % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        
        # append predicted and target global lists for visualizations
        if batch_idx == 0:
            train_all_pred.append(predicted)
            train_all_targ.append(targets)
        else:
            train_all_pred.append(torch.cat((train_all_pred[-1], predicted), 0))
            train_all_targ.append(torch.cat((train_all_targ[-1], targets), 0))
        
        print(' ')
        
        # appending accuracy and loss to global list for visualizations
        # not the test data is scaled according to cutsize for visualizations
        for index in range(cut_size):
            
            
            if opt.model == 'VGG19':
                loss_per_run = train_loss / (batch_idx + 1)
                train_loss_list_vgg.append(loss_per_run)

                acc_per_run = 100. * correct / total
                train_acc_list_vgg.append(acc_per_run)


            elif opt.model == 'Resnet18':
                loss_per_run = train_loss / (batch_idx + 1)
                train_loss_list_rn.append(loss_per_run)

                acc_per_run = 100. * correct / total
                train_acc_list_rn.append(acc_per_run)


    Train_acc = 100. * correct / total


# ## Testing Function Definition

# In[5]:


# instantiating global lists to collect data for visualizations
# the lists are the reason I did not load in the functions from a seperate script

test_acc_list_vgg = []
test_loss_list_vgg = []

test_acc_list_rn = []
test_loss_list_rn = []

test_all_pred = []
test_all_targ = []

res_best_test_acc = []
vgg_best_test_acc = []

def test(epoch):
    print('This Is Testing Epoch: %d' % epoch )
    global Test_acc
    global best_Test_acc
    global best_Test_acc_epoch
    net.eval()
    Testing_loss = 0
    correct = 0
    total = 0
    
    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        
        
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        # apply optimizer
        optimizer.zero_grad()
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        
        # apply crossentropyloss
        loss = criterion(outputs_avg, targets)
        Testing_loss += loss.item()
        
        # make prediction
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        #draw progress bar
        progress_bar.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Accuracy: %.3f%% (%d/%d)'
            % (Testing_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        
        # append predicted and target global lists for visualizations
        if batch_idx == 0:
            test_all_pred.append(predicted)
            test_all_targ.append(targets)
        else:
            test_all_pred.append(torch.cat((test_all_pred[-1], predicted), 0))
            test_all_targ.append(torch.cat((test_all_targ[-1], targets), 0))
        
        # appending accuracy and loss to global list for visualizations
        if opt.model == 'VGG19':
            loss_per_run = Testing_loss / (batch_idx+1)
            test_loss_list_vgg.append(loss_per_run)

            acc_per_run = 100. * correct / total
            test_acc_list_vgg.append(acc_per_run)


        elif opt.model == 'Resnet18':
            loss_per_run = Testing_loss / (batch_idx + 1 )
            test_loss_list_rn.append(loss_per_run)

            acc_per_run = 100.*correct / total
            test_acc_list_rn.append(acc_per_run)

        print(' ')
        
    # Save checkpoint.
    Test_acc = 100. * correct / total

    if Test_acc > best_Test_acc:
        print(' ')
        print('Awesome! Saving This Model..')
        print('Check This Out, The Best Test Accuracy So Far Is: %0.3f' % Test_acc + '%!!')
        state = {'net': net.state_dict() if use_cuda else net,
            'best_Test_acc': Test_acc,
            'best_Test_acc_epoch': epoch,
        }
        
        best_Test_acc = Test_acc
        best_Test_acc_epoch = epoch
        
        if not os.path.isdir('../model_checkpoints/' + opt.dataset + '_' + opt.model):
            os.mkdir('../model_checkpoints/' + opt.dataset + '_' + opt.model)
        if not os.path.isdir('../model_checkpoints/' + path):
            os.mkdir('../model_checkpoints/' + path)
        torch.save(state, os.path.join('../model_checkpoints/' + path, 'emoclass_model.t7'))
        
        
        if opt.model == 'VGG19':
            vgg_best_test_acc.append(best_Test_acc)
        elif opt.model == 'Resnet18':
            res_best_test_acc.append(best_Test_acc)


# ## Neural Network Parameters

# In[6]:



#Epoch choice
total_epoch = 12
start_epoch = 1  # start from epoch 1 or last checkpoint epoch

#Learning Rate Choice by run
learning_rate_decay_start = 0 
learning_rate_decay_every = 1
learning_rate_decay_rate = 0.8 

#basically batch size
cut_size = 43

#this model is built on a cuda PC, so if you have a mac or a non NVIDIA gpu, sorry not sorry.
use_cuda = torch.cuda.is_available()


# ## Command Line Arguements for VGG19 Convolutional Neural Network Model

# In[7]:


print('===> Reading Command Line Arguments')
opt = parser.parse_args('--model VGG19 --bs 128 --lr 0.01 --fold 10'.split())


# ## Data and Transformer Loader

# In[8]:


print('===> Loading Data Transformers for Augmentation...')
# define data transformers
transform_train = transforms.Compose([
    transforms.RandomCrop(cut_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

# model loaders use data transformers
print('===> Loading Data For Model...')
trainset = CK(split = 'Training', fold = opt.fold, transform = transform_train)
testset = CK(split = 'Testing', fold = opt.fold, transform = transform_test)

print('===> Preparing Data For Model...')

# Load trainning and testing data and apply parameters
trainloader = torch.utils.data.DataLoader(trainset, batch_size = opt.bs, shuffle = True, num_workers = 0)
testloader = torch.utils.data.DataLoader(testset, batch_size = 8, shuffle = False, num_workers=0)

print('===> Data Ready For Model Execution...')


# ## Model Loader and Executor

# In[9]:


#Count instantiators
best_Test_acc = 0  
best_Test_acc_epoch = 0

print('===> Loading Model Executor...')

# where to save best model
path = os.path.join( opt.dataset + '_' + opt.model)

# Load model
if opt.model == 'VGG19':
    net = VGG('VGG19')


# resume from best model if not started    
if opt.resume:
    # Load checkpoint.
    print('===> Continuing From Checkpoint...')
    assert os.path.isdir(path), 'ERROR: NO CHECKPOINT DIRECTORY FOUND!!!!'
    checkpoint = torch.load(os.path.join('../model_checkpoints/' + path,'emoclass_model.t7'))
    
    net.load_state_dict(checkpoint['net'])
    best_Test_acc = checkpoint['best_Test_acc']
    best_Test_acc_epoch = checkpoint['best_Test_acc_epoch']
    start_epoch = best_Test_acc_epoch + 1
    
else:
    print(' ')
    print('===> Building Model...')
    print(' ')

# initialize cuda!! Note, this is not an option it is a requirement.    
if use_cuda == True:
    net.cuda()

print('===> Preparing Optimizers For Model...')

# initialize loss and optimizer functions
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = opt.lr, momentum = 0.9, weight_decay = 5e-4)

print(' ')
print('===> Running Model...')
print(' ')

# start trainning and testing
for epoch in range(start_epoch, total_epoch + 1):
    train(epoch)
    print(' ')
    print('Epoch Trainning Done.')
    print(' ')
    test(epoch)
    print(' ')
    print('Epoch Testing Done')
    print(' ')
    
print('===> Calculating Confusion Matrix For Model...')
# Compute confusion matrix
vgg_matrix = confusion_matrix(test_all_targ[-1].data.cpu().numpy(), test_all_pred[-1].cpu().numpy())

# append for visualization in case no other appends have been made
if opt.model == 'VGG19':
    vgg_best_test_acc.append(best_Test_acc)
    
print("===> Best Test Accuracy: %0.3f" % best_Test_acc)
print("===> Best Test Accuracy Occured on Epoch: %d" % best_Test_acc_epoch)

# delete cuda memory cache to prevent memory errors
print('===> Clearing CUDA Memory Cache...')
del trainloader
torch.cuda.empty_cache()
del testloader
torch.cuda.empty_cache() 

print('===> Model Execution Complete...')


# ## Command Line Arguements for ResNet18 Convolutional Neural Network Model

# In[10]:


print('===> Reading Command Line Arguments')
opt = parser.parse_args('--model Resnet18 --bs 128 --lr 0.01 --fold 10'.split())


# ## Data and Transformer Loader

# In[11]:



print('===> Loading Data Transformers for Augmentation...')
# define data transformers
transform_train = transforms.Compose([
    transforms.RandomCrop(cut_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

# model loaders use data transformers
print('===> Loading Data For Model...')
trainset = CK(split = 'Training', fold = opt.fold, transform = transform_train)
testset = CK(split = 'Testing', fold = opt.fold, transform = transform_test)


print('===> Preparing Data For Model...')

# Load trainning and testing data and apply parameters
trainloader = torch.utils.data.DataLoader(trainset, batch_size = opt.bs, shuffle = True, num_workers = 0)
testloader = torch.utils.data.DataLoader(testset, batch_size = 8, shuffle = False, num_workers=0)

print('===> Data Ready For Model Execution...')


# ## Model Loader and Executor

# In[12]:


#Count instantiators
best_Test_acc = 0  
best_Test_acc_epoch = 0

print('===> Loading Model Executor...')

# where to save best model
path = os.path.join( opt.dataset + '_' + opt.model)

# Load model
if opt.model == 'Resnet18':
    net = ResNet18()
    
# resume from best model if not started    
if opt.resume:
    # Load checkpoint.
    print(' ')
    print('===> Continuing From Checkpoint...')
    print(' ')
    
    assert os.path.isdir(path), 'ERROR: NO CHECKPOINT DIRECTORY FOUND!!!!'
    checkpoint = torch.load(os.path.join('../model_checkpoints/' + path,'emoclass_model.t7'))
    
    net.load_state_dict(checkpoint['net'])
    best_Test_acc = checkpoint['best_Test_acc']
    best_Test_acc_epoch = checkpoint['best_Test_acc_epoch']
    start_epoch = best_Test_acc_epoch + 1
    
else:
    print(' ')
    print('===> Building Model...')
    print(' ')

# initialize cuda!! Note, this is not an option it is a requirement.
if use_cuda == True:
    net.cuda()

print('===> Preparing Optimizers For Model...')
# initialize loss and optimizer functions
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = opt.lr, momentum = 0.9, weight_decay = 5e-4)

print(' ')
print('===> Running Model...')
print(' ')

# start trainning and testing
for epoch in range(start_epoch, total_epoch + 1):
    train(epoch)
    print(' ')
    print('Epoch Trainning Done.')
    print(' ')
    test(epoch)
    print(' ')
    print('Epoch Testing Done.')
    print(' ')

print('===> Calculating Confusion Matrix For Model...')
# Compute confusion matrix
res_matrix = confusion_matrix(test_all_targ[-1].data.cpu().numpy(), test_all_pred[-1].cpu().numpy())

# append for visualization in case no other appends have been made
if opt.model == 'Resnet18':
    res_best_test_acc.append(best_Test_acc)

print("===> Best Test Accuracy: %0.3f" % best_Test_acc)
print("===> Best Test Accuracy Occured on Epoch: %d" % best_Test_acc_epoch)

# delete cuda memory cache to prevent memory errors
print('===> Clearing CUDA Memory Cache...')
del trainloader
torch.cuda.empty_cache()
del testloader
torch.cuda.empty_cache() 

print('===> Model Execution Complete...')


# ## Visualizations

# In[13]:


# Plot normalized confusion matrix
print('===> Creating VGG19 Confusion Matrix')
plt.figure(figsize=(10, 8))
plot_confusion_matrix(vgg_matrix, cmap = plt.cm.Reds, normalize=True,
                      title= 'VGG19 Convolutional Neural Network Model \n Normalized Confusion Matrix (Model Accuracy: %0.3f%%)' % max(vgg_best_test_acc))

plt.savefig('../model_visualizations/vgg19_model_confusion_matrix.png')
plt.show()


# In[14]:


# Plot normalized confusion matrix
print('===> Creating ResNet18 Confusion Matrix')
plt.figure(figsize=(10, 8))
plot_confusion_matrix(res_matrix, cmap = plt.cm.Greens, normalize=True,
                      title= 'ResNet18 Convolutional Neural Network Model \n Normalized Confusion Matrix (Model Accuracy: %0.3f%%)' % max(res_best_test_acc))

plt.savefig('../model_visualizations/resnet18_model_confusion_matrix.png')
plt.show()


# ## Loss Model Plots

# In[15]:


print('===> Creating Loss Plot')
fig, ax = plt.subplots(2, 2, figsize = (28, 10))
fig.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = None, hspace = 0.8)



#first plot
tral_vgg, = ax[0, 0].plot( train_loss_list_vgg)

tst_vgg_tr, = ax[0, 0].plot(test_loss_list_vgg[:6709], c = 'red' )


ax[0, 0].xaxis.set_major_locator(MultipleLocator(516))
ax[0, 0].yaxis.set_major_locator(MultipleLocator(0.5))
ax[0, 0].grid(which='major', color='#CCCCCC', linestyle='--')
ax[0, 0].set_xticklabels([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], fontsize = 16)
ax[0, 0].set_yticklabels([ '-0.5','0', '0.5', '1.0', '1.5', '2.0', '2.5','3.0'],fontsize=16)

ax[0, 0].set_title('VGG19 Model Loss On CK+ Data For 12 Epochs', size = 30, pad = 15, fontname = 'Gill Sans MT')
ax[0, 0].legend([tral_vgg, tst_vgg_tr], ['Training',  'Testing'], prop={'size': 16} )
ax[0, 0].set_xlabel('Epoch', size = 25, labelpad = 12, fontname = 'Gill Sans MT')
ax[0, 0].set_ylabel('Loss Rate', size = 25, labelpad = 5, fontname = 'Gill Sans MT')


for tick in ax[0, 0].get_xticklabels():
    tick.set_fontname("Gill Sans MT")

for tick in ax[0, 0].get_yticklabels():
    tick.set_fontname("Gill Sans MT")
    
#second plot

tral_rn, = ax[0, 1].plot( train_loss_list_rn, c = 'olivedrab')

tst_rn_tr, = ax[0, 1].plot(test_loss_list_rn[:6709], c = 'mediumorchid')

ax[0, 1].xaxis.set_major_locator(MultipleLocator(516))
ax[0, 1].yaxis.set_major_locator(MultipleLocator(0.5))
ax[0, 1].grid(which='major', color='#CCCCCC', linestyle='--')
ax[0, 1].set_xticklabels([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], fontsize = 16)
ax[0, 1].set_yticklabels([ '-0.5','0', '0.5', '1.0', '1.5', '2.0', '2.5','3.0', '3.5'],fontsize=16) 

ax[0, 1].set_title('ResNet18 Model Loss On CK+ for 12 Epochs', size = 30, pad = 15, fontname = 'Gill Sans MT')
ax[0, 1].legend([tral_rn, tst_rn_tr], ['Training', 'Testing'], prop={'size': 16} )
ax[0, 1].set_xlabel('Epoch', size = 25, labelpad = 12, fontname = 'Gill Sans MT')
ax[0, 1].set_ylabel('Loss Rate', size = 25, labelpad = 5, fontname = 'Gill Sans MT')


for tick in ax[0, 1].get_xticklabels():
    tick.set_fontname("Gill Sans MT")

for tick in ax[0, 1].get_yticklabels():
    tick.set_fontname("Gill Sans MT")

    
#third plot
tral_vgg, = ax[1, 0].plot( train_loss_list_vgg)
tral_rn, = ax[1, 0].plot( train_loss_list_rn, c = 'olivedrab')



ax[1, 0].xaxis.set_major_locator(MultipleLocator(516))
ax[1, 0].yaxis.set_major_locator(MultipleLocator(0.5))
ax[1, 0].grid(which='major', color='#CCCCCC', linestyle='--')
ax[1, 0].set_xticklabels([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], fontsize = 16)
ax[1, 0].set_yticklabels([ '-0.5','0', '0.5', '1.0', '1.5', '2.0', '2.5','3.0'],fontsize=16)

ax[1, 0].set_title('Model Loss on CK+ Training Data For 12 Epochs \n [Lowest Loss]  VGG19: %0.3f | ResNet18: %0.3f' % ( min(train_loss_list_vgg),  min(train_loss_list_rn)), size = 30, pad = 15, fontname = 'Gill Sans MT')
ax[1, 0].legend([tral_vgg, tral_rn], ['VGG19',  'ResNet18'], prop={'size': 16} )
ax[1, 0].set_xlabel('Epoch', size = 25, labelpad = 12, fontname = 'Gill Sans MT')
ax[1, 0].set_ylabel('Loss Rate', size = 25, labelpad = 5, fontname = 'Gill Sans MT')


for tick in ax[1, 0].get_xticklabels():
    tick.set_fontname("Gill Sans MT")

for tick in ax[1, 0].get_yticklabels():
    tick.set_fontname("Gill Sans MT")
    
    
#fourth plot
vgg_test, = ax[1, 1].plot(test_loss_list_vgg[:6709], c = 'red' )
rn_test, =ax[1, 1].plot(test_loss_list_rn[:6709], c = 'mediumorchid')


ax[1, 1].xaxis.set_major_locator(MultipleLocator(516))
ax[1, 1].yaxis.set_major_locator(MultipleLocator(0.5))
ax[1, 1].grid(which='major', color='#CCCCCC', linestyle='--')
ax[1, 1].set_yticklabels([ '-0.5','0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5'],fontsize=16)
ax[1, 1].set_xticklabels([ -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                     10, 11, 12, 13], fontsize = 16)

ax[1, 1].set_title('Model Loss on CK+ Testing Data For 12 Epochs  \n [Lowest Loss]  VGG19: %0.3f | ResNet18: %0.3f' % ( min(test_loss_list_vgg),  min(test_loss_list_rn)), size = 30, pad = 15, fontname = 'Gill Sans MT')
ax[1, 1].legend([vgg_test, rn_test], ['VGG19', 'ResNet18'], prop={'size': 16})

ax[1, 1].set_xlabel('Epoch', size = 25, labelpad = 12, fontname = 'Gill Sans MT')
ax[1, 1].set_ylabel('Loss Rate', size = 25, labelpad = 5, fontname = 'Gill Sans MT')

for tick in ax[1, 1].get_xticklabels():
    tick.set_fontname("Gill Sans MT")

for tick in ax[1, 1].get_yticklabels():
    tick.set_fontname("Gill Sans MT")
    
    
plt.savefig('../model_visualizations/model_loss_viz.png');


# ## Accuracy Model Plots

# In[16]:


print('===> Creating Accuracy Plot')
fig, ax = plt.subplots(2, 2, figsize = (28, 10))
fig.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = None, hspace = 0.8)



#first plot
tral_vgg, = ax[0, 0].plot(train_acc_list_vgg, c = 'mediumpurple')

tst_vgg_tr, = ax[0, 0].plot(test_acc_list_vgg[:6709], c = 'olive' )


ax[0, 0].xaxis.set_major_locator(MultipleLocator(516))
ax[0, 0].yaxis.set_major_locator(MultipleLocator(10))
ax[0, 0].grid(which='major', color='#CCCCCC', linestyle='--')
ax[0, 0].set_xticklabels([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                     10, 11, 12, 13], fontsize = 16)
ax[0, 0].set_yticklabels(['-1' ,'0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'],fontsize=16)
ax[0, 0].set_title('VGG19 Model Accuracy For 12 Epochs', size = 30, pad = 15, fontname = 'Gill Sans MT')
ax[0, 0].legend([tral_vgg, tst_vgg_tr], ['Training','Testing'], prop={'size': 16} )
ax[0, 0].set_xlabel('Epoch', size = 25, labelpad = 12, fontname = 'Gill Sans MT')
ax[0, 0].set_ylabel('Accuracy in %', size = 25, labelpad = 5, fontname = 'Gill Sans MT')


for tick in ax[0, 0].get_xticklabels():
    tick.set_fontname("Gill Sans MT")

for tick in ax[0, 0].get_yticklabels():
    tick.set_fontname("Gill Sans MT")
    
#second plot

tral_rn, = ax[0, 1].plot( train_acc_list_rn, c = 'deeppink')

tst_rn_tr, = ax[0, 1].plot(test_acc_list_rn[:6709], c = 'cadetblue')

ax[0, 1].xaxis.set_major_locator(MultipleLocator(516))
ax[0, 1].yaxis.set_major_locator(MultipleLocator(10))
ax[0, 1].grid(which='major', color='#CCCCCC', linestyle='--')
ax[0, 1].set_xticklabels([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                     10, 11, 12, 13], fontsize = 16)
ax[0, 1].set_yticklabels(['-1' ,'0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'],fontsize=16)
ax[0, 1].set_title('ResNet18 Model Accuracy For 12 Epochs', size = 30, pad = 15, fontname = 'Gill Sans MT')
ax[0, 1].legend([tral_rn, tst_rn_tr], ['Training','Testing'], prop={'size': 16} )
ax[0, 1].set_xlabel('Epoch', size = 25, labelpad = 12, fontname = 'Gill Sans MT')
ax[0, 1].set_ylabel('Accuracy in %', size = 25, labelpad = 5, fontname = 'Gill Sans MT')


for tick in ax[0, 1].get_xticklabels():
    tick.set_fontname("Gill Sans MT")

for tick in ax[0, 1].get_yticklabels():
    tick.set_fontname("Gill Sans MT")

# third plot 
tral_vgg, = ax[1, 0].plot(train_acc_list_vgg, c = 'mediumpurple')

tral_rn, = ax[1, 0].plot( train_acc_list_rn, c = 'deeppink')


ax[1, 0].xaxis.set_major_locator(MultipleLocator(516))
ax[1, 0].yaxis.set_major_locator(MultipleLocator(10))
ax[1, 0].grid(which='major', color='#CCCCCC', linestyle='--')
ax[1, 0].set_xticklabels([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                     10, 11, 12, 13], fontsize = 16)
ax[1, 0].set_yticklabels([ '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'],fontsize=16)
ax[1, 0].set_title('Model Accuracy on Trainning Data For 12 Epochs \n [Best Accuracies]  VGG19: %0.3f%% | ResNet18: %0.3f%%' % ( max(train_acc_list_vgg),  max(train_acc_list_rn)), size = 30, pad = 15, fontname = 'Gill Sans MT')
ax[1, 0].legend([tral_vgg, tral_rn], ['VGG19', 'ResNet18'], prop={'size': 16} )
ax[1, 0].set_xlabel('Epoch', size = 25, labelpad = 12, fontname = 'Gill Sans MT')
ax[1, 0].set_ylabel('Accuracy in %', size = 25, labelpad = 5, fontname = 'Gill Sans MT')


for tick in ax[1, 0].get_xticklabels():
    tick.set_fontname("Gill Sans MT")

for tick in ax[1, 0].get_yticklabels():
    tick.set_fontname("Gill Sans MT")
  

    
#fourth plot
vgg_test, = ax[1, 1].plot(test_acc_list_vgg[:6709], c = 'olive' )
rn_test, =ax[1, 1].plot(test_acc_list_rn[:6709], c = 'cadetblue')


ax[1, 1].xaxis.set_major_locator(MultipleLocator(516))
ax[1, 1].yaxis.set_major_locator(MultipleLocator(10))
ax[1, 1].grid(which='major', color='#CCCCCC', linestyle='--')
ax[1, 1].set_yticklabels(['-1', '0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'],fontsize=16)
ax[1, 1].set_xticklabels([  -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
                     10, 11, 12, 13], fontsize = 16)

ax[1, 1].set_title('Model Accuracy on Test Data For 12 Epochs  \n [Best Accuracies]  VGG19: %0.3f%% | ResNet18: %0.3f%%' % ( max(vgg_best_test_acc),  max(res_best_test_acc)), size = 30, pad = 15, fontname = 'Gill Sans MT')
ax[1, 1].legend([vgg_test, rn_test], ['VGG19', 'ResNet18'], prop={'size': 16})

ax[1, 1].set_xlabel('Epoch', size = 25, labelpad = 12, fontname = 'Gill Sans MT')
ax[1, 1].set_ylabel('Accuracy in %', size = 25, labelpad = 5, fontname = 'Gill Sans MT')

for tick in ax[1, 1].get_xticklabels():
    tick.set_fontname("Gill Sans MT")

for tick in ax[1, 1].get_yticklabels():
    tick.set_fontname("Gill Sans MT")
    
plt.savefig('../model_visualizations/model_accuracy_viz.png');
    


# In[17]:


print('===> Done Running Model Executor')

