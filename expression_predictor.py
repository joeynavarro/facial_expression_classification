#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from os.path import dirname, join
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from skimage import io
from skimage.transform import resize

#custom imports
from model_src.model_backend.transformers import  transforms
from model_src.built_models import *


# In[2]:


#basically batch size
cut_size = 43

print('===> Loading Data Transformers for Augmentation...')

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])


# In[3]:


print('===> Loading Color To Grayscale Function...')
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# In[4]:


directory = './predictor_images_to_predict/'
counter = 0
for filename in os.listdir(directory):
    
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        
        print('===> Reading Image ' + str(counter) + '...')
        raw_img = io.imread(os.path.join(directory, filename))

        print('===> Converting Image to Grayscale...')
        gray = rgb2gray(raw_img)
        gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)

        img = gray[:, :, np.newaxis]

        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        
        print('===> Creating Image Augmentation Transformations...')
        inputs = transform_test(img)

        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        print('===> Loading Model...')
        net = VGG('VGG19')
        checkpoint = torch.load(os.path.join('./model_checkpoints/CK+_VGG19/', 'emoclass_model.t7'))
        net.load_state_dict(checkpoint['net'])
        net.cuda()
        net.eval()

        ncrops, c, h, w = np.shape(inputs)

        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.cuda()

        outputs = net(inputs)

        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(outputs_avg, -1)
        _, predicted = torch.max(outputs_avg.data, 0)

        print('===> Recognizing Face...')
        # Facial Recognition
        dir_path = './model_checkpoints/predictor_weights'
        
        protoPath = join(dir_path, "deploy.prototxt")
        modelPath = join(dir_path, 'weights.caffemodel')

        model = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


        
        #read the image using cv2
        face_image = cv2.imread(os.path.join(directory, filename))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        #accessing the image.shape tuple and taking the elements
        (h,w) = face_image.shape[:2]
        
        #get our blog which is our input image
        blob = cv2.dnn.blobFromImage(cv2.resize(face_image, (300, 300)), 1.0, (300, 300), (104.0 , 177.0, 123.0))
        
        #input the blob into the model and get back the detections
        model.setInput(blob)
        detections = model.forward()
        
        #Iterate over all of the faces detected and extract their start and end points
        count = 0
        for i in range(0, detections.shape[2]):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            
            confidence = detections[0, 0, i, 2]
            
            #if the algorithm is more than 16.5% confident that the detection is a face, show a rectangle around it
            if (confidence > 0.98):
                cv2.rectangle(face_image, (startX, startY), (endX, endY), (0, 255, 0), 4)
                count = count + 1
             
        print('===> Predicting and Plotting...')
        # instantiate plot
        fig, ax = plt.subplots(2, 2, figsize = (17, 13))
        
        
        #fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace= 0.3, hspace=0.3)
        
        
        # plot quadrant I
        ax[0, 1].set_title('Image Face Detection', fontsize = 25, pad = 20, style = 'oblique',  fontname = 'Verdana')
        ax[0, 1].imshow(face_image)
        ax[0, 1].set_xticks([])
        ax[0, 1].set_yticks([])
        
        
        
        # plot quadrant II
        ax[0, 0].imshow(raw_img)
        ax[0, 0].set_title('Original Image', fontsize = 25, pad = 20, style = 'oblique',  fontname = 'Verdana')
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])
    


        # plot quadrant III
        ind = 0.1 + 0.6 * np.arange(len(class_names))    # the x locations for the groups
        width = 0.4       # the width of the bars: can also be len(x) sequence
        
        color_list = ['red','olive','darkslategray','yellow','teal','springgreen','lightgray']
        
        for i in range(len(class_names)):
             ax[1, 0].bar(ind[i], (score * 100).data.cpu().numpy()[i], width, color=color_list[i])

        ax[1, 0].set_title("Expression Prediction Scores ",fontsize = 25, pad = 20, style = 'oblique',  fontname = 'Verdana')
        ax[1, 0].set_xlabel("Expression Type",fontsize = 16, labelpad = 10, style = 'oblique',  fontname = 'Verdana')
        ax[1, 0].set_ylabel("Prediction Confidence in %",fontsize = 16, labelpad = 10,  fontname = 'Verdana', style = 'oblique')
        ax[1, 0].set_xticks(ind)
        ax[1, 0].set_xticklabels( class_names, fontsize = 13,  fontname = 'Verdana') 
        
        # plot quadrant IV
        # emoji read in and transformations
        im = Image.open('./data/model_emojis/%s.png' % str(class_names[int(predicted.cpu().numpy())]))
        
        # expand emoji so it's not pixalated
        
        desired_size = 500
        old_size = im.size  # old_size[0] is in (width, height) format

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        
        # use thumbnail() or resize() method to resize the input image
        im = im.resize(new_size, Image.ANTIALIAS)
        
        # create a new image and paste the resized on it
        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, ((desired_size-new_size[0]) // 2,
                    (desired_size-new_size[1]) // 2))
        delta_w = desired_size - new_size[0]
        delta_h = desired_size - new_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w-(delta_w // 2), delta_h-(delta_h // 2))
        new_emoji = ImageOps.expand(im, padding)
        
        ax[1, 1].imshow(new_emoji)
        ax[1, 1].set_title('Emoji Expression', fontsize=25, pad = 20, style = 'oblique',  fontname = 'Verdana')
        ax[1, 1].set_xlabel(" %s Face is Being Expressed" %str(class_names[int(predicted.cpu().numpy())]),fontsize = 30, labelpad = 20,  fontname = 'Verdana', fontweight = 'bold')
        ax[1, 1].set_xticks([])
        ax[1, 1].set_yticks([])
        
        fig.tight_layout(h_pad= 5.5, pad=5)

#         plt.show()
        print('===> Saving Prediction Figure...')
        plt.savefig(os.path.join('./predictor_images_predicted/', filename))
        plt.close()
        print('===> Finished Predicting Image ' + str(counter) + '!!!')
        print(' ')
        counter += 1

        
print('===> Finished Predicting All Images In Directory')    

