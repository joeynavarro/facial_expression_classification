from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data
import random
from numpy import random

'''
Warning!!!
This class only works for the ck+ dataset.

'''
class CK(data.Dataset):
    """ this class if for the ck+ dataset specifically.
    Arguments in fuction:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``

        Details for what this class does with the classified images:
        there are 135,177,75,207,84,249,54 images in data for each expression specifically
        class chooses 123, 159 , 66, 186, 75, 225, 48 images for training for each classified expression specifically
        class chooses 12, 8, 9, 21, 9, 24, 6 images for testing for each classified expression specifically
    """

    def __init__(self, split = 'Training', fold = 1, transform = None):
        self.transform = transform
        self.split = split  # specifies the training set or test set
        self.fold = fold # the used k-fold for cross validation
        self.data = h5py.File('../data/ck+/CK_data.h5', 'r', driver = 'core')

        number = len(self.data['data_label'])
        sum_number = [0, 135, 312, 387, 594, 678, 927, 981] # the sum of the classified images
        test_number = [135, 177, 75, 207, 84, 249, 54] # the number of images for each classification

        #Train test split
        test_index = []
        train_index = []

        #use all of the data available for testing
        for j in range(len(test_number)):
            for k in range(test_number[j]):
                if self.fold != 10: #the last fold start from the last element
                    test_index.append(sum_number[j] + (self.fold-1) * test_number[j] + k)

                if self.fold == 10:
                    test_index.append(sum_number[j + 1] - 1 - k)

            #1/3 of the test index for testing
            for s in range(round(test_number[j] / 3)):
                if self.fold != 10: #the last fold start from the last element
                    train_index.append(sum_number[j] + (self.fold-1) * test_number[j] + s)

                if self.fold == 10:
                    train_index.append(sum_number[j + 1] - 1 - s)






        # loads the picked numpy arrays
        if self.split == 'Training':
            self.train_data = []
            self.train_labels = []
            print(' ')
            print(f'Train Dataset Samples: {len(train_index)}')

            #scaling data
            for index in range(5):
                for ind in range(len(train_index)):
                    #create training data
                    # selecting random data_pixel and data_label from entirety of data for length of train_index for training data
                    choice = random.randint(0, len(test_index))
                    self.train_data.append(self.data['data_pixel'][test_index[choice]])
                    self.train_labels.append(self.data['data_label'][test_index[choice]])

        elif self.split == 'Testing':
            self.test_data = []
            self.test_labels = []
            print(' ')
            print(f'Test Dataset Samples: {len(test_index)}')
            print(' ')

            #scaling data
            for index in range(5):
                #create testing data
                for ind in range(len(test_index)):
                    self.test_data.append(self.data['data_pixel'][test_index[ind]])
                    self.test_labels.append(self.data['data_label'][test_index[ind]])

    def __getitem__(self, index):
        """
        Arguments:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'Testing':
            img, target = self.test_data[index], self.test_labels[index]

        # the following code is so that data is consistent with all other datasets
        # to return a PIL Image

        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis = 2)
        img = Image.fromarray(img)


        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'Testing':
            return len(self.test_data)
