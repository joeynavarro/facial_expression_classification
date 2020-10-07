
# confusion matrix plot function
import os
import sys
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from .create_train_data import CK
from sklearn.metrics import confusion_matrix
from .transformers import transforms as transforms
sys.path.append(os.path.realpath('.'))
from built_models import *



def plot_confusion_matrix(cm, cmap,
                          normalize=False,
                          title='Confusion matrix'
                          ):

    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Contempt']
    # This function prints and plots the confusion matrix.
    if normalize:
        cm = (cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]) * 100
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize = 20, pad =20, fontname = 'Gill Sans MT')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontname = 'Gill Sans MT', fontsize = 13)
    plt.yticks(tick_marks, classes, fontname = 'Gill Sans MT', fontsize = 13)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color = "white" if cm[i, j] > thresh else "black")


    plt.ylabel('True Label', fontsize = 18, labelpad = 10, fontname = 'Gill Sans MT')
    plt.xlabel('Predicted Label', fontsize = 18, labelpad = 10, fontname = 'Gill Sans MT')
    plt.tight_layout()
