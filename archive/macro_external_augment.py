import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pydot, graphviz
from pathlib import Path
import json
import itertools
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import sys
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from evaluationmatrix import fpr, weighted_average_recall
from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import confusion_matrix
import scipy.io as sio

import keras
from keras.models import Sequential, Model
from keras.utils import np_utils, plot_model
from keras import metrics
from keras import backend as K
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img
from keras.callbacks import EarlyStopping
from keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from keras.applications.resnet50 import preprocess_input as res_preprocess_input

from utilities import LossHistory
from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall
from networks import train_vgg16_imagenet, train_inceptionv3_imagenet, train_res50_imagenet

from siamese_utilities import blurring, image_contrast, gaussian_noise

macro_imgs_path = '/media/ice/OS/Datasets/resnet_datasets/images/'
target = '/media/ice/OS/Datasets/resnet_datasets/test_images_augment/'

def augment_macro(macro_imgs_path):
	img_list = []
	label_list = []
	data_filenames = [] # for passing parameter usage

	for root, folders, files in os.walk(macro_imgs_path):
		if root[-1] != '/':
			temp_str = root.split('/')[-1]
			
			for file in files:

				path = root + '/' + file	
				img_list += [path]
				label_list += [temp_str]

	print(len(img_list))
	print(len(label_list))
	# blurring(img_list, label_list, data_filenames, target, mode='macro')			
	image_contrast(img_list, label_list, data_filenames, target, contrast_count = 4, mode='macro')
	# gaussian_noise(img_list, label_list, data_filenames, target, mode='macro')	
	
augment_macro(macro_imgs_path)
