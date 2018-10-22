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

def train(type_of_test, train_id):
	sys.setrecursionlimit(10000)
	root = '/home/viprlab/Documents/ME_Res/'
	root_db = '/media/viprlab/01D31FFEF66D5170/Ice/resnet_datasets/'
	labels_path = root_db + 'labels/'
	images_path = root_db + 'images/'
	aug_img_path = root_db + 'augmented_image/'
	classes = 6

	if 'res' in train_id:
		preprocess = res_preprocess_input
	else:
		preprocess = vgg_preprocess_input


	train_datagen = ImageDataGenerator(
		rescale = 1./255,
		preprocessing_function = preprocess
		)	
	val_datagen = ImageDataGenerator(
		rescale = 1./255,

		)				

	# training configuration
	learning_rate = 0.0001
	adam = optimizers.Adam(lr=learning_rate, decay=1e-7)
	sgd = optimizers.SGD(lr=learning_rate, decay=1e-7, momentum=0.9, nesterov=True)	
	stopping = EarlyStopping(monitor='loss', min_delta = 0, mode = 'min', patience=10)

	model = type_of_test(classes = classes)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
	plot_model(model, show_shapes=True)

	
	val_generator = val_datagen.flow_from_directory(
        root_db + 'test/',
        target_size=(224, 224),
        batch_size=40,)

	model.fit_generator(train_datagen.flow_from_directory(images_path, batch_size=40, target_size=(224, 224)), steps_per_epoch=530, epochs=100, validation_data = val_generator, validation_steps=100, callbacks=[stopping])	
	model.save_weights("res_micro_grayscale_" + train_id + "_retrain.h5")
	# model.fit_generator(train_generator, epochs=1, steps_per_epoch=100)

	# plot_model(model, show_shapes=True)

train(train_vgg16_imagenet, 'vgg16')
train(train_res50_imagenet, 'res50')
train(train_inceptionv3_imagenet, 'incepv3')