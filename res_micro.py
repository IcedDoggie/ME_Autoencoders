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

# # menpo
# import menpo.io as mio
# from menpo.visualize import print_progress
# # from menpowidgets import visualize_images
# from menpo.landmark import labeller, face_ibug_68_to_face_ibug_68_trimesh
# from menpodetect import load_dlib_frontal_face_detector 
# from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional
# from menpofit.aam import PatchAAM, HolisticAAM
# from menpo.feature import fast_dsift, igo

from keras.models import Sequential, Model
from keras.utils import np_utils, plot_model
from keras import metrics
from keras import backend as K
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from keras.applications.vgg16 import VGG16 as keras_vgg16
from keras.preprocessing.image import ImageDataGenerator, array_to_img
# from keras.preprocessing.image_dev import ImageDataGenerator, array_to_img
import keras
from keras.callbacks import EarlyStopping
# from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import preprocess_input


from models import convolutional_autoencoder


def train():
	sys.setrecursionlimit(10000)
	root = '/home/ice/Documents/ME_Autoencoders/'
	root_db = '/media/ice/OS/Datasets/resnet_datasets/'
	labels_path = root_db + 'labels/'
	images_path = root_db + 'images/'
	aug_img_path = root_db + 'augmented_image/'
	classes = 6


	train_datagen = ImageDataGenerator(
		rescale = 1./255,
		# blurring = True,
		# color_shift = 20,
		preprocessing_function = preprocess_input			
		)	
	val_datagen = ImageDataGenerator(
		rescale = 1./255,

		)				
	# train_datagen.fit(images_list)
	# train_generator = train_datagen.flow_from_directory(images_path + 'train/', target_size=(224, 224), batch_size=1)

	adam = optimizers.Adam(lr=0.00001, decay=0.000001)
	stopping = EarlyStopping(monitor='loss', min_delta = 0, mode = 'min', patience=10)

	# model = ResNet50(weights='imagenet')
	# last_layer = model.get_layer("flatten_1").output
	# dense_classifier = Dense(classes, activation = 'softmax')(last_layer)
	# model = Model(inputs = model.input, outputs = dense_classifier)
	# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])

	conv_ae = convolutional_autoencoder(spatial_size = 224)
	conv_ae.compile(loss='kullback_leibler_divergence', optimizer = adam, metrics = [metrics.categorical_accuracy])
	plot_model(conv_ae, show_shapes=True, to_file='conv_ae.png')


	# for layer in model_res10.layers[:-1]:
	# 	layer.trainable = False	
	
	val_generator = val_datagen.flow_from_directory(
        root_db + 'test/',
        target_size=(224, 224),
        batch_size=40,)

	conv_ae.fit_generator(train_datagen.flow_from_directory(images_path, batch_size=40, target_size=(224, 224), class_mode='input'), steps_per_epoch=530, epochs=100, callbacks=[stopping])
	
	
	conv_ae.save_weights("Macro_Autoencoder.h5")
	# model.fit_generator(train_generator, epochs=1, steps_per_epoch=100)

	# plot_model(model, show_shapes=True)


train()