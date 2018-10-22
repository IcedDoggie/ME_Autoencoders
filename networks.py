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

from keras.models import Sequential, Model
from keras.utils import np_utils, plot_model, Sequence
from keras import metrics
from keras import optimizers
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import backend as K
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping
from keras.layers import GlobalAveragePooling2D

from utilities import loading_smic_table, loading_samm_table, loading_casme_table
from utilities import class_merging, read_image, create_generator_LOSO
from utilities import LossHistory, record_loss_accuracy
from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall
from models import VGG_16, temporal_module, layer_wise_conv_autoencoder, layer_wise_autoencoder, convolutional_autoencoder



def test_res50_imagenet(weights_name = 'imagenet'):
	resnet50 = ResNet50(weights = 'imagenet')
	resnet50 = Model(inputs = resnet50.input, outputs = resnet50.layers[-2].output)
	plot_model(resnet50, to_file='resnet50.png', show_shapes=True)

	return resnet50

def test_vgg16_imagenet(weights_name = 'imagenet'):
	vgg16 = VGG16(weights = 'imagenet')
	vgg16 = Model(inputs = vgg16.input, outputs = vgg16.layers[-2].output)
	plot_model(vgg16, to_file='vgg16.png', show_shapes=True)

	return vgg16
	
def test_inceptionv3_imagenet(weights_name = 'imagenet'):
	inceptionv3 = InceptionV3(weights = 'imagenet')
	inceptionv3 = Model(inputs = inceptionv3.input, outputs = inceptionv3.layers[-2].output)	
	plot_model(inceptionv3, to_file='inceptionv3.png', show_shapes=True)

	return inceptionv3


def test_vgg19_imagenet():
	vgg19 = VGG19(weights = 'imagenet')
	vgg19 = Model(inputs = vgg19.input, outputs = vgg19.layers[-2].output)
	plot_model(vgg19, to_file='vgg19.png', show_shapes=True)

	return vgg19


def test_mobilenet_imagenet():
	mobilenet = MobileNet(weights = 'imagenet')
	mobilenet = Model(inputs = mobilenet.input, outputs = mobilenet.layers[-2].output)
	plot_model(mobilenet, to_file='mobilenet.png', show_shapes=True)

	return mobilenet

def test_xception_imagenet():
	xception = Xception(weights = 'imagenet')
	xception = Model(inputs = xception.input, outputs = xception.layers[-2].output)
	plot_model(xception, to_file='xception.png', show_shapes=True)	

	return xception

def test_inceptionResV2_imagenet():
	inceptionresnetv2 = InceptionResNetV2(weights = 'imagenet')
	inceptionresnetv2 = Model(inputs = inceptionresnetv2.input, outputs = inceptionresnetv2.layers[-2].output)
	plot_model(inceptionresnetv2, to_file='inceptionresnetv2.png', show_shapes=True)

	return inceptionresnetv2	


def test_res50_finetuned(weights_name = 'imagenet'):
	resnet50 = ResNet50(weights = 'imagenet')
	last_layer = resnet50.layers[-2].output
	dense_classifier = Dense(3, activation = 'softmax')(last_layer)
	resnet50 = Model(inputs = resnet50.input, outputs = dense_classifier)
	resnet50.load_weights(weights_name)
	resnet50 = Model(inputs = resnet50.input, outputs = resnet50.layers[-2].output)
	plot_model(resnet50, to_file='resnet50.png', show_shapes=True)

	return resnet50

def test_vgg16_finetuned(weights_name = 'imagenet'):
	vgg16 = VGG16(weights = 'imagenet')
	last_layer = vgg16.layers[-2].output
	dense_classifier = Dense(3, activation = 'softmax')(last_layer)
	vgg16 = Model(inputs = vgg16.input, outputs = dense_classifier)
	vgg16.load_weights(weights_name)
	vgg16 = Model(inputs = vgg16.input, outputs = vgg16.layers[-2].output)
	plot_model(vgg16, to_file='vgg16.png', show_shapes=True)

	return vgg16

def test_inceptionv3_finetuned(weights_name = 'imagenet'):
	inceptionv3 = InceptionV3(weights = 'imagenet')
	last_layer = inceptionv3.layers[-2].output
	dense_classifier = Dense(3, activation = 'softmax')(last_layer)
	inceptionv3 = Model(inputs = inceptionv3.input, outputs = dense_classifier)
	inceptionv3.load_weights(weights_name)
	inceptionv3 = Model(inputs = inceptionv3.input, outputs = inceptionv3.layers[-2].output)	
	plot_model(inceptionv3, to_file='inceptionv3.png', show_shapes=True)

	return inceptionv3


def train_res50_imagenet(classes = 5):
	resnet50 = ResNet50(weights = 'imagenet')

	# load macro
	last_layer = resnet50.layers[-2].output
	dense_classifier = Dense(6, activation = 'softmax')(last_layer)
	resnet50 = Model(inputs = resnet50.input, outputs = dense_classifier)		
	resnet50.load_weights('res_micro_grayscale_res50_retrain.h5')	

	last_layer = resnet50.layers[-2].output
	dense_classifier = Dense(classes, activation = 'softmax')(last_layer)
	resnet50 = Model(inputs = resnet50.input, outputs = dense_classifier)
	plot_model(resnet50, to_file='resnet50.png', show_shapes=True)

	# for layer in resnet50.layers:
	# 	layer.trainable = True

	# for 2nd last block
	for layer in resnet50.layers[:-25]:
		layer.trainable = False	

	# # for 3rd last block
	# for layer in resnet50.layers[:-37]:
	# 	layer.trainable = False		

	# for layer in resnet50.layers[:-14]:
	# 	layer.trainable = False	

	return resnet50

def train_vgg16_imagenet(classes = 5):
	vgg16 = VGG16(weights = 'imagenet')

	# # load macro
	# last_layer = vgg16.layers[-2].output
	# dense_classifier = Dense(6, activation = 'softmax')(last_layer)
	# vgg16 = Model(inputs = vgg16.input, outputs = dense_classifier)		
	# vgg16.load_weights('res_micro_grayscale_vgg16_retrain.h5')	

	# LFW weights
	last_layer = vgg16.layers[-2].output
	dense_classifier = Dense(2622, activation = 'softmax')(last_layer)
	vgg16 = Model(inputs = vgg16.input, outputs = dense_classifier)			
	vgg16.load_weights('VGG_Face_Deep_16.h5')

	last_layer = vgg16.layers[-2].output
	dense_classifier = Dense(classes, activation = 'softmax')(last_layer)
	vgg16 = Model(inputs = vgg16.input, outputs = dense_classifier)	
	plot_model(vgg16, to_file='vgg16.png', show_shapes=True)
		
	# for layer in vgg16.layers:
	# 	layer.trainable = True

	# train last 2 block
	for layer in vgg16.layers[:-8]:
		layer.trainable = False

	# # train last 3 block
	# for layer in vgg16.layers[:-9]:
	# 	layer.trainable = False

	# # train last block
	# for layer in vgg16.layers[:-7]:
	# 	layer.trainable = False	

	return vgg16

def train_inceptionv3_imagenet(classes = 5):
	inceptionv3 = InceptionV3(weights = 'imagenet')

	# load macro
	last_layer = inceptionv3.layers[-2].output
	dense_classifier = Dense(6, activation = 'softmax')(last_layer)
	inceptionv3 = Model(inputs = inceptionv3.input, outputs = dense_classifier)		
	inceptionv3.load_weights('res_micro_grayscale_incepv3_retrain.h5')

	last_layer = inceptionv3.layers[-2].output
	dense_classifier = Dense(classes, activation = 'softmax')(last_layer)
	inceptionv3 = Model(inputs = inceptionv3.input, outputs = dense_classifier)	
	plot_model(inceptionv3, to_file='inceptionv3.png', show_shapes=True)

	# for layer in inceptionv3.layers:
	# 	layer.trainable = True
	
	# 2nd last incep block
	for layer in inceptionv3.layers[:-85]:
		layer.trainable = False

	# # 3rd last incep block
	# for layer in inceptionv3.layers[:-117]:
	# 	layer.trainable = False

	# for layer in inceptionv3.layers[:-34]:
	# 	layer.trainable = False	
def train_xception_imagenet():
	xception = Xception(weights = 'imagenet')
	last_layer = xception.layers[-2].output
	dense_classifier = Dense(3, activation = 'softmax')(last_layer)
	xception = Model(inputs = xception.input, outputs = dense_classifier)	
	plot_model(xception, to_file='xception.png', show_shapes=True)

	return xception