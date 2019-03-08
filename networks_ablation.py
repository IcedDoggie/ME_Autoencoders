import pandas as pd
import cv2
import numpy as np
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
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv3D, MaxPooling3D, ZeroPadding3D, AveragePooling2D
from keras.layers import LSTM, GlobalAveragePooling2D, GRU, Bidirectional, UpSampling2D
from keras.layers import BatchNormalization, Input, Activation, Lambda, concatenate, add
from keras.engine import InputLayer
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
	splittensor, Softmax4D
from theano import tensor as T
from keras.layers import Multiply, Concatenate, Add

from utilities import loading_smic_table, loading_samm_table, loading_casme_table
from utilities import class_merging, read_image, create_generator_LOSO
from utilities import LossHistory, record_loss_accuracy
from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall
from models import VGG_16, temporal_module, layer_wise_conv_autoencoder, layer_wise_autoencoder, convolutional_autoencoder, alexnet
from models import tensor_reshape, attention_control, att_shape, l2_normalize, l2_normalize_output_shape, repeat_element_autofeat

def train_shallow_resnet50(weights_name='imagenet', classes=5, ablation_flag = 1):
	model = ResNet50(weights = weights_name)

	# Ablation #1
	if ablation_flag == 1:
		x = model.layers[17].output


	# # Ablation #2
	elif ablation_flag == 2:
		x = model.layers[37].output

	# Ablation #3
	elif ablation_flag == 3:
		x = model.layers[49].output


	# # Ablation #4
	elif ablation_flag == 4:
		x = model.layers[79].output


	# GAP 
	x = GlobalAveragePooling2D()(x)
	x = Dense(classes, activation='softmax')(x)


	# # AVGPOOL-FLATTEN-FC
	# tens_shape = K.int_shape(x)[2]
	# x = AveragePooling2D(pool_size=(tens_shape, tens_shape))(x)
	# x = Flatten()(x)
	# x = Dense(classes, activation = 'softmax')(x)

	model = Model(inputs = model.input, outputs = x)
	plot_model(model, to_file='resnet.png', show_shapes=True)		
	return model

def train_shallow_inceptionv3(weights_name='imagenet', classes=5, ablation_flag=1):
	model = InceptionV3(weights = weights_name)

	# Ablation #1
	if ablation_flag == 1:
		x = model.get_layer('mixed0').output

	# Ablation #2
	elif ablation_flag == 2:
		x = model.get_layer('mixed1').output

	# Ablation #3
	elif ablation_flag == 3:
		x = model.get_layer('mixed2').output

	# Ablation #4
	elif ablation_flag == 4:
		x = model.get_layer('mixed3').output

	# GAP 
	x = GlobalAveragePooling2D()(x)
	x = Dense(classes, activation='softmax')(x)

	model = Model(inputs = model.input, outputs = x)
	plot_model(model, to_file='inceptionv3.png', show_shapes=True)	
	return model



def train_shallow_vgg16(weights_name='imagenet', classes=5, ablation_flag=1):
	model = VGG16(weights = weights_name)

	# Ablation #1
	if ablation_flag == 1:
		x = model.layers[3].output

	# Ablation #2
	elif ablation_flag == 2:
		x = model.layers[6].output

	# Ablation #3
	elif ablation_flag == 3:
		x = model.layers[10].output

	# Ablation #4 (FC workable)
	elif ablation_flag == 4:
		x = model.layers[14].output	

	# GAP 
	x = GlobalAveragePooling2D()(x)
	x = Dense(classes, activation='softmax')(x)	

	# # FC 
	# x = Flatten()(x)
	# x = Dense(4096, activation = 'relu')(x)
	# x = Dense(4096, activation = 'relu')(x)
	# x = Dense(classes, activation='softmax')(x)	


	model = Model(inputs = model.input, outputs = x)
	print(model.summary())
	plot_model(model, to_file='vgg16.png', show_shapes=True)	
	
	return model	

# train_shallow_resnet50()
# train_shallow_inceptionv3()
# train_shallow_vgg16()

# for sub in range(10):
# 	model = train_shallow_vgg16()
# 	del model