import pandas as pd
import cv2
import numpy as np
# import matplotlib.pyplot as plt
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
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import LSTM, GlobalAveragePooling2D, GRU, Bidirectional, UpSampling2D
from keras.layers import BatchNormalization, Input, Activation, Lambda, concatenate, add
from keras.layers import UpSampling2D, Concatenate
from keras.engine import InputLayer
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
	splittensor, Softmax4D
from theano import tensor as T
from keras.layers import Multiply

from utilities import loading_smic_table, loading_samm_table, loading_casme_table
from utilities import class_merging, read_image, create_generator_LOSO
from utilities import LossHistory, record_loss_accuracy
from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall
from models import VGG_16, temporal_module, layer_wise_conv_autoencoder, layer_wise_autoencoder, convolutional_autoencoder, alexnet
from models import tensor_reshape, attention_control, att_shape


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

def test_vgg16_finetuned(weights_name = 'imagenet', classes=5):
	vgg16 = VGG16(weights = 'imagenet')
	last_layer = vgg16.layers[-2].output
	dense_classifier = Dense(classes, activation = 'softmax')(last_layer)
	vgg16 = Model(inputs = vgg16.input, outputs = dense_classifier)
	# vgg16.load_weights(weights_name)
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


def train_res50_imagenet(classes = 5, freeze_flag = 'last'):
	resnet50 = ResNet50(weights = 'imagenet')

	# # load macro
	# last_layer = resnet50.layers[-2].output
	# dense_classifier = Dense(6, activation = 'softmax')(last_layer)
	# resnet50 = Model(inputs = resnet50.input, outputs = dense_classifier)		
	# resnet50.load_weights('res_micro_grayscale_augmentedres50_retrain.h5')	

	last_layer = resnet50.layers[-2].output
	dense_classifier = Dense(classes, activation = 'softmax')(last_layer)
	resnet50 = Model(inputs = resnet50.input, outputs = dense_classifier)
	plot_model(resnet50, to_file='resnet50.png', show_shapes=True)

	for layer in resnet50.layers:
		layer.trainable = True

	# # for 2nd last block
	if freeze_flag == '2nd_last':
		for layer in resnet50.layers[:-25]:
			layer.trainable = False	

	# # for 3rd last block
	elif freeze_flag == '3rd_last':
		for layer in resnet50.layers[:-37]:
			layer.trainable = False		

	# # last
	elif freeze_flag == 'last':
		for layer in resnet50.layers[:-14]:
			layer.trainable = False	
	print(resnet50.summary())

	return resnet50

def train_vgg16_imagenet(classes = 5, freeze_flag = 'last'):
	vgg16 = VGG16(weights = 'imagenet')

	# # load macro
	# last_layer = vgg16.layers[-2].output
	# dense_classifier = Dense(6, activation = 'softmax')(last_layer)
	# vgg16 = Model(inputs = vgg16.input, outputs = dense_classifier)		
	# vgg16.load_weights('res_micro_grayscale_augmentedvgg16_retrain.h5')	

	# # LFW weights
	# last_layer = vgg16.layers[-2].output
	# dense_classifier = Dense(2622, activation = 'softmax')(last_layer)
	# vgg16 = Model(inputs = vgg16.input, outputs = dense_classifier)			
	# vgg16.load_weights('VGG_Face_Deep_16.h5')

	last_layer = vgg16.layers[-2].output
	dense_classifier = Dense(classes, activation = 'softmax')(last_layer)
	vgg16 = Model(inputs = vgg16.input, outputs = dense_classifier)	
	plot_model(vgg16, to_file='vgg16.png', show_shapes=True)
		
	# for layer in vgg16.layers:
	# 	layer.trainable = True

	# train last 2 block
	if freeze_flag == '2nd_last':
		for layer in vgg16.layers[:-8]:
			layer.trainable = False

	# # train last 3 block
	elif freeze_flag == '3rd_last':
		for layer in vgg16.layers[:-9]:
			layer.trainable = False

	# # train last block
	elif freeze_flag == 'last':
		for layer in vgg16.layers[:-7]:
			layer.trainable = False	
	print(vgg16.summary())
	return vgg16

def train_inceptionv3_imagenet(classes = 5, freeze_flag = 'last'):
	inceptionv3 = InceptionV3(weights = 'imagenet')

	# # load macro
	# last_layer = inceptionv3.layers[-2].output
	# dense_classifier = Dense(6, activation = 'softmax')(last_layer)
	# inceptionv3 = Model(inputs = inceptionv3.input, outputs = dense_classifier)		
	# inceptionv3.load_weights('res_micro_grayscale_augmentedincepv3_retrain.h5')

	last_layer = inceptionv3.layers[-2].output
	dense_classifier = Dense(classes, activation = 'softmax')(last_layer)
	inceptionv3 = Model(inputs = inceptionv3.input, outputs = dense_classifier)	
	plot_model(inceptionv3, to_file='inceptionv3.png', show_shapes=True)

	for layer in inceptionv3.layers:
		layer.trainable = True
	
	# # 2nd last incep block
	if freeze_flag == '2nd_last':
		for layer in inceptionv3.layers[:-85]:
			layer.trainable = False

	# # 3rd last incep block
	elif freeze_flag == '3rd_last':
		for layer in inceptionv3.layers[:-117]:
			layer.trainable = False

	# # last block
	elif freeze_flag == 'last':
		for layer in inceptionv3.layers[:-34]:
			layer.trainable = False	

	print(inceptionv3.summary())

	return inceptionv3

def train_xception_imagenet():
	xception = Xception(weights = 'imagenet')
	last_layer = xception.layers[-2].output
	dense_classifier = Dense(5, activation = 'softmax')(last_layer)
	xception = Model(inputs = xception.input, outputs = dense_classifier)	
	plot_model(xception, to_file='xception.png', show_shapes=True)

	return xception

def train_alexnet_imagenet(classes = 5):
	model = alexnet(input_shape = (3, 227, 227), nb_classes = 1000, mean_flag = True)
	model.load_weights('alexnet_weights.h5')



	# add in own classes ( maybe not necessary)
	last_layer = model.layers[-3].output
	dense_classifier = Dense(5, activation = 'softmax', name='me_dense')(last_layer)
	model = Model(inputs = model.input, outputs = dense_classifier)
	plot_model(model, to_file = 'alexnet', show_shapes = True)
	print(model.summary())


	# freezing

	# 3rd Last
	for layer in model.layers[:-22]:
		layer.trainable = False

	# 2nd Last
	for layer in model.layers[:-20]:
		layer.trainable = False

	# Last
	for layer in model.layers[:-14]:
		layer.trainable = False


	return model

def train_shallow_alexnet_imagenet(classes = 5, freeze_flag=None):
	model = alexnet(input_shape = (3, 227, 227), nb_classes = 1000, mean_flag = True)
	model.load_weights('alexnet_weights.h5')

	# modify architecture
	last_conv_1 = model.layers[5].output
	conv_2 = Conv2D(256, (5, 5), strides=(1, 1), activation='relu', name='conv_2', kernel_initializer='he_normal', bias_initializer='he_normal')(last_conv_1)
	conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
	conv_2 = crosschannelnormalization(name="convpool_2")(conv_2)
	conv_2 = ZeroPadding2D((2,2))(conv_2)

	conv_2 = Flatten(name="flatten")(conv_2)
	conv_2 = Dropout(0.5)(conv_2)
	dense_1 = Dense(classes, kernel_initializer = 'he_normal', bias_initializer = 'he_normal', name='last_fc')(conv_2)
	prediction = Activation("softmax", name = 'softmax_activate')(dense_1)

	model = Model(inputs = model.input, outputs = prediction)		
	# plot_model(model, to_file='shallowalex', show_shapes =True)
	# print(model.summary())
	return model

def train_3conv_alexnet_imagenet(classes = 5):
	model = alexnet(input_shape = (3, 227, 227), nb_classes = 1000, mean_flag = True)
	model.load_weights('alexnet_weights.h5')

	# modify architecture
	last_conv_1 = model.layers[5].output
	conv_2 = Conv2D(256, (5, 5), strides=(1, 1), activation='relu', name='conv_2', kernel_initializer='he_normal', bias_initializer='he_normal')(last_conv_1)
	conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
	conv_2 = crosschannelnormalization(name="convpool_2")(conv_2)
	conv_2 = ZeroPadding2D((2,2))(conv_2)

	conv_3 = Conv2D(384, (3, 3), strides=(1, 1), activation='relu', name='conv_3', kernel_initializer='he_normal', bias_initializer='he_normal')(conv_2)
	conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_3)
	conv_3 = crosschannelnormalization(name="convpool_3")(conv_3)
	conv_3 = ZeroPadding2D((2,2))(conv_3)

	conv_3 = Flatten(name="flatten")(conv_3)
	conv_3 = Dropout(0.5)(conv_3)
	dense_1 = Dense(classes, kernel_initializer = 'he_normal', bias_initializer = 'he_normal')(conv_3)
	prediction = Activation("softmax")(dense_1)

	model = Model(inputs = model.input, outputs = prediction)		
	plot_model(model, to_file='shallowalex', show_shapes =True)
	print(model.summary())
	return model

def train_shallow_alexnet_imagenet_with_attention(classes = 5, freeze_flag = 'last'):
	model = train_shallow_alexnet_imagenet(classes)
	# get conv1 output
	conv1 = model.layers[2].output
	# get conv2 output
	conv2 = model.get_layer('conv_2').output
	# get input to conv2
	input_conv2 = model.layers[2].output
	# print(K.int_shape(conv2))
	# print(K.int_shape(input_conv2))
	# get shape for feature fc for non-static parameter
	# feature_fc_dim = K.int_shape(model.layers[-3].output)[1]
	# get model FC Layer with softmax
	softmax_activation_layer = model.layers[-1]
	# get activation for Last FC with softmax
	softmax_activation = model.layers[-1].output
	# print(feature_fc_dim)
	# print(softmax_activation)

	# reshape conv2 output into (w*h, num_filters)
	adapt_conv_shape = Lambda(tensor_reshape, output_shape=(256, ))(conv2)

	# perform softmax activation on adapted conv2 output
	att = softmax_activation_layer(adapt_conv_shape)



	# Restructuring att	and padding(make sure it conforms with conv1 shape)
	att = Lambda(attention_control, output_shape = att_shape)([att, softmax_activation])
	# print("original")
	# print(K.int_shape(att))
	att = UpSampling2D(size = (2, 2), data_format="channels_first")(att)
	# print("upsampling2d")
	# print(K.int_shape(att))
	att = ZeroPadding2D(((1, 0), (1, 0)))(att)
	# print("zeropadding")
	# print(K.int_shape(att))

	# Multiply(merge) with conv1 output
	apply_attention = Multiply()([att, input_conv2])

	# get necessary layers below attention applied layer
	late_layers = model.layers[3:]
	# print(late_layers)
	for layer in late_layers:
		# print(layer)
		apply_attention = layer(apply_attention)

	model = Model(inputs = model.input, outputs = apply_attention)
	plot_model(model, to_file = 'attention_shallow_alexnet.png', show_shapes=True)
	# print(model.summary())
	return model

def train_dual_stream_shallow_alexnet(classes = 5, freeze_flag=None):
	input_mag = Input(shape=(3, 227, 227))
	input_strain = Input(shape=(3, 227, 227))
	model_mag = train_shallow_alexnet_imagenet(classes = classes)
	model_strain = train_shallow_alexnet_imagenet(classes = classes)

	model_mag = Model(inputs = model_mag.input, outputs = model_mag.get_layer('flatten').output)
	model_strain = Model(inputs = model_strain.input, outputs = model_strain.get_layer('flatten').output)
	flatten_mag = model_mag(input_mag)
	flatten_strain = model_strain(input_strain)

	plot_model(model_mag, to_file = 'mag_model', show_shapes=True)
	plot_model(model_strain, to_file = 'strain_model', show_shapes=True)

	# concatenate
	concat = Concatenate(axis=-1)([flatten_mag, flatten_strain])
	dropout = Dropout(0.5)(concat)

	dense_1 = Dense(classes, kernel_initializer = 'he_normal', bias_initializer = 'he_normal', name='last_fc')(dropout)
	prediction = Activation("softmax", name = 'softmax_activate')(dense_1)


	model = Model(inputs = [input_mag, input_strain], outputs = prediction)
	plot_model(model, to_file = 'concat_model', show_shapes=True)
	print(model.summary())

	return model

# model = dual_stream_shallow_alexnet()

# # model = train_shallow_alexnet_imagenet()
# model = train_shallow_alexnet_imagenet_with_attention()
# print(model.get_layer('flatten').get_output_at(1))
# model = Model(inputs = model.input, outputs = model.get_layer('flatten').get_output_at(1))

# plot_model(model, to_file='test_attention', show_shapes=True)

# train_alexnet_imagenet()
# train_shallow_alexnet_imagenet()
# model = train_shallow_alexnet_imagenet()

# model = alexnet(input_shape = (3, 227, 227), nb_classes = 5, mean_flag = True)
# plot_model(model, to_file = 'alexnet', show_shapes = True)
# model.load_weights('alexnet_weights.h5')

# train_res50_imagenet()
# train_vgg16_imagenet()
# train_inceptionv3_imagenet()

# test attention
