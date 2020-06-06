import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pydot, graphviz
import sys
from theano import tensor as T


from keras.models import Sequential, Model
from keras.utils import np_utils, plot_model, Sequence
from keras import metrics, activations
from keras import optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.preprocessing import image as img
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Concatenate, Lambda
from keras.layers import BatchNormalization, Input, Activation, Lambda, concatenate, add
from keras.layers import Multiply, Concatenate, Add


from utilities import class_merging
from utilities import LossHistory, record_loss_accuracy
from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall, sklearn_macro_f1
from models import VGG_16, temporal_module, layer_wise_conv_autoencoder, layer_wise_autoencoder, convolutional_autoencoder
from networks import test_res50_finetuned, test_vgg16_finetuned, test_inceptionv3_finetuned
from networks import train_res50_imagenet, train_vgg16_imagenet, train_inceptionv3_imagenet, train_alexnet_imagenet, train_shallow_alexnet_imagenet
from networks import test_vgg16_imagenet, test_inceptionv3_imagenet, test_res50_imagenet
from networks import test_vgg19_imagenet, test_mobilenet_imagenet, test_xception_imagenet, test_inceptionResV2_imagenet
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
	splittensor, Softmax4D

from keras import backend as K
from keras.layers import Layer, Input



class Stitch_Unit(Layer):

	def __init__(self, activation=None, **kwargs):
		self.stitch_arr = None
		self.activation = activations.get(activation)
		super(Stitch_Unit, self).__init__(**kwargs)

	def build(self, inputs):
		stitch_dim = (inputs[2], inputs[3])

		self.stitch_arr = self.add_weight(name='stitcher',
									  shape=(stitch_dim),
									  initializer='uniform',
									  trainable=True)
		super(Stitch_Unit, self).build(self.stitch_arr)  # Be sure to call this at the end

	def call(self, x):
		linear_prod = K.dot(x, self.stitch_arr)
		if self.activation is not None:
			linear_prod = self.activation(linear_prod)
		
		return linear_prod

	def compute_output_shape(self, input_shape):
		# print("Compute Output Shape")
		depth = int(input_shape[1])
		resolution = input_shape[2]

		new_shape = (None, depth, resolution, resolution)
		# print("new shape")
		# print(new_shape)
		return new_shape


def mean_subtract(img):


    img = T.set_subtensor(img[:,0,:,:],img[:,0,:,:] - 123.68)
    img = T.set_subtensor(img[:,1,:,:],img[:,1,:,:] - 116.779)
    img = T.set_subtensor(img[:,2,:,:],img[:,2,:,:] - 103.939)

    return img / 255.0

# Implement Stitch Unit for FC
# class Stitch_Unit_FC(Layer):



def train_shallow_alexnet_imagenet_stitch(classes = 5, freeze_flag = None, mean_flag=True, spatial_size=227):
	input_A = Input(shape=(3, spatial_size, spatial_size))
	input_B = Input(shape=(3, spatial_size, spatial_size))

	# 1st Block
	if mean_flag:
		mean_subtraction = Lambda(mean_subtract, name='mean_subtraction')(input_A)
		conv_1 = Conv2D(96, (11, 11), strides=(4,4), activation='relu',
						   name='conv_1', kernel_initializer='he_normal', bias_initializer='he_normal')(mean_subtraction)
		mean_subtraction_B = Lambda(mean_subtract, name='mean_subtraction_B')(input_B)
		conv_1_B = Conv2D(96, (11, 11), strides=(4,4), activation='relu',
						   name='conv_1_B', kernel_initializer='he_normal', bias_initializer='he_normal')(mean_subtraction_B)

	else:
		conv_1 = Conv2D(96, (11, 11), strides=(4,4), activation='relu',
						   name='conv_1', kernel_initializer='he_normal', bias_initializer='he_normal')(input_A)
		conv_1_B = Conv2D(96, (11, 11), strides=(4,4), activation='relu',
						   name='conv_1_B', kernel_initializer='he_normal', bias_initializer='he_normal')(input_B)

	conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
	conv_2_B = MaxPooling2D((3, 3), strides=(2,2))(conv_1_B)
	concat_1 = Concatenate(axis=1)([conv_2, conv_2_B])
	stitch_1_main = Stitch_Unit(activation='relu')(concat_1)
	stitch_1 = Lambda(lambda x: x[:, 0:96, :, :], output_shape=(96, 27, 27))(stitch_1_main)
	stitch_1_B = Lambda(lambda x: x[:, 96:, :, :], output_shape=(96, 27, 27))(stitch_1_main)
	conv_2 = crosschannelnormalization(name="convpool_1")(stitch_1)
	conv_2_B = crosschannelnormalization(name="convpool_1_B")(stitch_1_B)
	conv_2 = ZeroPadding2D((2,2))(conv_2)
	conv_2_B = ZeroPadding2D((2,2))(conv_2_B)

	# 2nd Block (Straight Style)
	conv_2 = Conv2D(256, (5, 5), activation='relu', name='conv_2', kernel_initializer='he_normal', bias_initializer='he_normal')(conv_2)
	conv_2_B = Conv2D(256, (5, 5), activation='relu', name='conv_2_B', kernel_initializer='he_normal', bias_initializer='he_normal')(conv_2_B)


	conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
	conv_2_B = MaxPooling2D((3, 3), strides=(2, 2))(conv_2_B)
	concat_2 = Concatenate(axis=1)([conv_2, conv_2_B])
	stitch_2_main = Stitch_Unit(activation='relu')(concat_2)
	stitch_2 = Lambda(lambda x: x[:, 0:256, :, :], output_shape=(256, 13, 13))(stitch_2_main)
	stitch_2_B = Lambda(lambda x: x[:, 256:, :, :], output_shape=(256, 13, 13))(stitch_2_main)
	conv_2 = crosschannelnormalization()(stitch_2)
	conv_2_B = crosschannelnormalization()(stitch_2_B)
	conv_2 = ZeroPadding2D((1,1))(conv_2)
	conv_2_B = ZeroPadding2D((1,1))(conv_2_B)

	# 3rd Block
	conv_3 = Conv2D(384, (3, 3), activation='relu', name='conv_3', kernel_initializer='he_normal', bias_initializer='he_normal')(conv_2)
	conv_3_B = Conv2D(384, (3, 3), activation='relu', name='conv_3_B', kernel_initializer='he_normal', bias_initializer='he_normal')(conv_2_B)
	concat_3 = Concatenate(axis=1)([conv_3, conv_3_B])
	stitch_3_main = Stitch_Unit(activation='relu')(concat_3)
	stitch_3 = Lambda(lambda x: x[:, 0:384, :, :], output_shape=(384, 13, 13))(stitch_3_main)
	stitch_3_B = Lambda(lambda x: x[:, 384:, :, :], output_shape=(384, 13, 13))(stitch_3_main)
	conv_3 = ZeroPadding2D((1,1))(stitch_3)
	conv_3_B = ZeroPadding2D((1,1))(stitch_3_B)

	# 4th Block
	conv_4 = Conv2D(384, (3, 3), activation='relu', name='conv_4', kernel_initializer='he_normal', bias_initializer='he_normal')(conv_3)
	conv_4_B = Conv2D(384, (3, 3), activation='relu', name='conv_4_B', kernel_initializer='he_normal', bias_initializer='he_normal')(conv_3_B)
	concat_4 = Concatenate(axis=1)([conv_4, conv_4_B])
	stitch_4_main = Stitch_Unit(activation='relu')(concat_4)
	stitch_4 = Lambda(lambda x: x[:, 0:384, :, :], output_shape=(384, 13, 13))(stitch_4_main)
	stitch_4_B = Lambda(lambda x: x[:, 384:, :, :], output_shape=(384, 13, 13))(stitch_4_main)
	conv_4 = ZeroPadding2D((1,1))(stitch_4)
	conv_4_B = ZeroPadding2D((1,1))(stitch_4_B)	

	# 5th Block
	conv_5 = Conv2D(256, (3, 3), activation='relu', name='conv_5', kernel_initializer='he_normal', bias_initializer='he_normal')(conv_4)
	conv_5_B = Conv2D(256, (3, 3), activation='relu', name='conv_5_B', kernel_initializer='he_normal', bias_initializer='he_normal')(conv_4_B)
	dense_1 = MaxPooling2D((3, 3), strides=(2, 2))(conv_5)
	dense_1_B = MaxPooling2D((3, 3), strides=(2, 2))(conv_5_B)
	# concat_5 = Concatenate(axis=1)([dense_1, dense_1_B])
	# stitch_5_main = Stitch_Unit(activation='relu')(concat_5)
	# stitch_5 = Lambda(lambda x: x[:, 0:256, :, :], output_shape=(256, 6, 6))(stitch_5_main)
	# stitch_5_B = Lambda(lambda x: x[:, 256:, :, :], output_shape=(256, 6, 6))(stitch_5_main)
	
	# FC Layers (to be implemented)

	concat = Multiply()([dense_1, dense_1_B])
	concat = Flatten()(concat)
	dropout = Dropout(0.5)(concat)

	dense_1 = Dense(classes, kernel_initializer = 'he_normal', bias_initializer = 'he_normal', name='last_fc')(dropout)
	prediction = Activation("softmax", name = 'softmax_activate')(dense_1)
	model = Model(inputs = [input_A, input_B], outputs = prediction)
	plot_model(model, to_file = 'train_dual_stream_shallow_alexnet_stitch', show_shapes=True)
	print(model.summary())

	return model



# model = train_shallow_alexnet_imagenet_stitch(classes=5, freeze_flag=None, mean_flag=True)


# x = np.random.rand(1, 3, 227, 227)

# x_2 = np.random.rand(1, 3, 227, 227)

# y = np.asarray([[0, 0, 0, 0, 1]])
# print(x.shape)
# print(x_2.shape)


# adam = optimizers.Adam(lr=0.0001, decay=1e-7)

# # model = Model(inputs = [x, x_2], outputs = y)
# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.sparse_categorical_accuracy])	
# predict = model.predict([x, x_2])
# print(predict.shape)
# print(np.max(predict))
# print(np.min(predict))
# print(predict_2.shape)
# print(np.max(predict_2))
# print(np.min(predict_2))
# plot_model(model, show_shapes=True, to_file='stitch_unit.png')
