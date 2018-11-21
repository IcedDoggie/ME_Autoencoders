import numpy as np
import random
import theano

from keras.datasets import mnist
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import LSTM, GlobalAveragePooling2D, GRU, Bidirectional, UpSampling2D
from keras.layers import BatchNormalization, Lambda, Input
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
from keras.layers import GlobalAveragePooling2D, Concatenate

from keras.losses import *



def euclidean_distance(vects):
	x, y = vects
	sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
	return K.sqrt(K.maximum(sum_square, K.epsilon()))

def concat_feature(vects):
	x1, x2 = vects
	# x1 = K.reshape(x1, (1, 4096))
	# x2 = K.reshape(x2, (1, 4096))
	print(K.int_shape(vects))
	print(K.int_shape(x1))
	print(K.int_shape(x2))
	concat_feat = K.concatenate([x1, x2], axis=-1)
	print(K.int_shape(concat_feat))

	return concat_feat

def deconcat_feature(vects):

	fx, fx_hat = K.eval(vects)

	return fx, fx_hat

def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
	'''Contrastive loss from Hadsell-et-al.'06
	http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	'''
	margin = 1
	sqaure_pred = K.square(y_pred)
	margin_square = K.square(K.maximum(margin - y_pred, 0))
	return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)
	


def feature_distance_loss(y_true, vects):
	print(K.int_shape(vects))
	shape_i, shape_j = K.int_shape(vects)
	subshape = int(shape_j/2)
	fx = vects[:, 0:subshape]
	fx_hat = vects[:, subshape: shape_j]

	# sum_square = (1 / (2*K.count_params(fx))) * (K.sum(K.square(fx - fx_hat), axis=1, keepdims=True))
	feature_loss = K.mean(K.square(fx_hat - fx), axis=-1)

	# K.eval(feature_loss)
	return feature_loss

def feature_distance_cross_db_loss(y_true, vects):

	shape_i, shape_j = K.int_shape(vects)
	subshape = int(shape_j/3)
	fx = vects[:, 0:subshape]
	fx_hat = vects[:, subshape: shape_j]

	# sum_square = (1 / (2*K.count_params(fx))) * (K.sum(K.square(fx - fx_hat), axis=1, keepdims=True))
	feature_loss = K.mean(K.square(fx_hat - fx), axis=-1)

	# K.eval(feature_loss)
	return feature_loss	

# def feature_output_shape(shapes):
# 	shape1, shape2 = shapes
# 	return (shape1, 2)

# TODO Implement Siamese Pairs
def create_siamese_pairs(X_ori, X_aug, y_ori, y_aug):
	pairs = []
	labels = []
	no_of_augmentation = 6
	aug_tracker = 0

	for img_counter in range(len(X_ori)):
		aug_count = 0
		curr_ori_img = X_ori[img_counter]
		curr_ori_label = y_ori[img_counter]

		while aug_count < no_of_augmentation:
			curr_aug_img = X_aug[aug_tracker]
			curr_aug_label = y_aug[aug_tracker]
			pairs += [[curr_ori_img, curr_aug_img]] 
			labels += [[curr_ori_label, curr_aug_label]]

			aug_count += 1
			aug_tracker += 1
	pairs = np.asarray(pairs)
	labels = np.asarray(labels)

	print("True Pairs Created ")
	print(pairs.shape)
	print(labels.shape)
	return pairs, labels

def create_siamese_pairs_crossdb(X_ori, X_aug, y_ori, y_aug):
	pairs = []
	labels = []

	# emotion based sampling
	print(X_ori.shape)
	print(X_aug.shape)
	for img_counter in range(len(X_ori)):
		curr_ori_img = X_ori[img_counter]
		curr_ori_label = y_ori[img_counter]
		compare_param_ori = np.argmax(curr_ori_label)

		for sub_counter in range(len(X_aug)):
			curr_aug_img = X_aug[sub_counter]
			curr_aug_label = y_aug[sub_counter]
			compare_param_aug = np.argmax(curr_aug_label)

			if compare_param_ori == compare_param_aug:
				pairs += [[curr_ori_img, curr_aug_img]]
				labels += [[curr_ori_label, curr_aug_label]]
				# print("matched!")
				# print(curr_aug_label)
				# print(curr_ori_label)
	pairs = np.asarray(pairs)
	labels = np.asarray(labels)

	print("True Pairs Created ")

	print(pairs.shape)
	print(labels.shape)
	return pairs, labels	
		# print(curr_ori_label)

	# print(y_ori)
	# print(y_aug)

def siamese_base_network(classes=5):
	input_layer = Input(shape = (3, 64, 64))
	x = Conv2D(filters = 64, kernel_size = (5, 5), strides = (1, 1), activation = 'relu')(input_layer)
	x = MaxPooling2D(pool_size = (2, 2), strides = (1, 1))(x)
	x = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), activation = 'relu')(x)
	x = MaxPooling2D(pool_size = (2, 2), strides = (1, 1))(x)
	x = Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), activation = 'relu')(x)
	x = MaxPooling2D(pool_size = (2, 2), strides = (1, 1))(x)
	x = Flatten()(x)
	x = Dense(300 , activation = 'relu')(x)
	x = Dense(classes, activation = 'softmax')(x)

	siamese_model = Model(inputs = input_layer, outputs = x)

	return siamese_model

def siamese_base(classes = 3, freeze_flag = None):
	siamese_net = siamese_base_network(classes = classes)
	last_layer = siamese_net.layers[-2].output
	siamese_feat = Model(inputs = siamese_net.input, outputs = last_layer)

	auxiliary_output = Dense(classes, activation = 'softmax')
	auxiliary_output_2 = Dense(classes, activation = 'softmax')


	plot_model(siamese_feat, to_file='siamese_feature_encoder.png', show_shapes=True)
	input_a = Input(shape=(3, 64, 64))
	input_b = Input(shape=(3, 64, 64))

	feature_a = siamese_feat(input_a)
	feature_b = siamese_feat(input_b)	

	softmax_a = auxiliary_output(feature_a)
	softmax_b = auxiliary_output_2(feature_b)


	# concat feature a and b
	features = Concatenate(axis=-1)([feature_a, feature_b])
	siamese = Model(inputs = [input_a, input_b], outputs = [softmax_a, softmax_b, features])	

	plot_model(siamese, to_file='siamese_base.png', show_shapes=True)
	print("MileStone #2")
	print(siamese.summary())
		

	return siamese	


def siamese_vgg16_imagenet(classes = 5):
	vgg16 = VGG16(weights = 'imagenet')
	last_layer = vgg16.layers[-2].output
	vgg16 = Model(inputs = vgg16.input, outputs = last_layer)
	auxiliary_output = Dense(classes, activation = 'softmax')


	# for layer in inceptionv3.layers:
	# 	layer.trainable = True
	
	# # train last 2 block
	# for layer in vgg16.layers[:-8]:
	# 	layer.trainable = False

	# # train last 3 block
	# for layer in vgg16.layers[:-9]:
	# 	layer.trainable = False

	# # train last block
	# for layer in vgg16.layers[:-7]:
	# 	layer.trainable = False	

	plot_model(vgg16, to_file='vgg16.png', show_shapes=True)
	input_a = Input(shape=(3, 224, 224))
	input_b = Input(shape=(3, 224, 224))
	feature_a = vgg16(input_a)
	feature_b = vgg16(input_b)	

	softmax_a = auxiliary_output(feature_a)

	# concat feature a and b
	features = Concatenate(axis=-1)([feature_a, feature_b])


	distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feature_a, feature_b])


	print("Done")	
	# vgg16 = Model(inputs = [input_a, input_b], outputs = [softmax_a, distance])	
	vgg16 = Model(inputs = [input_a, input_b], outputs = [softmax_a, features])	

	plot_model(vgg16, to_file='Siamese_vgg16.png', show_shapes=True)
	print("MileStone #2")
	print(vgg16.summary())
		

	return vgg16

def siamese_res50_network(classes = 5):
	res50 = ResNet50(weights = 'imagenet')
	last_layer = res50.layers[-2].output
	res50 = Model(inputs = res50.input, outputs = last_layer)
	auxiliary_output = Dense(classes, activation = 'softmax')


	# for layer in res50.layers:
	# 	layer.trainable = True

	# # for 2nd last block
	# for layer in res50.layers[:-25]:
	# 	layer.trainable = False	

	# for 3rd last block
	for layer in res50.layers[:-37]:
		layer.trainable = False		

	# for layer in res50.layers[:-14]:
	# 	layer.trainable = False	

	plot_model(res50, to_file='res50.png', show_shapes=True)
	input_a = Input(shape=(3, 224, 224))
	input_b = Input(shape=(3, 224, 224))
	feature_a = res50(input_a)
	feature_b = res50(input_b)	

	softmax_a = auxiliary_output(feature_a)

	# concat feature a and b
	features = Concatenate(axis=-1)([feature_a, feature_b])


	distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feature_a, feature_b])


	print("Done")	
	# vgg16 = Model(inputs = [input_a, input_b], outputs = [softmax_a, distance])	
	res50 = Model(inputs = [input_a, input_b], outputs = [softmax_a, features])	

	plot_model(res50, to_file='Siamese_res50.png', show_shapes=True)
	print("MileStone #2")
	print(res50.summary())
		

	return res50	

def siamese_vgg16_crossdb_imagenet(classes = 3, freeze_flag = 'last'):
	vgg16 = VGG16(weights = 'imagenet')
	last_layer = vgg16.layers[-2].output
	vgg16 = Model(inputs = vgg16.input, outputs = last_layer)
	auxiliary_output = Dense(classes, activation = 'softmax')
	auxiliary_output_2 = Dense(classes, activation = 'softmax')


	# for layer in inceptionv3.layers:
	# 	layer.trainable = True
	
	# # train last 2 block
	if freeze_flag == '2nd_last':	
		for layer in vgg16.layers[:-8]:
			layer.trainable = False

	# # train last 3 block
	elif freeze_flag == '3rd_last':
		for layer in vgg16.layers[:-9]:
			layer.trainable = False

	# train last block
	elif freeze_flag == 'last':
		for layer in vgg16.layers[:-7]:
			layer.trainable = False	

	plot_model(vgg16, to_file='vgg16.png', show_shapes=True)
	input_a = Input(shape=(3, 224, 224))
	input_b = Input(shape=(3, 224, 224))
	# input_c = Input(shape=(3, 224, 224))
	feature_a = vgg16(input_a)
	feature_b = vgg16(input_b)	
	# feature_c = vgg16(input_c)

	softmax_a = auxiliary_output(feature_a)
	softmax_b = auxiliary_output_2(feature_b)
	# softmax_c = auxiliary_output(feature_c)

	# concat feature a and b
	features = Concatenate(axis=-1)([feature_a, feature_b])


	# distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feature_a, feature_b])


	print("Done")	
	# vgg16 = Model(inputs = [input_a, input_b], outputs = [softmax_a, distance])	
	vgg16 = Model(inputs = [input_a, input_b], outputs = [softmax_a, softmax_b, features])	

	plot_model(vgg16, to_file='Siamese_vgg16.png', show_shapes=True)
	print("MileStone #2")
	print(vgg16.summary())
		

	return vgg16	

# siamese_vgg16_imagenet()
# siamese_base()