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
from sklearn.preprocessing import normalize
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
from models import tensor_reshape, attention_control, att_shape, l2_normalize, l2_normalize_output_shape


def train_3conv_alexnet_imagenet(classes = 5, freeze_flag=None):
	# model = alexnet(input_shape = (3, 227, 227), nb_classes = 1000, mean_flag = True)
	# model.load_weights('alexnet_weights.h5')

	# modify architecture
	input_data = Input(shape = (3, 227, 227, 10))
	conv_2 = Conv3D(256, (5, 5, 5), strides=(1, 1, 1), activation='relu', name='conv_2', kernel_initializer='he_normal', bias_initializer='he_normal')(input_data)
	conv_2 = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(conv_2)
	conv_2 = BatchNormalization(axis=1)(conv_2)
	# conv_2 = crosschannelnormalization(name="convpool_2")(conv_2)
	conv_2 = ZeroPadding3D((2, 2, 2))(conv_2)

	conv_3 = Conv3D(384, (3, 3, 3), strides=(1, 1, 1), activation='relu', name='conv_3', kernel_initializer='he_normal', bias_initializer='he_normal')(conv_2)
	conv_3 = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(conv_3)
	conv_3 = BatchNormalization(axis=1)(conv_3)
	# conv_3 = crosschannelnormalization(name="convpool_3")(conv_3)
	conv_3 = ZeroPadding3D((2, 2, 2))(conv_3)

	conv_3 = Flatten(name="flatten")(conv_3)
	conv_3 = Dropout(0.5)(conv_3)
	dense_1 = Dense(classes, kernel_initializer = 'he_normal', bias_initializer = 'he_normal')(conv_3)
	prediction = Activation("softmax")(dense_1)

	model = Model(inputs = input_data, outputs = prediction)		
	plot_model(model, to_file='shallow 3D conv', show_shapes =True)
	print(model.summary())
	return model

# model = train_3conv_alexnet_imagenet()