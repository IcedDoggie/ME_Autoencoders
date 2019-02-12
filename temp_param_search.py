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
from keras.engine import InputLayer
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
	splittensor, Softmax4D
from theano import tensor as T
from keras.layers import Multiply, Concatenate, Add
from keras.layers import LSTM, GlobalAveragePooling2D, GRU, Bidirectional, UpSampling2D, Reshape


from utilities import loading_smic_table, loading_samm_table, loading_casme_table
from utilities import class_merging, read_image, create_generator_LOSO
from utilities import LossHistory, record_loss_accuracy
from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall
from models import VGG_16, temporal_module, layer_wise_conv_autoencoder, layer_wise_autoencoder, convolutional_autoencoder, alexnet
from models import tensor_reshape, attention_control, att_shape, l2_normalize, l2_normalize_output_shape, repeat_element_autofeat


def train_vgg16_imagenet(classes = 5, freeze_flag = 'last'):
	vgg16 = VGG16(weights = 'imagenet')
	
	last_layer = vgg16.layers[-2].output
	dense_classifier = Dense(classes, activation = 'softmax')(last_layer)
	vgg16 = Model(inputs = vgg16.input, outputs = dense_classifier)	
	plot_model(vgg16)
	for layer in vgg16.layers[:20]:
		layer.trainable = False	

	plot_model(vgg16, to_file='vgg16.png', show_shapes=True)
		

	print(vgg16.summary())
	return vgg16

def train_alexnet_imagenet(classes = 5):
	model = alexnet(input_shape = (3, 227, 227), nb_classes = 1000, mean_flag = True)
	model.load_weights('alexnet_weights.h5')

	# add in own classes ( maybe not necessary)
	last_layer = model.layers[-3].output
	dense_classifier = Dense(5, activation = 'softmax', name='me_dense')(last_layer)
	model = Model(inputs = model.input, outputs = dense_classifier)
	plot_model(model, to_file = 'alexnet', show_shapes = True)
	print(model.summary())


	return model

def train_shallow_alexnet_imagenet(classes = 5, freeze_flag = None):
	model = alexnet(input_shape = (3, 227, 227), nb_classes = 1000, mean_flag = True)
	model.load_weights('alexnet_weights.h5')


	################# Use 2 conv with weights ######################
	conv_2 = model.layers[13].output
	conv_2 = Flatten(name = 'flatten')(conv_2)
	conv_2 = Dropout(0.5)(conv_2)
	################################################################


	dense_1 = Dense(classes, kernel_initializer = 'he_normal', bias_initializer = 'he_normal')(conv_2)
	prediction = Activation("softmax")(dense_1)

	model = Model(inputs = model.input, outputs = prediction)		
	plot_model(model, to_file='shallowalex', show_shapes =True)
	print(model.summary())
	return model

def train_dual_stream_shallow_alexnet(classes = 5, freeze_flag=None):
	sys.setrecursionlimit(10000)

	input_mag = Input(shape=(3, 227, 227))
	input_strain = Input(shape=(3, 227, 227))
	model_mag = train_shallow_alexnet_imagenet(classes = classes)
	model_strain = train_shallow_alexnet_imagenet(classes = classes)

	# FOR MULTIPLYING / ADDITION
	model_mag = Model(inputs = model_mag.input, outputs = model_mag.layers[-5].output)
	model_strain = Model(inputs = model_strain.input, outputs = model_strain.layers[-5].output)

	plot_model(model_mag, show_shapes=True, to_file = 'model_mag')
	plot_model(model_strain, show_shapes=True, to_file = 'model_strain')

	# # FOR CONCATENATION
	# model_mag = Model(inputs = model_mag.input, outputs = model_mag.layers[-4].output)
	# model_strain = Model(inputs = model_strain.input, outputs = model_strain.layers[-4].output)

	flatten_mag = model_mag(input_mag)
	flatten_strain = model_strain(input_strain)

	plot_model(model_mag, to_file = 'mag_model', show_shapes=True)
	plot_model(model_strain, to_file = 'strain_model', show_shapes=True)

	# concatenate FOR MULTIPLY OR ADD
	concat = Multiply()([flatten_mag, flatten_strain])
	# concat = Add()([flatten_mag, flatten_strain])

	# # # concatenate FOR CONCATENATION
	# concat = Concatenate(axis=1)([flatten_mag, flatten_strain])



	concat = Flatten()(concat) # FOR MULTIPLY ADD
	#concat = Lambda(l2_normalize, output_shape=l2_normalize_output_shape)(concat)
	dropout = Dropout(0.5)(concat)

	# fc_1 = Dense(4096, kernel_initializer = 'he_normal', bias_initializer = 'he_normal')(dropout)
	# fc_2 = Dense(4096, kernel_initializer = 'he_normal', bias_initializer = 'he_normal')(fc_1)

	dense_1 = Dense(classes, kernel_initializer = 'he_normal', bias_initializer = 'he_normal', name='last_fc')(dropout)
	prediction = Activation("softmax", name = 'softmax_activate')(dense_1)


	model = Model(inputs = [input_mag, input_strain], outputs = prediction)
	plot_model(model, to_file = 'train_dual_stream_shallow_alexnet', show_shapes=True)
	print(model.summary())

	return model

def temporal_module(data_dim, timesteps_TIM, classes, weights_path=None):
	model = Sequential()
	model.add(LSTM(3000, return_sequences=False, input_shape=(timesteps_TIM, data_dim)))
	#model.add(LSTM(3000, return_sequences=False))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(classes, activation='sigmoid'))

	if weights_path:
		model.load_weights(weights_path)
	print(model.summary())
	return model

def state_cnn_lstm():
	input_img = Input(shape=(3, 64, 64))
	x = Conv2D(32, (3, 3), strides=(1,1), activation='relu', name='conv_1', kernel_initializer='he_normal', bias_initializer='he_normal')(input_img)
	x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
	x = Conv2D(64, (3, 3), strides=(1,1), activation='relu', name='conv_2', kernel_initializer='he_normal', bias_initializer='he_normal')(x)
	x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
	x = Conv2D(64, (3, 3), strides=(1,1), activation='relu', name='conv_3', kernel_initializer='he_normal', bias_initializer='he_normal')(x)
	x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
	# x = Flatten()(x)
	x = Dense(512, kernel_initializer = 'he_normal', bias_initializer = 'he_normal', activation = 'relu')(x)
	x = Dense(512, kernel_initializer = 'he_normal', bias_initializer = 'he_normal', activation = 'relu')(x)

	# # x = Flatten()(x)
	# x = Reshape((64, 5, 512))(x)
	# x = LSTM(512, return_sequences=False, input_shape=(5, 512))(x)
	# x = LSTM(512, return_sequences=False)(x)
	
	model = Model(inputs = input_img, outputs = x)
	plot_model(model, to_file = 'cnnlstm', show_shapes=True)
	print(model.summary())

state_cnn_lstm()	
# train_vgg16_imagenet()
# train_alexnet_imagenet()
# train_shallow_alexnet_imagenet()
# train_dual_stream_shallow_alexnet()
