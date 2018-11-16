from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import LSTM, GlobalAveragePooling2D, GRU, Bidirectional, UpSampling2D
from keras.layers import BatchNormalization, Input, Activation, Lambda, concatenate, add
from keras.engine import InputLayer
import pydot, graphviz

from keras.utils import np_utils, plot_model, Sequence
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, splittensor, Softmax4D

import theano
from theano import tensor as T


def VGG_16(spatial_size, classes, channels, channel_first=True, weights_path=None):
	model = Sequential()
	if channel_first:
		model.add(ZeroPadding2D((1,1),input_shape=(channels, spatial_size, spatial_size)))
	else:
		model.add(ZeroPadding2D((1,1),input_shape=(spatial_size, spatial_size, channels)))


	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2))) # 33

	model.add(Flatten())
	model.add(Dense(4096, activation='relu')) # 34
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu')) # 35
	model.add(Dropout(0.5))
	model.add(Dense(2622, activation='softmax')) # Dropped


	if weights_path:
		model.load_weights(weights_path)
	model.pop()
	model.add(Dense(classes, activation='softmax')) # 36

	return model

def temporal_module(data_dim, timesteps_TIM, classes, weights_path=None):
	model = Sequential()
	model.add(LSTM(3000, return_sequences=False, input_shape=(timesteps_TIM, data_dim)))
	#model.add(LSTM(3000, return_sequences=False))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(classes, activation='sigmoid'))

	if weights_path:
		model.load_weights(weights_path)

	return model	

def layer_wise_conv_autoencoder(input_shape, encoded_dim, decoded_dim):

	input_img = Input( shape = input_shape )

	maxPool = MaxPooling2D((2, 2), padding = 'same')(input_img)	
	conv1 = Conv2D(encoded_dim, (3, 3), activation = 'relu', padding = 'same')(maxPool)
	maxPool1 = MaxPooling2D((2, 2), padding = 'same')(conv1)
	# maxPool2 = MaxPooling2D((2, 2), padding = 'same')(maxPool1)
	batchNorm1 = BatchNormalization()(maxPool1)
	upsamp1 = UpSampling2D(2)(batchNorm1)
	upsamp2 = UpSampling2D(2)(upsamp1)
	decoded1 = Conv2D(decoded_dim, (3, 3), activation = 'relu', padding = 'same')(upsamp2)

	autoencoder1 = Model(input = input_img, output = decoded1)
	encoder1 = Model(input = input_img, output = batchNorm1)
	
	return autoencoder1, encoder1

def layer_wise_autoencoder(input_shape, encoded_dim, decoded_dim):

	input_img = Input( shape = input_shape )

	# gap = GlobalAveragePooling2D(data_format='channels_first')(input_img)	
	dense1 = Dense(encoded_dim, activation = 'relu')(input_img)
	maxPool1 = MaxPooling2D((2, 2), padding = 'same')(dense1)
	batchNorm1 = BatchNormalization()(maxPool1)
	decoded1 = Dense(decoded_dim, activation = 'relu')(batchNorm1)

	autoencoder1 = Model(input = input_img, output = decoded1)
	encoder1 = Model(input = input_img, output = batchNorm1)
	
	return autoencoder1, encoder1


def convolutional_autoencoder(spatial_size, channel_first=True, weights_path=None):
	model = Sequential()


	model.add(Conv2D(112, (3, 3), activation='relu', input_shape=(3, spatial_size, spatial_size), padding='same'))
	model.add(MaxPooling2D( pool_size=(2, 2), strides=2, padding='same'))
	# model.add(Conv2D(112, (3, 3), activation='relu', padding='same'))
	# model.add(MaxPooling2D( pool_size=(2, 2), strides=2, padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D( pool_size=(2, 2), strides=2, padding='same'))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D( pool_size=(2, 2), strides=2, padding='same'))
	model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D( pool_size=(2, 2), strides=2, padding='same'))		
	model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D( pool_size=(2, 2), strides=2, padding='same'))

	# decoder
	model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D(2))	
	model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D(2))	
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D(2))				
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D(2))
	# model.add(Conv2D(112, (3, 3), activation='relu', padding='same'))
	# model.add(UpSampling2D(2))	
	model.add(Conv2D(112, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D(2))
	model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

	if weights_path:
		model.load_weights(weights_path)


	return model
from theano import tensor as T
def mean_subtract(img):


    img = T.set_subtensor(img[:,0,:,:],img[:,0,:,:] - 123.68)
    img = T.set_subtensor(img[:,1,:,:],img[:,1,:,:] - 116.779)
    img = T.set_subtensor(img[:,2,:,:],img[:,2,:,:] - 103.939)

    return img / 255.0

def alexnet(input_shape, nb_classes, mean_flag): 
	# code adapted from https://github.com/heuritech/convnets-keras

	inputs = Input(shape=input_shape, name='main_input')

	if mean_flag:
		mean_subtraction = Lambda(mean_subtract, name='mean_subtraction')(inputs)
		conv_1 = Conv2D(96, (11, 11), strides=(4,4), activation='relu',
						   name='conv_1', kernel_initializer='he_normal', bias_initializer='he_normal')(mean_subtraction)
	else:
		conv_1 = Conv2D(96, (11, 11), strides=(4,4), activation='relu',
						   name='conv_1', kernel_initializer='he_normal', bias_initializer='he_normal')(inputs)

	conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
	conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
	conv_2 = ZeroPadding2D((2,2))(conv_2)
	conv_2 = concatenate([
		Conv2D(128, (5, 5), activation="relu", kernel_initializer='he_normal', bias_initializer='he_normal', name='conv_2_'+str(i+1))(
		splittensor(ratio_split=2,id_split=i)(conv_2)
		) for i in range(2)], axis=1, name="conv_2")

	conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
	conv_3 = crosschannelnormalization()(conv_3)
	conv_3 = ZeroPadding2D((1,1))(conv_3)
	conv_3 = Conv2D(384, (3, 3), activation='relu', name='conv_3', kernel_initializer='he_normal', bias_initializer='he_normal')(conv_3)

	conv_4 = ZeroPadding2D((1,1))(conv_3)
	conv_4 = concatenate([
		Conv2D(192, (3, 3), activation="relu", kernel_initializer='he_normal', bias_initializer='he_normal', name='conv_4_'+str(i+1))(
		splittensor(ratio_split=2,id_split=i)(conv_4)
		) for i in range(2)], axis=1, name="conv_4")

	conv_5 = ZeroPadding2D((1,1))(conv_4)
	conv_5 = concatenate([
		Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal', bias_initializer='he_normal', name='conv_5_'+str(i+1))(
		splittensor(ratio_split=2,id_split=i)(conv_5)
		) for i in range(2)], axis=1, name="conv_5")

	dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

	dense_1 = Flatten(name="flatten")(dense_1)
	dense_1 = Dense(4096, activation='relu',name='dense_1', kernel_initializer='he_normal', bias_initializer='he_normal')(dense_1)
	dense_2 = Dropout(0.5)(dense_1)
	dense_2 = Dense(4096, activation='relu',name='dense__2', kernel_initializer='he_normal', bias_initializer='he_normal')(dense_2)
	dense_3 = Dropout(0.5)(dense_2)
	dense_3 = Dense(nb_classes,name='dense_3_new', kernel_initializer='he_normal', bias_initializer='he_normal')(dense_3)

	prediction = Activation("softmax",name="softmax")(dense_3)

	alexnet = Model(inputs = inputs, outputs = prediction)
	
	return alexnet



# model = alexnet(input_shape = (3, 227, 227), nb_classes = 5, mean_flag = True)
# model = Model(inputs = model.input, outputs = model.layers[-22].output)
# print(model.summary())
# plot_model(model, to_file = 'alexnet', show_shapes = True)
# # model.load_weights('alexnet_weights.h5')
