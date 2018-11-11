from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import LSTM, GlobalAveragePooling2D, GRU, Bidirectional, UpSampling2D
from keras.layers import BatchNormalization, Input
from keras.engine import InputLayer
import pydot, graphviz

from keras.utils import np_utils, plot_model, Sequence


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
	model.add(Conv2D(112, (3, 3), activation='relu', padding='same'))
	model.add(UpSampling2D(2))
	model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

	if weights_path:
		model.load_weights(weights_path)


	return model


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

def get_model(summary=False):
	""" Return the Keras model of the network
	"""
	model = Sequential()
	model.add(ZeroPadding3D((1,1,1),input_shape=(3, 6, 112, 112)))

	# 1st layer group
	# model.add(InputLayer(batch_input_shape = (None, 3, 6, 112, 112)))
	model.add(Conv3D(64, (3, 3, 3), activation='relu', 
							padding='same', name='conv1',
							strides=(1, 1, 1), 
							))
	model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
						   padding='valid', name='pool1'))
	# 2nd layer group
	model.add(Conv3D(128, (3, 3, 3), activation='relu', 
							padding='same', name='conv2',
							strides=(1, 1, 1)))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
						   padding='valid', name='pool2'))
	# 3rd layer group
	model.add(Conv3D(256, (3, 3, 3), activation='relu', 
							padding='same', name='conv3a',
							strides=(1, 1, 1)))
	model.add(Conv3D(256, (3, 3, 3), activation='relu', 
							padding='same', name='conv3b',
							strides=(1, 1, 1)))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
						   padding='valid', name='pool3'))
	# 4th layer group
	model.add(Conv3D(512, (3, 3, 3), activation='relu', 
							padding='same', name='conv4a',
							strides=(1, 1, 1)))
	model.add(Conv3D(512, (3, 3, 3), activation='relu', 
							padding='same', name='conv4b',
							strides=(1, 1, 1)))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
						   padding='valid', name='pool4'))
	# 5th layer group
	model.add(Conv3D(512, (3, 3, 3), activation='relu', 
							padding='same', name='conv5a',
							strides=(1, 1, 1)))
	model.add(Conv3D(512, (3, 3, 3), activation='relu', 
							padding='same', name='conv5b',
							strides=(1, 1, 1)))
	model.add(ZeroPadding3D(padding=(1, 1, 1)))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
						   padding='valid', name='pool5'))
	model.add(Flatten())
	# FC layers group
	model.add(Dense(4096, activation='relu', name='fc6'))
	model.add(Dropout(.5))
	model.add(Dense(4096, activation='relu', name='fc7'))
	model.add(Dropout(.5))
	model.add(Dense(487, activation='softmax', name='fc8'))
	if summary:
		print(model.summary())
	return model	

def alex_net(weights_path = None):

	model = Sequential()
	model.add(Conv2D(64, kernel_size=(11, 11), padding='same', input_shape=(3, 224, 224)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3, 3)))

	model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3, 3)))

	model.add(Conv2D(192, kernel_size=(3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(192, kernel_size=(3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3, 3)))

	model.add(Flatten())
	model.add(Dense(2048), activation = 'relu')
	model.add(Dense(2048), activation = 'relu')
	model.add(Dense(1000), activation = 'softmax')


model = get_model(summary=True)
print(model.summary())
plot_model(model, to_file = 'c3d', show_shapes=True)
model.load_weights('/home/ice/Documents/ME_Autoencoders/c3d-sports1M_weights.h5')
# print("weights loaded")
