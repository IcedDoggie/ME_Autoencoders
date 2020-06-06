from __future__ import print_function
from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import pydot, graphviz

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
from keras.layers.convolutional import Conv1D, MaxPooling1D, ZeroPadding1D


from keras.utils import np_utils, plot_model, Sequence


# the squashing function.
# we use 0.5 in stead of 1 in hinton's paper.
# if 1, the norm of vector will be zoomed out.
# if 0.5, the norm will be zoomed in while original norm is less than 0.5
# and be zoomed out while original norm is greater than 0.5.
def squash(x, axis=-1):
	s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
	scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
	return scale * x


# define our own softmax function instead of K.softmax
# because K.softmax can not specify axis.
def softmax(x, axis=-1):
	ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
	return ex / K.sum(ex, axis=axis, keepdims=True)


# define the margin loss like hinge loss
def margin_loss(y_true, y_pred):
	lamb, margin = 0.5, 0.1
	return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
		1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)


class Capsule(Layer):
	"""A Capsule Implement with Pure Keras
	There are two vesions of Capsule.
	One is like dense layer (for the fixed-shape input),
	and the other is like timedistributed dense (for various length input).

	The input shape of Capsule must be (batch_size,
										input_num_capsule,
										input_dim_capsule
									   )
	and the output shape is (batch_size,
							 num_capsule,
							 dim_capsule
							)

	Capsule Implement is from https://github.com/bojone/Capsule/
	Capsule Paper: https://arxiv.org/abs/1710.09829
	"""

	def __init__(self,
				 num_capsule,
				 dim_capsule,
				 routings=3,
				 share_weights=True,
				 activation='squash',
				 **kwargs):
		super(Capsule, self).__init__(**kwargs)
		self.num_capsule = num_capsule
		self.dim_capsule = dim_capsule
		self.routings = routings
		self.share_weights = share_weights
		if activation == 'squash':
			self.activation = squash
		else:
			self.activation = activations.get(activation)

	def build(self, input_shape):
		input_dim_capsule = input_shape[-1]
		if self.share_weights:
			self.kernel = self.add_weight(
				name='capsule_kernel',
				shape=(1, input_dim_capsule,
					   self.num_capsule * self.dim_capsule),
				initializer='glorot_uniform',
				trainable=True)
		else:
			input_num_capsule = input_shape[-2]
			self.kernel = self.add_weight(
				name='capsule_kernel',
				shape=(input_num_capsule, input_dim_capsule,
					   self.num_capsule * self.dim_capsule),
				initializer='glorot_uniform',
				trainable=True)

	def call(self, inputs):
		"""Following the routing algorithm from Hinton's paper,
		but replace b = b + <u,v> with b = <u,v>.

		This change can improve the feature representation of Capsule.

		However, you can replace
			b = K.batch_dot(outputs, hat_inputs, [2, 3])
		with
			b += K.batch_dot(outputs, hat_inputs, [2, 3])
		to realize a standard routing.
		"""

		if self.share_weights:
			hat_inputs = K.conv1d(inputs, self.kernel)
		else:
			hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

		batch_size = K.shape(inputs)[0]
		input_num_capsule = K.shape(inputs)[1]
		hat_inputs = K.reshape(hat_inputs,
							   (batch_size, input_num_capsule,
								self.num_capsule, self.dim_capsule))
		hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

		b = K.zeros_like(hat_inputs[:, :, :, 0])
		for i in range(self.routings):
			c = softmax(b, 1)
			o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
			if i < self.routings - 1:
				b = K.batch_dot(o, hat_inputs, [2, 3])
				if K.backend() == 'theano':
					o = K.sum(o, axis=1)

		return o

	def compute_output_shape(self, input_shape):
		return (None, self.num_capsule, self.dim_capsule)


def capsule_net(classes=5):

	input_image = Input(shape=(None, None, 3))
	x = Conv2D(64, (3, 3), activation='relu')(input_image)
	x = Conv2D(64, (3, 3), activation='relu')(x)
	x = AveragePooling2D((2, 2))(x)
	x = Conv2D(128, (3, 3), activation='relu')(x)
	x = Conv2D(128, (3, 3), activation='relu')(x)
	x = Reshape((-1, 128))(x)
	capsule = Capsule(classes, 16, 3, True)(x)
	output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)

	output = Flatten()(output)
	output = Dense(classes, activation='softmax')(output)

	model = Model(inputs=input_image, outputs=output)

	

	# we use a margin loss
	# model.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])
	model.summary()

	return model

# capsule_net()