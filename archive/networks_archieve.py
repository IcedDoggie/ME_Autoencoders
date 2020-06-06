def train_sssn_lrcn(timesteps_TIM, classes):

	x = Input(shape=(10, 3, 227, 227))

	model = train_shallow_alexnet_imagenet(classes=classes)
	model = Model(inputs=model.input, outputs=model.layers[-4].output)
	model = train_res50_imagenet()
	plot_model(model, show_shapes=True, to_file='train_sssn_lrcn.png')

	for layer_counter in range(len(model.layers) - 1):

		# print(model.layers[layer_counter + 1])
		# print(type(model.layers[layer_counter + 1]))
		# # model.layers[layer_counter + 1] = TimeDistributed(model.layers[layer_counter + 1])
		# if type(model.layers[layer_counter + 1] == 'keras.layers.merge.Concatenate'):
		# 	x = TimeDistributed(model.layers[layer_counter + 1])(splittensor(ratio_split=2,id_split=[0, 1])(x))
		# 	# x = concatenate([
		# 	# 	Conv2D(128, (5, 5), activation="relu", kernel_initializer='he_normal', bias_initializer='he_normal', name='conv_2_'+str(i+1), dilation_rate=2)(
		# 	# 	splittensor(ratio_split=2,id_split=i)(x)
		# 	# 	) for i in range(2)], axis=1, name="conv_2")
		# 	# (splittensor(ratio_split=2,id_split=i)(x)) for i in range(2)], axis=1, name="conv_2")		
		# else:			
		# 	x = TimeDistributed(model.layers[layer_counter + 1])(x)
		x = TimeDistributed(model.layers[layer_counter + 1])(x)	
	print(K.int_shape(x))
	model = Model(inputs=input_frame, outputs=model.output)
	plot_model(model, show_shapes=True, to_file='train_sssn_lrcn_TIME_DISTRIBUTED.png')
	print(model.summary())
	# encoded_frame = TimeDistributed(Lambda( Flatten() ))(input_frame)
	# encoded_frame = TimeDistributed(Lambda(Dense(3000, activation='relu')))(encoded_frame)
	# print(K.print_tensor(encoded_frame))
	# model = Model(inputs=input_frame, outputs=encoded_frame)
	return model

	# encoded_frame = K.reshape(encoded_frame, ())
	# lstm1 = LSTM(3000, return_sequences=False)(encoded_frame)
	# model = Model(inputs=input_frame, outputs=lstm1.output)
	# plot_model(model, show_shapes=True, to_file='train_sssn_lrcn.png')

	# lstm1 = LSTM(3000, return_sequences=False)(encoded_frame)

def train_convolutional_latent_features(classes = 5, spatial_size = 227):
	input_data = Input(shape=(3, spatial_size, spatial_size))

	# Encoder
	conv_2 = Conv2D(64, (5, 5), strides=(1, 1), activation='relu', name='conv_1', kernel_initializer='he_normal', bias_initializer='he_normal')(input_data)
	conv_2 = MaxPooling2D((3, 3), strides=(1, 1))(conv_2)
	conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
	conv_2 = ZeroPadding2D((2,2))(conv_2)

	conv_2 = Conv2D(16, (11, 11), strides=(2, 2), activation='relu', name='conv_2', kernel_initializer='he_normal', bias_initializer='he_normal')(conv_2)
	conv_2 = MaxPooling2D((3, 3), strides=(1, 1))(conv_2)
	conv_2 = crosschannelnormalization(name="convpool_2")(conv_2)
	conv_2 = ZeroPadding2D((2,2))(conv_2)

	conv_2 = Conv2D(8, (11, 11), strides=(2, 2), activation='relu', name='conv_5', kernel_initializer='he_normal', bias_initializer='he_normal')(conv_2)
	conv_2 = MaxPooling2D((3, 3), strides=(1, 1))(conv_2)
	conv_2 = crosschannelnormalization(name="convpool_5")(conv_2)
	conv_2 = ZeroPadding2D((2,2))(conv_2)	

	conv_2 = Conv2D(1, (11, 11), strides=(3, 3), activation='relu', name='conv_6', kernel_initializer='he_normal', bias_initializer='he_normal')(conv_2)
	conv_2 = MaxPooling2D((3, 3), strides=(1, 1))(conv_2)
	conv_2 = crosschannelnormalization(name="convpool_6")(conv_2)
	conv_2 = ZeroPadding2D(((3, 2), (3, 2)), name='latent_features')(conv_2)		



	model = Model(inputs = input_data, outputs = conv_2)
	plot_model(model, to_file = 'train_convolutional_latent_features', show_shapes=True)
	print(model.summary())
	return model

def train_tri_stream_shallow_alexnet_pooling_merged_latent_features(classes=5, freeze_flag=None):
	input_gray = Input(shape=(3, 227, 227))
	input_mag = Input(shape=(3, 227, 227))
	input_strain = Input(shape=(3, 227, 227))

	model_mag = train_shallow_alexnet_imagenet(classes = classes)
	model_strain = train_shallow_alexnet_imagenet(classes = classes)
	model_mag = Model(inputs = model_mag.input, outputs = model_mag.layers[-5].output)
	model_strain = Model(inputs = model_strain.input, outputs = model_strain.layers[-5].output)


	model_gray = train_convolutional_latent_features(classes = classes)	
	auto_feat = model_gray(input_gray)
	duplicated_feat = Lambda(repeat_element_autofeat, output_shape=(256, 17, 17))(auto_feat)


	mag_feat = model_mag(input_mag)
	strain_feat = model_strain(input_strain)

	print(duplicated_feat)
	print(K.int_shape(duplicated_feat))
	# concatenate via multipling the padded convolutions
	concat = Multiply()([duplicated_feat, mag_feat, strain_feat])
	concat = Flatten()(concat)
	dropout = Dropout(0.5)(concat)

	dense_1 = Dense(classes, kernel_initializer = 'he_normal', bias_initializer = 'he_normal', name='last_fc')(dropout)
	prediction = Activation("softmax", name = 'softmax_activate')(dense_1)

	# Decoder
	conv_2 = Conv2D(1, (11, 11), strides=(2, 2), activation='relu', name='conv_7', kernel_initializer='he_normal', bias_initializer='he_normal')(auto_feat)
	conv_2 = UpSampling2D((5, 5))(conv_2)
	conv_2 = crosschannelnormalization(name="convpool_7")(conv_2)
	conv_2 = ZeroPadding2D((2,2))(conv_2)	

	conv_2 = Conv2D(8, (11, 11), strides=(2, 2), activation='relu', name='conv_8', kernel_initializer='he_normal', bias_initializer='he_normal')(conv_2)
	conv_2 = UpSampling2D((3, 3))(conv_2)
	conv_2 = crosschannelnormalization(name="convpool_8")(conv_2)
	conv_2 = ZeroPadding2D((2,2))(conv_2)

	conv_2 = Conv2D(16, (11, 11), strides=(1, 1), activation='relu', name='conv_3', kernel_initializer='he_normal', bias_initializer='he_normal')(conv_2)
	conv_2 = UpSampling2D((4, 4))(conv_2)
	conv_2 = crosschannelnormalization(name="convpool_3")(conv_2)

	conv_2 = Conv2D(64, (5, 5), strides=(1, 1), activation='relu', name='conv_4', kernel_initializer='he_normal', bias_initializer='he_normal')(conv_2)
	conv_2 = UpSampling2D((4, 4))(conv_2)
	conv_2 = crosschannelnormalization(name="convpool_4")(conv_2)
	conv_2 = ZeroPadding2D((2,2), name = 'decoder_feat')(conv_2)
	recon_out = Conv2D(3, (2, 2), strides=(1, 1), activation='relu', name='conv_5', kernel_initializer='he_normal', bias_initializer='he_normal')(conv_2)


	model = Model(inputs = [input_mag, input_strain, input_gray], outputs = [prediction, recon_out])
	plot_model(model, to_file = 'train_tri_stream_shallow_alexnet_pooling_merged_latent_features', show_shapes=True)
	print(model.summary())

	return model

def train_dual_stream_with_auxiliary_attention_networks(classes=5, freeze_flag=None):

	input_mag = Input(shape=(3, 227, 227))
	input_strain = Input(shape=(3, 227, 227))
	input_gray = Input(shape=(3, 227, 227))
	model_mag = train_shallow_alexnet_imagenet(classes = classes)
	model_strain = train_shallow_alexnet_imagenet(classes = classes)
	aux_model_grayscale = train_alexnet_imagenet(classes = classes)

	model_mag = Model(inputs = model_mag.input, outputs = model_mag.layers[-5].output)
	model_strain = Model(inputs = model_strain.input, outputs = model_strain.layers[-5].output)
	aux_model_grayscale = Model(inputs = aux_model_grayscale.input, outputs = aux_model_grayscale.layers[-8].output)

	pooled_mag = model_mag(input_mag)
	pooled_strain = model_strain(input_strain)
	pooled_grayscale = aux_model_grayscale(input_gray)	
	padding_attention = ZeroPadding2D(padding=(2, 2), data_format="channels_first")(pooled_grayscale)

	plot_model(model_mag, to_file = 'mag_model', show_shapes=True)
	plot_model(model_strain, to_file = 'strain_model', show_shapes=True)
	plot_model(aux_model_grayscale, to_file = 'aux_model_grayscale', show_shapes=True)

	# concatenate
	concat = Multiply()([pooled_mag, pooled_strain])
	concat = Multiply()([concat, padding_attention])
	concat = Flatten()(concat)
	#concat = Lambda(l2_normalize, output_shape=l2_normalize_output_shape)(concat)
	dropout = Dropout(0.5)(concat)

	dense_1 = Dense(classes, kernel_initializer = 'he_normal', bias_initializer = 'he_normal', name='last_fc')(dropout)
	prediction = Activation("softmax", name = 'softmax_activate')(dense_1)


	model = Model(inputs = [input_mag, input_strain, input_gray], outputs = prediction)
	plot_model(model, to_file = 'train_dual_stream_with_auxiliary_attention_networks', show_shapes=True)
	print(model.summary())

	return model	

def train_dual_stream_with_auxiliary_attention_networks_dual_loss(classes=5, freeze_flag=None):

	input_mag = Input(shape=(3, 227, 227))
	input_strain = Input(shape=(3, 227, 227))
	input_gray = Input(shape=(3, 227, 227))
	model_mag = train_shallow_alexnet_imagenet(classes = classes)
	model_strain = train_shallow_alexnet_imagenet(classes = classes)
	aux_model_grayscale = train_alexnet_imagenet(classes = classes)

	softmax_grayscale = aux_model_grayscale(input_gray)

	model_mag = Model(inputs = model_mag.input, outputs = model_mag.layers[-5].output)
	model_strain = Model(inputs = model_strain.input, outputs = model_strain.layers[-5].output)
	# aux_model_grayscale = Model(inputs = aux_model_grayscale.layers[0].get_input_at(0), outputs = aux_model_grayscale.layers[-8].get_output_at(1))
	aux_grayscale = aux_model_grayscale.layers[-8].output

	pooled_mag = model_mag(input_mag)
	pooled_strain = model_strain(input_strain)
	# pooled_grayscale = aux_grayscale(input_gray)	
	padding_attention = ZeroPadding2D(padding=(2, 2), data_format="channels_first")(aux_grayscale)

	plot_model(model_mag, to_file = 'mag_model', show_shapes=True)
	plot_model(model_strain, to_file = 'strain_model', show_shapes=True)
	plot_model(aux_model_grayscale, to_file = 'aux_model_grayscale', show_shapes=True)

	# concatenate
	concat = Multiply()([pooled_mag, pooled_strain])
	concat = Multiply()([concat, padding_attention])
	concat = Flatten()(concat)
	#concat = Lambda(l2_normalize, output_shape=l2_normalize_output_shape)(concat)
	dropout = Dropout(0.5)(concat)

	dense_1 = Dense(classes, kernel_initializer = 'he_normal', bias_initializer = 'he_normal', name='last_fc')(dropout)
	prediction = Activation("softmax", name = 'softmax_activate')(dense_1)


	model = Model(inputs = [input_mag, input_strain, input_gray], outputs = [prediction])
	plot_model(model, to_file = 'train_dual_stream_with_auxiliary_attention_networks_dual_loss', show_shapes=True)
	print(model.summary())

	return model	

def train_tri_stream_shallow_alexnet_pooling_merged_slow_fusion(classes=5, freeze_flag=None):	
	input_gray = Input(shape=(3, 227, 227))
	input_mag = Input(shape=(3, 227, 227))
	input_strain = Input(shape=(3, 227, 227))

	model_gray = train_shallow_alexnet_imagenet(classes = classes)
	model_mag = train_shallow_alexnet_imagenet(classes = classes)
	model_strain = train_shallow_alexnet_imagenet(classes = classes)
	model_gray_first_level = Model(inputs = model_gray.input, outputs = model_gray.layers[-9].output)
	model_mag_first_level = Model(inputs = model_mag.input, outputs = model_mag.layers[-9].output)
	model_strain_first_level = Model(inputs = model_strain.input, outputs = model_strain.layers[-9].output)


	gray_first_level = model_gray_first_level(input_gray)
	mag_first_level = model_mag_first_level(input_mag)
	strain_first_level = model_strain_first_level(input_strain)

	concat_gray_mag = Multiply()([gray_first_level, mag_first_level])
	concat_mag_strain = Multiply()([mag_first_level, strain_first_level])
	concat_strain_gray = Multiply()([strain_first_level, gray_first_level])

	# 2nd level conv and merging
	conv_2 = Conv2D(256, (5, 5), strides=(1, 1), activation='relu', name='conv_2', kernel_initializer='he_normal', bias_initializer='he_normal')(concat_gray_mag)
	conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
	conv_2 = crosschannelnormalization(name="convpool_2")(conv_2)
	conv_2 = ZeroPadding2D((2,2))(conv_2)

	conv_3 = Conv2D(256, (5, 5), strides=(1, 1), activation='relu', name='conv_3', kernel_initializer='he_normal', bias_initializer='he_normal')(concat_mag_strain)
	conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_3)
	conv_3 = crosschannelnormalization(name="convpool_3")(conv_3)
	conv_3 = ZeroPadding2D((2,2))(conv_3)

	conv_4 = Conv2D(256, (5, 5), strides=(1, 1), activation='relu', name='conv_4', kernel_initializer='he_normal', bias_initializer='he_normal')(concat_strain_gray)
	conv_4 = MaxPooling2D((3, 3), strides=(2, 2))(conv_4)
	conv_4 = crosschannelnormalization(name="convpool_4")(conv_4)
	conv_4 = ZeroPadding2D((2,2))(conv_4)	

	concat_all = Multiply()([conv_2, conv_3, conv_4])

	flatten = Flatten(name="flatten")(concat_all)
	dropout = Dropout(0.5)(flatten)
	dense_1 = Dense(classes, kernel_initializer = 'he_normal', bias_initializer = 'he_normal', name='last_fc')(dropout)
	prediction = Activation("softmax", name = 'softmax_activate')(dense_1)

	model = Model(inputs = [input_gray, input_mag, input_strain], outputs = prediction)		
	print(model.summary())
	plot_model(model, to_file='train_tri_stream_shallow_alexnet_pooling_merged_slow_fusion', show_shapes=True)


	return model

def train_shallow_alexnet_imagenet_FCN(classes = 5, freeze_flag = None):
	model = alexnet(input_shape = (3, 227, 227), nb_classes = 1000, mean_flag = True)
	model.load_weights('alexnet_weights.h5')

	# modify architecture
	last_conv_1 = model.layers[5].output
	conv_2 = Conv2D(256, (5, 5), strides=(1, 1), activation='relu', name='conv_2', kernel_initializer='he_normal', bias_initializer='he_normal')(last_conv_1)
	conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
	conv_2 = crosschannelnormalization(name="convpool_2")(conv_2)
	conv_2 = ZeroPadding2D((2,2))(conv_2)

	conv_2 = Dropout(0.5)(conv_2)
	
	conv_activate = Conv2D(classes, kernel_size=(1, 1), strides = (1, 1), activation = 'relu', kernel_initializer = 'he_normal', bias_initializer = 'he_normal', name='conv_activate')(conv_2)	
	conv_activate = GlobalAveragePooling2D(data_format = 'channels_first')(conv_activate)

	model = Model(inputs = model.input, outputs = conv_activate)
	plot_model(model, to_file='shallowalex_fcn', show_shapes =True)
	print(model.summary())

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

		