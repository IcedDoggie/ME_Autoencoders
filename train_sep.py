import numpy as np
import sys
import math
import operator
import csv
import glob,os
import xlrd
import cv2
import pandas as pd
import pydot, graphviz

from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import confusion_matrix
import scipy.io as sio

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from keras import optimizers, metrics
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.utils import np_utils, plot_model

from utilities import loading_smic_table, loading_samm_table, loading_casme_table
from utilities import class_merging, read_image, create_generator_LOSO, create_generator_nonLOSO
from utilities import LossHistory, record_loss_accuracy
from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall
from models import VGG_16, temporal_module, layer_wise_conv_autoencoder, layer_wise_autoencoder


def train_sep():
	# general variables and path
	working_dir = '/home/ice/Documents/ME_Autoencoders/'
	root_dir = '/media/ice/OS/Datasets/Combined Dataset/'
	train_id = "new_code"
	weights_dir = root_dir + "Weights/" + train_id + "/"


	casme2_db = 'CASME2_TIM10'
	samm_db = 'SAMM_TIM10'
	smic_db = 'SMIC_TIM10'
	classes = 3
	spatial_size = 224
	channels = 3
	timesteps_TIM = 10
	data_dim = 4096
	tot_mat = np.zeros((classes, classes))

	# labels reading
	casme2_table = loading_casme_table(root_dir, casme2_db)
	samm_table, _ = loading_samm_table(root_dir, samm_db, objective_flag=0)
	smic_table = loading_smic_table(root_dir, smic_db)
	casme2_table = class_merging(casme2_table)
	samm_table = class_merging(samm_table)
	smic_table = smic_table[0]

	# images reading, read according to table
	samm_list, samm_labels = read_image(root_dir, samm_db, samm_table)
	smic_list, smic_labels = read_image(root_dir, smic_db, smic_table)
	casme_list, casme_labels = read_image(root_dir, casme2_db, casme2_table)

	total_list = samm_list + smic_list + casme_list
	total_labels = samm_labels + smic_labels + casme_labels
	

	# training configuration
	learning_rate = 0.0001
	history = LossHistory()
	stopping = EarlyStopping(monitor='loss', min_delta = 0, mode = 'min', patience=5)
	sgd = optimizers.SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)
	adam = optimizers.Adam(lr=learning_rate, decay=learning_rate * 2)
	batch_size = 1
	epochs = 1

	if os.path.exists(weights_dir) == False:
		os.mkdir(weights_dir) 


	datagen = ImageDataGenerator(

		)

	# Train
	total_list = samm_list + casme_list
	total_labels = samm_labels + casme_labels

	vgg_model = VGG_16(spatial_size, classes, channels, weights_path = 'VGG_Face_Deep_16.h5')	
	vgg_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
	# freeze model
	for layer in vgg_model.layers[:33]:
		layer.trainable = False
	temporal_model = temporal_module(data_dim, timesteps_TIM, classes)
	temporal_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
	loso_generator = create_generator_nonLOSO(total_list, total_labels, classes)

	for X, y in loso_generator:
		vgg_model.fit_generator(datagen.flow(X, y, batch_size = batch_size, shuffle=True), steps_per_epoch = int(len(X)/batch_size), epochs = epochs, callbacks=[history, stopping])
		vgg_model_encoder = Model(inputs = vgg_model.input, outputs = vgg_model.layers[35].output)
		record_loss_accuracy(root_dir, train_id, "Combined Dataset", history)
		plot_model(vgg_model, to_file = "VGG16_Auto_sep.png", show_shapes=True)	

		# Temporal Learning
		y = y[::10, :]
		spatial_features = vgg_model_encoder.predict(X, batch_size = batch_size)
		spatial_features = spatial_features.reshape(int(X.shape[0]/timesteps_TIM), timesteps_TIM, spatial_features.shape[1])
		temporal_model.fit(spatial_features, y, batch_size = batch_size, epochs = epochs)

	# save weights
	spatial_weights_name = weights_dir + "vgg16_sep_" + str(sub) + ".h5"
	temporal_weights_name = weights_dir + "temporal_sep_" + str(sub) + ".h5"		
	vgg_model.save_weights(spatial_weights_name)
	temporal_model.save_weights(temporal_weights_name)

	# Resource Clear up
	del X, y


	# Test
	total_list = smic_list
	total_labels = smic_labels

	test_loso_generator = create_generator_nonLOSO(total_list, total_labels, classes, train_phase = False)
	for X, y, non_binarized_y in test_loso_generator:
		# Spatial Encoding

		spatial_features = vgg_model_encoder.predict(X, batch_size = batch_size)
		spatial_features = spatial_features.reshape(int(X.shape[0]/timesteps_TIM), timesteps_TIM, spatial_features.shape[1])
		predicted_class = temporal_model.predict_classes(spatial_features, batch_size = batch_size)

		non_binarized_y = non_binarized_y[0]
		non_binarized_y = non_binarized_y[::10]

		print(predicted_class)
		print(non_binarized_y)			
		ct = confusion_matrix(non_binarized_y, predicted_class)
		order = np.unique(np.concatenate((predicted_class, non_binarized_y)))	
		mat = np.zeros((classes, classes))
		for m in range(len(order)):
			for n in range(len(order)):
				mat[int(order[m]), int(order[n])] = ct[m, n]
				   
		tot_mat = mat + tot_mat

		[f1, precision, recall] = fpr(tot_mat, classes)
		file = open(root_dir + 'Classification/' + 'Result/'+ 'Combined Dataset' + '/f1_' + str(train_id) +  '.txt', 'a')
		file.write(str(f1) + "\n")
		file.close()
		war = weighted_average_recall(tot_mat, classes, len(X))
		uar = unweighted_average_recall(tot_mat, classes)
		print("war: " + str(war))
		print("uar: " + str(uar))

	# Resource CLear up
	del X, y, non_binarized_y, vgg_model, temporal_model

train_sep()