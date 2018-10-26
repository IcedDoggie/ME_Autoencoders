import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
from keras import backend as K
from keras.callbacks import EarlyStopping

from utilities import loading_smic_table, loading_samm_table, loading_casme_table
from utilities import class_merging, read_image, create_generator_LOSO, class_discretization
from utilities import LossHistory, record_loss_accuracy
from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall, sklearn_macro_f1
from siamese_models import siamese_vgg16_imagenet
from siamese_models import siamese_dual_loss, create_siamese_pairs
from evaluationmatrix import majority_vote, temporal_predictions_averaging

# TODO
# function to create pairs ( think a bit first), true pair, false pair. NEED TO CREATE PAIRS

def train(type_of_test, train_id, net, feature_type = 'grayscale', db='Combined_Dataset_Apex', spatial_size = 224, tf_backend_flag = False):

	sys.setrecursionlimit(10000)
	# /media/ice/OS/Datasets/Combined_Dataset_Apex/CASME2_TIM10/CASME2_TIM10	
	# general variables and path
	working_dir = '/home/ice/Documents/ME_Autoencoders/'
	root_dir = '/media/ice/OS/Datasets/' + db + '/'
	weights_path = '/media/ice/OS/Datasets/'
	if os.path.isdir(weights_path + 'Weights/'+ str(train_id) ) == False:
		os.mkdir(weights_path + 'Weights/'+ str(train_id) )	

	weights_path = weights_path + "Weights/" + train_id + "/"

	if feature_type == 'grayscale':
		casme2_db = 'CASME2_TIM10'
		samm_db = 'SAMM_TIM10'
		smic_db = 'SMIC_TIM10'
		timesteps_TIM = 1
	elif feature_type == 'flow':
		casme2_db = 'CASME2_Optical'
		samm_db = 'SAMM_Optical'
		smic_db = 'SMIC_Optical'
		timesteps_TIM = 1	

	classes = 5
	spatial_size = spatial_size
	channels = 3
	data_dim = 4096
	tot_mat = np.zeros((classes, classes))

	# labels reading (ori)
	casme2_table = loading_casme_table(root_dir, casme2_db)
	casme2_table = class_discretization(casme2_table, 'CASME_2')
	casme_list, casme_labels = read_image(root_dir, casme2_db, casme2_table)
	total_list = casme_list
	total_labels = casme_labels

	# labels reading (augmented)
	casme2_aug_table = loading_casme_table(root_dir, "CASME_Micro_Augmented")
	casme2_aug_table = class_discretization(casme2_aug_table, 'CASME_Micro_Augmented')
	casme_aug_list, casme_aug_labels = read_image(root_dir, "CASME_Micro_Augmented", casme2_aug_table)
	total_aug_list = casme_aug_list
	total_aug_labels = casme_aug_labels


	pred = []
	y_list = []

	# training configuration
	learning_rate = 0.0001
	history = LossHistory()
	stopping = EarlyStopping(monitor='loss', min_delta = 0, mode = 'min', patience=5)
	sgd = optimizers.SGD(lr=learning_rate, decay=1e-7, momentum=0.9, nesterov=True)
	adam = optimizers.Adam(lr=learning_rate, decay=learning_rate * 2)
	batch_size = 30
	epochs = 1
	total_samples = 0

	if os.path.exists(weights_path) == False:
		os.mkdir(weights_path) 


	for sub in range(len(total_list)):

		# model initialization for LOSO 
		model = siamese_vgg16_imagenet()
		# Losses will be summed up
		model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=adam, metrics=[metrics.categorical_accuracy])
		# model.compile(loss=siamese_dual_loss, optimizer=adam, metrics=[metrics.categorical_accuracy])

		loso_generator = create_generator_LOSO(total_list, total_labels, classes, sub, net, spatial_size = spatial_size, train_phase='svc')
		loso_generator_aug = create_generator_LOSO(total_aug_list, total_aug_labels, classes, sub, net, spatial_size = spatial_size, train_phase='svc')

		for (alpha, beta) in zip(loso_generator, loso_generator_aug):
			X, y, non_binarized_y = alpha[0], alpha[1], alpha[2]
			X_aug, y_aug, non_binarized_y_aug = beta[0], beta[1], beta[2]
			pairs, labels_pairs = create_siamese_pairs(X, X_aug, y, y_aug)

		regress_zero = np.zeros(shape = (labels_pairs[:, 0, :].shape[0], 1))	

		# model.fit([pairs[:, 0, :, :, :], pairs[:, 1, :, :, :]], [labels_pairs[:, 0, :], ], batch_size=batch_size, epochs=epochs, callbacks=[stopping], shuffle=False)
		plot_model(model, show_shapes=True, to_file='vgg16_out_of_siamese.png')
		model.fit([pairs[:, 0, :, :, :], pairs[:, 1, :, :, :]], [labels_pairs[:, 0, :], regress_zero], batch_size=batch_size, epochs=epochs, shuffle=False)

		weights_name = weights_path + str(sub) + '.h5'
		model.save_weights(weights_name)

		test_loso_generator = create_generator_LOSO(total_list, total_labels, classes, sub, net, spatial_size = spatial_size, train_phase = False)


		for X, y, non_binarized_y in test_loso_generator:
			# Spatial Encoding
			model = Model(inputs=model.layers[0].input, outputs=model.layers[3].output)
			# vgg16.load_weights(weights_name)
			plot_model(model, to_file='test_stage', show_shapes=True)
			predicted_class = model.predict(X, batch_size = batch_size)
			predicted_class = np.argmax(predicted_class, axis=1)
			if tf_backend_flag == True:

				spatial_features = np.reshape(spatial_features, (spatial_features.shape[0], spatial_features.shape[-1]))

			# predicted_class = majority_vote(predicted_class, X, batch_size, timesteps_TIM)
			# predicted_class = temporal_predictions_averaging(predicted_class, timesteps_TIM)

			non_binarized_y = non_binarized_y[0]
			non_binarized_y = non_binarized_y[::timesteps_TIM]

			print(predicted_class)
			print(non_binarized_y)	

			# for sklearn macro f1 calculation
			for counter in range(len(predicted_class)):
				pred += [predicted_class[counter]]
				y_list += [non_binarized_y[counter]]


			ct = confusion_matrix(non_binarized_y, predicted_class)
			order = np.unique(np.concatenate((predicted_class, non_binarized_y)))	
			mat = np.zeros((classes, classes))
			for m in range(len(order)):
				for n in range(len(order)):
					mat[int(order[m]), int(order[n])] = ct[m, n]
				   
			tot_mat = mat + tot_mat

			[f1, precision, recall] = fpr(tot_mat, classes)
			file = open(root_dir + 'Classification/' + 'Result/'+ db + '/f1_' + str(train_id) +  '.txt', 'a')
			file.write(str(f1) + "\n")
			file.close()
			total_samples += len(non_binarized_y)
			war = weighted_average_recall(tot_mat, classes, total_samples)
			uar = unweighted_average_recall(tot_mat, classes)
			macro_f1, weighted_f1 = sklearn_macro_f1(y_list, pred)
			print("war: " + str(war))
			print("uar: " + str(uar))
			print("Macro_f1: " + str(macro_f1))
			print("Weighted_f1: " + str(weighted_f1))	
def test(type_of_test, train_id, net, feature_type = 'grayscale', db='Combined_Dataset_Apex', spatial_size = 224, tf_backend_flag = False):

	sys.setrecursionlimit(10000)
	# /media/ice/OS/Datasets/Combined_Dataset_Apex/CASME2_TIM10/CASME2_TIM10	
	# general variables and path
	working_dir = '/home/viprlab/Documents/ME_Autoencoders/'
	root_dir = '/media/viprlab/01D31FFEF66D5170/Ice/' + db + '/'
	weights_path = '/media/viprlab/01D31FFEF66D5170/Ice/'
	if os.path.isdir(weights_path + 'Weights/'+ str(train_id) ) == False:
		os.mkdir(weights_path + 'Weights/'+ str(train_id) )	

	weights_path = weights_path + "Weights/" + train_id + "/"

	if feature_type == 'grayscale':
		casme2_db = 'CASME2_TIM10'
		samm_db = 'SAMM_TIM10'
		smic_db = 'SMIC_TIM10'
		timesteps_TIM = 1
	elif feature_type == 'flow':
		casme2_db = 'CASME2_Optical'
		samm_db = 'SAMM_Optical'
		smic_db = 'SMIC_Optical'
		timesteps_TIM = 1	

	classes = 5
	spatial_size = spatial_size
	channels = 3
	data_dim = 4096
	tot_mat = np.zeros((classes, classes))

	# labels reading (ori)
	casme2_table = loading_casme_table(root_dir, casme2_db)
	casme2_table = class_discretization(casme2_table, 'CASME_2')
	casme_list, casme_labels = read_image(root_dir, casme2_db, casme2_table)
	total_list = casme_list
	total_labels = casme_labels

	# labels reading (augmented)
	casme2_aug_table = loading_casme_table(root_dir, "CASME_Micro_Augmented")
	casme2_aug_table = class_discretization(casme2_aug_table, 'CASME_Micro_Augmented')
	casme_aug_list, casme_aug_labels = read_image(root_dir, "CASME_Micro_Augmented", casme2_aug_table)
	total_aug_list = casme_aug_list
	total_aug_labels = casme_aug_labels


	pred = []
	y_list = []

	# training configuration
	learning_rate = 0.0001
	history = LossHistory()
	stopping = EarlyStopping(monitor='loss', min_delta = 0, mode = 'min', patience=5)
	sgd = optimizers.SGD(lr=learning_rate, decay=1e-7, momentum=0.9, nesterov=True)
	adam = optimizers.Adam(lr=learning_rate, decay=learning_rate * 2)
	batch_size = 30
	epochs = 1
	total_samples = 0

	if os.path.exists(weights_path) == False:
		os.mkdir(weights_path) 


	for sub in range(len(total_list)):

		# model initialization for LOSO 
		weights_name = weights_path + str(sub) + '.h5'
		model = siamese_vgg16_imagenet()
		model.load_weights(weights_name)
		# Losses will be summed up
		model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=adam, metrics=[metrics.categorical_accuracy])


		test_loso_generator = create_generator_LOSO(total_list, total_labels, classes, sub, net, spatial_size = spatial_size, train_phase = False)


		for X, y, non_binarized_y in test_loso_generator:
			# Spatial Encoding
			model = Model(inputs=model.layers[0].input, outputs=model.layers[3].output)
			# vgg16.load_weights(weights_name)
			plot_model(model, to_file='test_stage', show_shapes=True)
			predicted_class = model.predict(X, batch_size = batch_size)
			predicted_class = np.argmax(predicted_class, axis=1)
			if tf_backend_flag == True:

				spatial_features = np.reshape(spatial_features, (spatial_features.shape[0], spatial_features.shape[-1]))

			# predicted_class = majority_vote(predicted_class, X, batch_size, timesteps_TIM)
			# predicted_class = temporal_predictions_averaging(predicted_class, timesteps_TIM)

			non_binarized_y = non_binarized_y[0]
			non_binarized_y = non_binarized_y[::timesteps_TIM]

			print(predicted_class)
			print(non_binarized_y)	

			# for sklearn macro f1 calculation
			for counter in range(len(predicted_class)):
				pred += [predicted_class[counter]]
				y_list += [non_binarized_y[counter]]


			ct = confusion_matrix(non_binarized_y, predicted_class)
			order = np.unique(np.concatenate((predicted_class, non_binarized_y)))	
			mat = np.zeros((classes, classes))
			for m in range(len(order)):
				for n in range(len(order)):
					mat[int(order[m]), int(order[n])] = ct[m, n]
				   
			tot_mat = mat + tot_mat

			[f1, precision, recall] = fpr(tot_mat, classes)
			file = open(root_dir + 'Classification/' + 'Result/'+ db + '/f1_' + str(train_id) +  '.txt', 'a')
			file.write(str(f1) + "\n")
			file.close()
			total_samples += len(non_binarized_y)
			war = weighted_average_recall(tot_mat, classes, total_samples)
			uar = unweighted_average_recall(tot_mat, classes)
			macro_f1, weighted_f1 = sklearn_macro_f1(y_list, pred)
			print("war: " + str(war))
			print("uar: " + str(uar))
			print("Macro_f1: " + str(macro_f1))
			print("Weighted_f1: " + str(weighted_f1))
train(siamese_vgg16_imagenet, train_id='test_siam', net = 'vgg', feature_type='grayscale', db='Siamese Macro-Micro', spatial_size = 224, tf_backend_flag = False)