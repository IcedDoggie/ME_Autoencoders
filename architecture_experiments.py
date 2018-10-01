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
from utilities import class_merging, read_image, create_generator_LOSO
from utilities import LossHistory, record_loss_accuracy
from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall
from models import VGG_16, temporal_module, layer_wise_conv_autoencoder, layer_wise_autoencoder, convolutional_autoencoder
from networks import test_res50_finetuned, test_vgg16_finetuned, test_inceptionv3_finetuned
from networks import train_res50_imagenet, train_vgg16_imagenet, train_inceptionv3_imagenet

def train_softmax(type_of_test, train_id, feature_type = 'grayscale', db='Combined Dataset', spatial_size = 224, tf_backend_flag = False):

	sys.setrecursionlimit(10000)
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
		timesteps_TIM = 10
	elif feature_type == 'flow':
		casme2_db = 'CASME2_Optical'
		samm_db = 'SAMM_Optical'
		smic_db = 'SMIC_Optical'
		timesteps_TIM = 9	

	classes = 3
	spatial_size = spatial_size
	channels = 3
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
	total_list = casme_list
	total_labels = casme_labels

	# training configuration
	learning_rate = 0.0001
	sgd = optimizers.SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)
	adam = optimizers.Adam(lr=learning_rate, decay=learning_rate * 2)
	stopping = EarlyStopping(monitor='loss', min_delta = 0, mode = 'min', patience=5)	
	batch_size  = 30
	epochs = 50
	total_samples = 0


	# backend
	if tf_backend_flag:
		K.set_image_dim_ordering('tf')	

	# pre-process input images and normalization
	for sub in range(len(total_list)):
		# model
		model = type_of_test()
		model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
		# clf = SVC(kernel = 'linear', C = 1, decision_function_shape='ovr')
		loso_generator = create_generator_LOSO(total_list, total_labels, classes, sub, spatial_size = spatial_size, train_phase='svc')

		# model freezing
		for layer in model.layers[:-2]:
			layer.trainable = False
		for X, y, non_binarized_y in loso_generator:
			model.fit(X, y, batch_size = batch_size, epochs = epochs, callbacks=[stopping])
			# spatial_features = model.predict(X, batch_size = batch_size)
			# clf.fit(spatial_features, non_binarized_y)


		# Resource Clear up
		del X, y

		# Test Time 
		test_loso_generator = create_generator_LOSO(total_list, total_labels, classes, sub, spatial_size = spatial_size, train_phase = False)


		for X, y, non_binarized_y in test_loso_generator:
			# Spatial Encoding
			predicted_class = model.predict(X, batch_size = batch_size)
			# predicted_class = clf.predict(spatial_features)

			non_binarized_y = non_binarized_y[0]

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
			file = open(root_dir + 'Classification/' + 'Result/'+ db + '/f1_' + str(train_id) +  '.txt', 'a')
			file.write(str(f1) + "\n")
			file.close()
			total_samples += len(non_binarized_y)
			war = weighted_average_recall(tot_mat, classes, total_samples)
			uar = unweighted_average_recall(tot_mat, classes)
			print("war: " + str(war))
			print("uar: " + str(uar))

		weights_name = weights_path + str(sub) + '.h5'
		model.save_weights(weights_name)
		# Resource CLear up
		del X, y, non_binarized_y	
	return f1, war, uar, tot_mat
def train(type_of_test, train_id, feature_type = 'grayscale', db='Combined Dataset', spatial_size = 224, tf_backend_flag = False):


	sys.setrecursionlimit(10000)
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
		timesteps_TIM = 10
	elif feature_type == 'flow':
		casme2_db = 'CASME2_Optical'
		samm_db = 'SAMM_Optical'
		smic_db = 'SMIC_Optical'
		timesteps_TIM = 9	

	classes = 3
	spatial_size = spatial_size
	channels = 3
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
	total_list = casme_list
	total_labels = casme_labels

	# training configuration
	learning_rate = 0.0001
	sgd = optimizers.SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)
	adam = optimizers.Adam(lr=learning_rate, decay=learning_rate * 2)
	stopping = EarlyStopping(monitor='loss', min_delta = 0, mode = 'min', patience=5)	
	batch_size  = 30
	epochs = 50
	total_samples = 0


	# backend
	if tf_backend_flag:
		K.set_image_dim_ordering('tf')	

	# pre-process input images and normalization
	for sub in range(len(total_list)):
		# model
		model = type_of_test()
		model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
		clf = SVC(kernel = 'linear', C = 1, decision_function_shape='ovr')
		loso_generator = create_generator_LOSO(total_list, total_labels, classes, sub, spatial_size = spatial_size, train_phase='svc')

		# model freezing
		for layer in model.layers[:-2]:
			layer.trainable = False
		for X, y, non_binarized_y in loso_generator:
			model.fit(X, y, batch_size = batch_size, epochs = epochs, callbacks=[stopping])
			model = Model(inputs = model.input, outputs = model.layers[-2].output)
			spatial_features = model.predict(X, batch_size = batch_size)
			clf.fit(spatial_features, non_binarized_y)


		# Resource Clear up
		del X, y

		# Test Time 
		test_loso_generator = create_generator_LOSO(total_list, total_labels, classes, sub, spatial_size = spatial_size, train_phase = False)


		for X, y, non_binarized_y in test_loso_generator:
			# Spatial Encoding
			spatial_features = model.predict(X, batch_size = batch_size)
			predicted_class = clf.predict(spatial_features)

			non_binarized_y = non_binarized_y[0]

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
			file = open(root_dir + 'Classification/' + 'Result/'+ db + '/f1_' + str(train_id) +  '.txt', 'a')
			file.write(str(f1) + "\n")
			file.close()
			total_samples += len(non_binarized_y)
			war = weighted_average_recall(tot_mat, classes, total_samples)
			uar = unweighted_average_recall(tot_mat, classes)
			print("war: " + str(war))
			print("uar: " + str(uar))

		weights_name = weights_path + str(sub) + '.h5'
		model.save_weights(weights_name)
		# Resource CLear up
		del X, y, non_binarized_y	
	return f1, war, uar, tot_mat
	

def test(type_of_test, train_id = 'vgg16_f', feature_type = 'grayscale', db='Combined Dataset', spatial_size = 224, tf_backend_flag = False):
	sys.setrecursionlimit(10000)
	# general variables and path
	working_dir = '/home/viprlab/Documents/ME_Autoencoders/'
	root_dir = '/media/viprlab/01D31FFEF66D5170/Ice/' + db + '/'
	train_id = '/media/viprlab/01D31FFEF66D5170/Ice/Weights/' + train_id + '/'
	weights_dir = root_dir + "Weights/" + train_id + "/"

	if feature_type == 'grayscale':
		casme2_db = 'CASME2_TIM10'
		samm_db = 'SAMM_TIM10'
		smic_db = 'SMIC_TIM10'
		timesteps_TIM = 10
	elif feature_type == 'flow':
		casme2_db = 'CASME2_Optical'
		samm_db = 'SAMM_Optical'
		smic_db = 'SMIC_Optical'
		timesteps_TIM = 9		
	classes = 3
	spatial_size = spatial_size
	channels = 3

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
	total_list = casme_list
	total_labels = casme_labels
	counter = 0
	for temp in total_list:
		for item in temp:
			counter+= 1

	print(len(total_labels))	

	# training configuration
	learning_rate = 0.0001
	sgd = optimizers.SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)
	adam = optimizers.Adam(lr=learning_rate, decay=learning_rate * 2)
	batch_size = 30
	epochs = 1	
	total_samples = 0

	# backend
	if tf_backend_flag:
		K.set_image_dim_ordering('tf')

	# pre-process input images and normalization
	for sub in range(len(total_list)):
		print(sub)
	# 	# model
		w_path = train_id + str(sub) + '.h5'
		model = type_of_test(w_path)
		model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
		clf = SVC(kernel = 'linear', C = 1, decision_function_shape='ovr')
		loso_generator = create_generator_LOSO(total_list, total_labels, classes, sub, spatial_size = spatial_size, train_phase='svc')

		for X, y, non_binarized_y in loso_generator:
			# non_binarized_y = non_binarized_y[0]
			spatial_features = model.predict(X, batch_size = batch_size)
			if tf_backend_flag:
				spatial_features = spatial_features.reshape(spatial_features.shape[0], spatial_features.shape[-1])

			clf.fit(spatial_features, non_binarized_y)


		# Resource Clear up
		del X, y

		# Test Time 
		test_loso_generator = create_generator_LOSO(total_list, total_labels, classes, sub, spatial_size = spatial_size, train_phase = 'False')
		

		for X, y, non_binarized_y in test_loso_generator:
			# Spatial Encoding
			spatial_features = model.predict(X, batch_size = batch_size)
			if tf_backend_flag:
				spatial_features = spatial_features.reshape(spatial_features.shape[0], spatial_features.shape[-1])

			predicted_class = clf.predict(spatial_features)

			non_binarized_y = non_binarized_y[0]
			# non_binarized_y = non_binarized_y[::10]

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

			total_samples += len(non_binarized_y)
			war = weighted_average_recall(tot_mat, classes, total_samples)
			uar = unweighted_average_recall(tot_mat, classes)
			print("war: " + str(war))
			print("uar: " + str(uar))

		# Resource CLear up
		del X, y, non_binarized_y	

	return f1, war, uar, tot_mat


# f1, war, uar, tot_mat = train(train_vgg16_imagenet, train_id = 'vgg16_g', feature_type = 'grayscale', db='Combined Dataset', spatial_size=224, tf_backend_flag = False)
# f1_2, war_2, uar_2, tot_mat_2 = train(train_res50_imagenet, train_id = 'res50_g', feature_type = 'grayscale', db='Combined Dataset', spatial_size=224, tf_backend_flag = False)
# f1_3, war_3, uar_3, tot_mat_3 = train(train_inceptionv3_imagenet, train_id = 'incepv3_g', feature_type = 'grayscale', db='Combined Dataset', spatial_size=299, tf_backend_flag = False)
# f1_3, war_3, uar_3, tot_mat_3 = test(test_inceptionv3_imagenet, feature_type = 'grayscale', db='Combined Dataset', spatial_size = 299, tf_backend_flag = False)
f1, war, uar, tot_mat = test(test_vgg16_imagenet, train_id = 'vgg16_f', feature_type = 'flow', db='Combined Flow', spatial_size=224, tf_backend_flag = False)
f1_2, war_2, uar_2, tot_mat_2 = test(test_res50_imagenet, train_id = 'res50_f', feature_type = 'flow', db='Combined Flow', spatial_size=224, tf_backend_flag = False)
f1_3, war_3, uar_3, tot_mat_3 = test(test_inceptionv3_imagenet, train_id = 'incepv3_f', feature_type = 'flow', db='Combined Flow', spatial_size=299, tf_backend_flag = False)


print("RESULTS FOR VGG 16")
print("F1: " + str(f1))
print("war: " + str(war))
print("uar: " + str(uar))
print(tot_mat)

print("RESULTS FOR RES 50")
print("F1: " + str(f1_2))
print("war: " + str(war_2))
print("uar: " + str(uar_2))
print(tot_mat_2)

print("RESULTS FOR Inception V3")
print("F1: " + str(f1_3))
print("war: " + str(war_3))
print("uar: " + str(uar_3))
print(tot_mat_3)

# print("RESULTS FOR Xception")
# print("F1: " + str(f1_4))
# print("war: " + str(war_4))
# print("uar: " + str(uar_4))
# print(tot_mat_4)