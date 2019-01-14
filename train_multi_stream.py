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
from models import VGG_16, temporal_module, layer_wise_conv_autoencoder, layer_wise_autoencoder, convolutional_autoencoder
from networks import test_res50_finetuned, test_vgg16_finetuned, test_inceptionv3_finetuned
from networks import train_res50_imagenet, train_vgg16_imagenet, train_inceptionv3_imagenet, train_alexnet_imagenet, train_shallow_alexnet_imagenet
from networks import test_vgg16_imagenet, test_inceptionv3_imagenet, test_res50_imagenet
from networks import test_vgg19_imagenet, test_mobilenet_imagenet, test_xception_imagenet, test_inceptionResV2_imagenet
from evaluationmatrix import majority_vote, temporal_predictions_averaging
from utilities import epoch_analysis
from networks import train_shallow_alexnet_imagenet_with_attention, train_dual_stream_shallow_alexnet, train_tri_stream_shallow_alexnet_pooling_merged, train_dual_stream_with_auxiliary_attention_networks
from networks import train_dual_stream_with_auxiliary_attention_networks_dual_loss, train_tri_stream_shallow_alexnet_pooling_merged_slow_fusion, train_tri_stream_shallow_alexnet_pooling_merged_latent_features
from siamese_models import euclidean_distance_loss

def train(type_of_test, train_id, preprocessing_type, classes=5, feature_type = 'grayscale', db='Combined Dataset', spatial_size = 224, classifier_flag = 'svc', tf_backend_flag = False, attention=False, freeze_flag = 'last'):

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
		timesteps_TIM = 1
	elif feature_type == 'flow':
		casme2_db = 'CASME2_Optical'
		samm_db = 'SAMM_Optical'
		smic_db = 'SMIC_Optical_Christy'
		timesteps_TIM = 1	
	elif feature_type == 'flow_strain':
		casme2_db = 'CASME2_Flow_Strain_Normalized'
		smic_db = 'SMIC_Flow_Strain_Christy'
		timesteps_TIM = 1
	elif feature_type == 'flow_strain_224':
		casme2_db = 'CASME2_Flow_OS_224'
		timesteps_TIM = 1
	elif feature_type == 'gray_weighted_flow':
		casme2_db = 'CASME2_Optical_Gray_Weighted'
		samm_db = 'SAMM_Optical_Gray_Weighted'
		smic_db = 'SMIC_Optical_Gray_Weighted'
		timesteps_TIM = 1
	elif feature_type == 'flow_strain_major':
		casme2_db = 'CASME2_Flow_Strain_major'
		timesteps_TIM = 1
	elif feature_type == 'flow_strain_minor':
		casme2_db = 'CASME2_Flow_Strain_minor'
		samm_db = 'SAMM_Flow_Strain_minor'
		smic_db = 'SMIC_Flow_Strain_minor'
		timesteps_TIM = 1		

	classes = classes
	spatial_size = spatial_size
	channels = 5
	data_dim = 4096
	# tot_mat = np.zeros((classes, classes))

	# labels reading
	casme2_table = loading_casme_table(root_dir, casme2_db)
	casme2_table = class_discretization(casme2_table, 'CASME_2')	
	casme_list, casme_labels = read_image(root_dir, casme2_db, casme2_table)

	# # temporarily using casme as variable for easy
	# casme2_table, _ = loading_samm_table(root_dir, samm_db, objective_flag=0)
	# casme2_table = class_discretization(casme2_table, 'SAMM')	
	# casme_list, casme_labels = read_image(root_dir, samm_db, casme2_table)

	# # temporarily using casme as variable for easy
	# casme2_table = loading_smic_table(root_dir, smic_db)
	# casme2_table = casme2_table[0]
	# casme_list, casme_labels = read_image(root_dir, smic_db, casme2_table)



	# images reading, read according to table
	# print(casme2_table)

	# total_list = samm_list + smic_list + casme_list
	# total_labels = samm_labels + smic_labels + casme_labels
	total_list = casme_list
	total_labels = casme_labels

	# MULTI STREAM SETTINGS (TRI STREAM)
	sec_db = 'CASME2_Optical_Gray_Weighted'
	casme2_2 = loading_casme_table(root_dir, sec_db)
	casme2_2 = class_discretization(casme2_2, 'CASME_2')
	casme_list_2, casme_labels_2 = read_image(root_dir, sec_db, casme2_2)

	# third_db = 'CASME2_Optical_Gray_Weighted'
	# casme2_3 = loading_casme_table(root_dir, third_db)
	# casme2_3 = class_discretization(casme2_3, 'CASME_2')
	# casme_list_3, casme_labels_3 = read_image(root_dir, third_db, casme2_3)

	# # # MULTI STREAM SETTINGS (TRI STREAM)
	# sec_db = 'SAMM_Flow_Strain_minor'
	# casme2_2, _ = loading_samm_table(root_dir, sec_db, objective_flag=0)
	# casme2_2 = class_discretization(casme2_2, 'SAMM')	
	# casme_list_2, casme_labels_2 = read_image(root_dir, sec_db, casme2_2)

	# third_db = 'SAMM_Optical_Gray_Weighted'
	# casme2_3, _ = loading_samm_table(root_dir, third_db, objective_flag=0)
	# casme2_3 = class_discretization(casme2_3, 'SAMM')	
	# casme_list_3, casme_labels_3 = read_image(root_dir, third_db, casme2_3)

	# # MULTI STREAM SETTINGS (TRI STREAM)
	# sec_db = 'SMIC_Flow_Strain_minor'
	# casme2_2 = loading_smic_table(root_dir, sec_db)
	# casme2_2 = casme2_2[0]
	# casme_list_2, casme_labels_2 = read_image(root_dir, sec_db, casme2_2)

	# third_db = 'SMIC_Optical_Gray_Weighted'
	# casme2_3 = loading_smic_table(root_dir, third_db)
	# casme2_3 = casme2_3[0]
	# casme_list_3, casme_labels_3 = read_image(root_dir, third_db, casme2_3)

	# # DUAL STREAM SETTINGS
	# sec_db = 'SMIC_Flow_Strain_Christy'
	# casme2_2 = loading_smic_table(root_dir, sec_db)
	# casme2_2 = casme2_2[0]
	# casme_list_2, casme_labels_2 = read_image(root_dir, sec_db, casme2_2)

	# # DUAL STREAM SETTINGS
	# sec_db = 'SAMM_Flow_Strain_minor'
	# casme2_2, _ = loading_samm_table(root_dir, sec_db, objective_flag=0)
	# casme2_2 = class_discretization(casme2_2, 'SAMM')	
	# casme_list_2, casme_labels_2 = read_image(root_dir, sec_db, casme2_2)



	# training configuration
	learning_rate = 0.0001
	history = LossHistory()	
	sgd = optimizers.SGD(lr=learning_rate, decay=1e-7, momentum=0.9, nesterov=True)
	adam = optimizers.Adam(lr=learning_rate, decay=1e-7)
	stopping = EarlyStopping(monitor='loss', min_delta = 0, mode = 'min', patience=5)	
	batch_size  = 60
	epochs = 1
	total_samples = 0

	# codes for epoch analysis
	epochs_step = 100
	epochs = 1
	macro_f1_list = []
	weighted_f1_list = []
	loss_list = []
	tot_mat_list = []
	war_list = []
	f1_list = []
	uar_list = []
	pred_list = []
	y_list_list = []	
	for counter in range(epochs_step):
		# create separate tot_mat for diff epoch
		tot_mat_list += [np.zeros((classes, classes))]
		macro_f1_list += [0]
		weighted_f1_list += [0]
		loss_list += [0]
		war_list += [0]
		f1_list += [0]
		uar_list += [0]
		pred_list += [[]]
		y_list_list += [[]]

	# backend
	if tf_backend_flag:
		K.set_image_dim_ordering('tf')	

	# pre-process input images and normalization
	for sub in range(len(total_list)):
		# model
		model = type_of_test(classes = classes, freeze_flag = freeze_flag)
		model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])		
		# model.compile(loss=['categorical_crossentropy', euclidean_distance_loss], optimizer=adam, metrics=[metrics.categorical_accuracy])		


		f1_king = 0
		# epoch by epoch
		for epoch_counter in range(epochs_step):
			tot_mat = tot_mat_list[epoch_counter]
			pred = pred_list[epoch_counter]
			y_list = y_list_list[epoch_counter]
			print("Current Training Epoch: " + str(epochs))

			clf = SVC(kernel = 'linear', C = 1, decision_function_shape='ovr')
			loso_generator = create_generator_LOSO(casme_list, casme_labels, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase='svc')
			loso_generator_2 = create_generator_LOSO(casme_list_2, casme_labels_2, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase='svc')
			# loso_generator_3 = create_generator_LOSO(casme_list_3, casme_labels_3, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase='svc')

			# for (alpha, beta, omega) in zip(loso_generator, loso_generator_2, loso_generator_3):
			# 	X, y, non_binarized_y = alpha[0], alpha[1], alpha[2]
			# 	X_2, y_2, non_binarized_y_2 = beta[0], beta[1], beta[2]
			# 	X_3, y_3, non_binarized_y_3 = omega[0], omega[1], omega[2]

			for (alpha, beta) in zip(loso_generator, loso_generator_2):
				X, y, non_binarized_y = alpha[0], alpha[1], alpha[2]
				X_2, y_2, non_binarized_y_2 = beta[0], beta[1], beta[2]

				model.fit([X, X_2], y, batch_size = batch_size, epochs = epochs, shuffle = False, callbacks=[history])

				# model.fit([X, X_2, X_3], y, batch_size = batch_size, epochs = epochs, shuffle = False, callbacks=[history])
				# model.fit([X, X_2, X_3], [y, X_2], batch_size = batch_size, epochs = epochs, shuffle = False, callbacks=[history])

				# svm
				if classifier_flag == 'svc':
					if attention == True:
						print("attention")
						encoder = Model(inputs = model.input, outputs = model.get_layer('softmax_activate').get_output_at(1))
						# encoder = Model(inputs = model.input, outputs = model.layers[-7].get_output_at(1))
						plot_model(encoder, to_file='encoder.png', show_shapes=True)
					else:
						encoder = Model(inputs = model.input, outputs = model.get_layer('softmax_activate').output)
					spatial_features = encoder.predict([X, X_2, X_3], batch_size = batch_size)
					if tf_backend_flag == True:
						spatial_features = np.reshape(spatial_features, (spatial_features.shape[0], spatial_features.shape[-1]))

					clf.fit(spatial_features, non_binarized_y)


			

			# Resource Clear up
			del X, y

			# Test Time 
			test_loso_generator = create_generator_LOSO(casme_list, casme_labels, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase = False)
			test_loso_generator_2 = create_generator_LOSO(casme_list_2, casme_labels_2, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase = False)
			# test_loso_generator_3 = create_generator_LOSO(casme_list_3, casme_labels_3, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase = False)

			# for (alpha, beta, omega) in zip(test_loso_generator, test_loso_generator_2, test_loso_generator_3):
			# 	X, y, non_binarized_y = alpha[0], alpha[1], alpha[2]
			# 	X_2, y_2, non_binarized_y_2 = beta[0], beta[1], beta[2]	
			# 	X_3, y_3, non_binarized_y_3 = omega[0], omega[1], omega[2]				
	
			for (alpha, beta) in zip(test_loso_generator, test_loso_generator_2):
				X, y, non_binarized_y = alpha[0], alpha[1], alpha[2]
				X_2, y_2, non_binarized_y_2 = beta[0], beta[1], beta[2]	

				# Spatial Encoding
				# svm
				if classifier_flag == 'svc':
					spatial_features = encoder.predict([X, X_2, X_3], batch_size = batch_size)
					if tf_backend_flag == True:
						spatial_features = np.reshape(spatial_features, (spatial_features.shape[0], spatial_features.shape[-1]))
					predicted_class = clf.predict(spatial_features)

				# # softmax
				# elif classifier_flag == 'softmax':
				# 	spatial_features = model.predict([X, X_2, X_3])
				# 	predicted_class = np.argmax(spatial_features, axis=1)
				# softmax
				elif classifier_flag == 'softmax':
					spatial_features = model.predict([X, X_2])
					predicted_class = np.argmax(spatial_features, axis=1)

				non_binarized_y = non_binarized_y[0]
				non_binarized_y_2 = non_binarized_y_2[0]

				print("ROW 2 ROW 3 should be the same SANITY CHECK")
				print(predicted_class)
				print(non_binarized_y)	
				print(non_binarized_y_2)

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


				if epoch_counter < 1:	 # avoid numerical problem
					total_samples += len(non_binarized_y)

				war = weighted_average_recall(tot_mat, classes, total_samples)
				uar = unweighted_average_recall(tot_mat, classes)
				macro_f1, weighted_f1 = sklearn_macro_f1(y_list, pred)

				# results logging
				tot_mat_list[epoch_counter] = tot_mat
				macro_f1_list[epoch_counter] = macro_f1
				weighted_f1_list[epoch_counter] = weighted_f1
				loss_list[epoch_counter] = history.losses
				war_list[epoch_counter] = war
				f1_list[epoch_counter] = f1
				uar_list[epoch_counter] = uar
				pred_list[epoch_counter] = pred
				y_list_list[epoch_counter] = y_list

			# save the maximum epoch only (replace with maximum f1)
			if f1 > f1_king:
				f1_king = f1
				weights_name = weights_path + str(sub) + '.h5'
				model.save_weights(weights_name)

			# Resource CLear up
			del X, y, non_binarized_y


	# perform evaluation on each epoch
	for epoch_counter in range(epochs_step):
		tot_mat = tot_mat_list[epoch_counter]
		f1 = f1_list[epoch_counter]
		war = war_list[epoch_counter]
		uar = uar_list[epoch_counter]
		macro_f1 = macro_f1_list[epoch_counter]		
		weighted_f1 = weighted_f1_list[epoch_counter]
		loss = loss_list[epoch_counter]
		epoch_analysis(root_dir, train_id, db, f1, war, uar, macro_f1, weighted_f1, loss)

		# print(tot_mat)


	# print confusion matrix of highest f1
	highest_idx = np.argmax(f1_list)
	print("Best Results: ")
	print(tot_mat_list[highest_idx])
	print("Micro F1: " + str(f1_list[highest_idx]))
	print("Macro F1: " + str(macro_f1_list[highest_idx]))
	print("WAR: " + str(war_list[highest_idx]))
	print("UAR: " + str(uar_list[highest_idx]))


	f1 = f1_list[highest_idx]
	macro_f1 = macro_f1_list[highest_idx]
	war = war_list[highest_idx]
	uar = uar_list[highest_idx]
	tot_mat = tot_mat_list[highest_idx]
	weighted_f1 = weighted_f1_list[highest_idx]

	return f1, war, uar, tot_mat, macro_f1, weighted_f1

def test(type_of_test, train_id, preprocessing_type, feature_type = 'grayscale', db='Combined Dataset', spatial_size = 224, tf_backend_flag = False):

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
		timesteps_TIM = 1
	elif feature_type == 'flow':
		casme2_db = 'CASME2_Optical'
		samm_db = 'SAMM_Optical'
		smic_db = 'SMIC_Optical'
		timesteps_TIM = 1
	elif feature_type == 'flow_strain':
		casme2_db = 'CASME2_Flow_OS'
		timesteps_TIM = 1
	elif feature_type == 'flow_strain_224':
		casme2_db = 'CASME2_Flow_OS_224'
		timesteps_TIM = 1

	classes = 5
	spatial_size = spatial_size
	channels = 3
	data_dim = 4096
	tot_mat = np.zeros((classes, classes))

	# labels reading
	casme2_table = loading_casme_table(root_dir, casme2_db)
	# samm_table, _ = loading_samm_table(root_dir, samm_db, objective_flag=0)
	# smic_table = loading_smic_table(root_dir, smic_db)
	casme2_table = class_discretization(casme2_table, 'CASME_2')
	# samm_table = class_discretization(samm_table, 'SAMM')
	# smic_table = smic_table[0]

	# images reading, read according to table
	# samm_list, samm_labels = read_image(root_dir, samm_db, samm_table)
	# smic_list, smic_labels = read_image(root_dir, smic_db, smic_table)
	casme_list, casme_labels = read_image(root_dir, casme2_db, casme2_table)

	# total_list = samm_list + smic_list + casme_list
	# total_labels = samm_labels + smic_labels + casme_labels
	total_list = casme_list
	total_labels = casme_labels


	pred = []
	y_list = []

	# training configuration
	learning_rate = 0.0001
	sgd = optimizers.SGD(lr=learning_rate, decay=1e-7, momentum=0.9, nesterov=True)
	adam = optimizers.Adam(lr=learning_rate, decay=learning_rate * 2)
	stopping = EarlyStopping(monitor='loss', min_delta = 0, mode = 'min', patience=5)	
	batch_size  = 30
	epochs = 50
	total_samples = 0


	# backend
	if tf_backend_flag:
		K.set_image_dim_ordering('tf')	

	# pre-process input images and normalization
	for sub in range(len(casme_list)):
		# model
		model = type_of_test()
		# load weights
		model.load_weights(weights_path + str(sub) + '.h5')

		model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
		clf = SVC(kernel = 'linear', C = 1, decision_function_shape='ovr')
		loso_generator = create_generator_LOSO(casme_list, casme_labels, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase='svc')


		for X, y, non_binarized_y in loso_generator:

			# model.fit(X, y, batch_size = batch_size, epochs = epochs, callbacks=[stopping])
			# model = Model(inputs = model.input, outputs = model.layers[-2].output)
			spatial_features = model.predict(X, batch_size = batch_size)
			if tf_backend_flag == True:

				spatial_features = np.reshape(spatial_features, (spatial_features.shape[0], spatial_features.shape[-1]))

			# print(spatial_features)
			clf.fit(spatial_features, non_binarized_y)


		# Resource Clear up
		del X, y

		# Test Time 
		test_loso_generator = create_generator_LOSO(total_list, total_labels, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase = False)


		for X, y, non_binarized_y in test_loso_generator:
			# Spatial Encoding
			spatial_features = model.predict(X, batch_size = batch_size)
			if tf_backend_flag == True:

				spatial_features = np.reshape(spatial_features, (spatial_features.shape[0], spatial_features.shape[-1]))

			predicted_class = clf.predict(spatial_features)
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

		# weights_name = weights_path + str(sub) + '.h5'
		# model.save_weights(weights_name)
		# Resource CLear up
		del X, y, non_binarized_y	
	return f1, war, uar, tot_mat, macro_f1, weighted_f1


# f1, war, uar, tot_mat, macro_f1, weighted_f1 =  train(train_dual_stream_shallow_alexnet, 'shallow_alexnet_multi_38J', preprocessing_type=None, feature_type = 'flow_strain', db='Combined_Dataset_Apex_Flow', spatial_size = 227, classifier_flag='softmax', tf_backend_flag = False, attention = False, freeze_flag=None, classes=3)
f1, war, uar, tot_mat, macro_f1, weighted_f1 =  train(train_dual_stream_shallow_alexnet, 'shallow_alexnet_multi_31J_MULTIPLY_2FC', preprocessing_type=None, feature_type = 'flow', db='Combined_Dataset_Apex_Flow', spatial_size = 227, classifier_flag='softmax', tf_backend_flag = False, attention = False, freeze_flag=None, classes=5)

print("RESULTS FOR shallow alex multi-stream")
print("F1: " + str(f1))
print("war: " + str(war))
print("uar: " + str(uar))
print("Macro_f1: " + str(macro_f1))
print("Weighted_f1: " + str(weighted_f1))
print(tot_mat)	

	