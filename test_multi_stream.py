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
from networks import train_shallow_alexnet_imagenet_with_attention, train_dual_stream_shallow_alexnet, train_tri_stream_shallow_alexnet_pooling_merged
from networks import train_tri_stream_shallow_alexnet_pooling_merged_latent_features
from visualization import plot_confusion_matrix

def test(type_of_test, train_id, preprocessing_type, classes=5, feature_type = 'grayscale', db='Combined Dataset', spatial_size = 224, classifier_flag = 'svc', tf_backend_flag = False, attention=False, freeze_flag = 'last'):

	sys.setrecursionlimit(10000)
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

	classes = 5
	spatial_size = spatial_size
	channels = 3
	data_dim = 4096
	# tot_mat = np.zeros((classes, classes))

	# labels reading
	# casme2_table = loading_casme_table(root_dir, casme2_db)
	samm_table, _ = loading_samm_table(root_dir, samm_db, objective_flag=0)
	# smic_table = loading_smic_table(root_dir, smic_db)
	# casme2_table = class_discretization(casme2_table, 'CASME_2')
	samm_table = class_discretization(samm_table, 'SAMM')
	# smic_table = smic_table[0]



	# images reading, read according to table
	casme_list, casme_labels = read_image(root_dir, samm_db, samm_table)
	# smic_list, smic_labels = read_image(root_dir, smic_db, smic_table)
	# print(casme2_table)
	# casme_list, casme_labels = read_image(root_dir, casme2_db, casme2_table)

	# total_list = samm_list + smic_list + casme_list
	# total_labels = samm_labels + smic_labels + casme_labels
	total_list = casme_list
	total_labels = casme_labels

	# MULTI STREAM SETTINGS
	sec_db = 'SAMM_Optical_Gray_Weighted'
	casme2_2, _ = loading_samm_table(root_dir, sec_db, objective_flag=0)
	casme2_2 = class_discretization(casme2_2, 'SAMM')	
	casme_list_2, casme_labels_2 = read_image(root_dir, sec_db, casme2_2)
	
	# third_db = 'CASME2_Flow_Strain_Normalized'
	# casme2_3 = loading_casme_table(root_dir, third_db)
	# casme2_3 = class_discretization(casme2_3, 'CASME_2')
	# casme_list_3, casme_labels_3 = read_image(root_dir, third_db, casme2_3)


	# training configuration
	learning_rate = 0.001
	history = LossHistory()	
	sgd = optimizers.SGD(lr=learning_rate, decay=1e-7, momentum=0.9, nesterov=True)
	adam = optimizers.Adam(lr=learning_rate, decay=1e-7)
	stopping = EarlyStopping(monitor='loss', min_delta = 0, mode = 'min', patience=5)	
	batch_size  = 1
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
	pred = []
	y_list = []	

	tot_mat = np.zeros((classes, classes))


	# backend
	if tf_backend_flag:
		K.set_image_dim_ordering('tf')	

	# pre-process input images and normalization
	for sub in range(len(total_list)):

		# model
		model = type_of_test(classes = classes, freeze_flag = freeze_flag)
		weights_name = weights_path + str(sub) + '.h5'
		model.load_weights(weights_name)
		model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])		
		# model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=adam, metrics=[metrics.categorical_accuracy])		




		# Test Time 
		test_loso_generator = create_generator_LOSO(casme_list, casme_labels, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase = False)
		test_loso_generator_2 = create_generator_LOSO(casme_list_2, casme_labels_2, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase = False)
		# test_loso_generator_3 = create_generator_LOSO(casme_list_3, casme_labels_3, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase = False)

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


			# if epoch_counter < 1:	 # avoid numerical problem
			total_samples += len(non_binarized_y)

			war = weighted_average_recall(tot_mat, classes, total_samples)
			uar = unweighted_average_recall(tot_mat, classes)
			macro_f1, weighted_f1 = sklearn_macro_f1(y_list, pred)




	# print confusion matrix of highest f1
	# highest_idx = np.argmax(f1_list)
	print("Best Results: ")
	print(tot_mat)
	print("Micro F1: " + str(f1))
	print("Macro F1: " + str(macro_f1))
	print("WAR: " + str(war))
	print("UAR: " + str(uar))

	classes = ['anger', 'happiness', 'contempt', 'surprise', 'other']
	plot_confusion_matrix(tot_mat, classes)

	return f1, war, uar, tot_mat, macro_f1, weighted_f1


f1, war, uar, tot_mat, macro_f1, weighted_f1 =  test(train_dual_stream_shallow_alexnet, 'shallow_alexnet_multi_38', preprocessing_type=None, feature_type = 'flow', db='Combined_Dataset_Apex_Flow', spatial_size = 227, classifier_flag='softmax', tf_backend_flag = False, attention = False, freeze_flag=None, classes=5)
