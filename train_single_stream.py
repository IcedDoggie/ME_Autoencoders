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
from utilities import class_merging, read_image, create_generator_LOSO, class_discretization, create_generator_LOSO_image_cutting_augmentation
from utilities import LossHistory, record_loss_accuracy
from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall, sklearn_macro_f1
from models import VGG_16, temporal_module, layer_wise_conv_autoencoder, layer_wise_autoencoder, convolutional_autoencoder
from evaluationmatrix import majority_vote, temporal_predictions_averaging
from utilities import epoch_analysis
from networks import train_dual_stream_shallow_alexnet, train_shallow_alexnet_imagenet
from networks_ablation import train_shallow_inceptionv3, train_shallow_resnet50, train_shallow_vgg16
# from capsule_net import capsule_net, margin_loss

from losses import e2_keras, earth_mover_loss
from utilities import compute_distribution, compute_distribution_OS
import lera
from lera.keras import LeraCallback

def train(type_of_test, train_id, preprocessing_type, classes=5, feature_type = 'grayscale', db='Combined_Dataset_Apex_Flow', spatial_size = 224, classifier_flag = 'svc', tf_backend_flag = False, attention=False, freeze_flag = 'last'):

	sys.setrecursionlimit(10000)
	# general variables and path
	working_dir = '/home/viprlab/Documents/ME_Autoencoders'
	root_dir = '/media/viprlab/01D31FFEF66D5170/Ice/' + db + '/'
	weights_path = '/media/viprlab/01D31FFEF66D5170/Ice/'
	if os.path.isdir(weights_path + 'Weights/'+ str(train_id) ) == False:
		os.mkdir(weights_path + 'Weights/'+ str(train_id) )	

	# # path for babeen
	# working_dir = '/home/babeen/Documents/ME_Autoencoders'
	# root_dir = '/home/babeen/Documents/MMU_Datasets/' + db + '/'
	# weights_path = '/home/babeen/Documents/MMU_Datasets/'
	# if os.path.isdir(weights_path + 'Weights/'+ str(train_id) ) == False:
	# 	os.mkdir(weights_path + 'Weights/'+ str(train_id) )			

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
		casme2_db = 'CASME2_Flow_Strain_Normalized'
		timesteps_TIM = 1
	elif feature_type == 'flow_strain_224':
		casme2_db = 'CASME2_Flow_OS_224'
		timesteps_TIM = 1
	elif feature_type == 'magnified_flow':
		casme2_db = 'CASME2_Optical_Magnified_a10'
		timesteps_TIM = 1
	elif feature_type == 'magnified_RGB_3':
		casme2_db = 'CASME2_magnified_output_3_cropped_apex'
		timesteps_TIM = 1
	elif feature_type == 'magnified_RGB_5':
		casme2_db = 'CASME2_magnified_output_5_cropped_apex'
		timesteps_TIM = 1				
	elif feature_type == 'magnified_RGB_YT_30':
		casme2_db = 'ratio_30_apex'
		timesteps_TIM = 1		

	elif feature_type == 'strain_only':
		casme2_db = 'CASME2_Strain'
		timesteps_TIM = 1



	classes = classes
	spatial_size = spatial_size
	channels = 3
	data_dim = 4096
	# tot_mat = np.zeros((classes, classes))

	# labels reading
	casme2_table = loading_casme_table(root_dir, casme2_db)
	# samm_table, _ = loading_samm_table(root_dir, samm_db, objective_flag=0)
	# smic_table = loading_smic_table(root_dir, smic_db)
	casme2_table = class_discretization(casme2_table, 'CASME_2')
	# samm_table = class_discretization(samm_table, 'SAMM')
	# smic_table = smic_table[0]

	casme2_OS = loading_casme_table(root_dir, 'CASME2_Strain')
	casme2_OS = class_discretization(casme2_OS, 'CASME_2')
	casme2_OS_list, _ = read_image(root_dir, 'CASME2_Strain', casme2_OS)


	# images reading, read according to table
	# samm_list, samm_labels = read_image(root_dir, samm_db, samm_table)
	# smic_list, smic_labels = read_image(root_dir, smic_db, smic_table)
	# print(casme2_table)
	casme_list, casme_labels = read_image(root_dir, casme2_db, casme2_table)

	# total_list = samm_list + smic_list + casme_list
	# total_labels = samm_labels + smic_labels + casme_labels
	total_list = casme_list
	total_labels = casme_labels
	# total_list = samm_list
	# total_labels = samm_labels

	# total_list = smic_list
	# total_labels = smic_labels

	# print(total_list)
	# print(total_labels)


	# training configuration
	learning_rate = 0.0001
	history = LossHistory()	
	sgd = optimizers.SGD(lr=learning_rate, decay=1e-7, momentum=0.9, nesterov=True)
	adam = optimizers.Adam(lr=learning_rate, decay=1e-7)
	stopping = EarlyStopping(monitor='loss', min_delta = 0, mode = 'min', patience=5)	
	batch_size  = 32
	epochs = 100
	total_samples = 0

	# training config
	tot_mat = np.zeros((classes, classes))
	pred = []
	y_list = []



	# backend
	if tf_backend_flag:
		K.set_image_dim_ordering('tf')	

	# pre-process input images and normalization
	for sub in range(len(total_list)):
		# model
		model = type_of_test(classes = classes, freeze_flag = freeze_flag)

		# # # for model that requires parameters
		# model = type_of_test(classes = classes, ablation_flag=2)

		# # model
		# model = type_of_test(classes=classes)
		# model.compile(loss=margin_loss, optimizer=adam, metrics=[metrics.categorical_accuracy])

		# Send hyperparameters
		lera_str = 'Sub_' + str(sub)
		lera.log_hyperparams({ 'title':  lera_str})


		model.compile(loss=['categorical_crossentropy', earth_mover_loss, earth_mover_loss], optimizer=adam, metrics=[metrics.categorical_accuracy])		
		f1_king = 0

		mean_e2 = np.zeros((classes))


		######################### Training ############################
		loso_generator = create_generator_LOSO(total_list, total_labels, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase='svc')
		OS_loso_generator = create_generator_LOSO(casme2_OS_list, total_labels, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase='svc')

		# for X, y, non_binarized_y in loso_generator:
		for (alpha, beta) in zip(loso_generator, OS_loso_generator):
			X, y, non_binarized_y = alpha[0], alpha[1], alpha[2]
			X_2, _, _ = beta[0], beta[1], beta[2]				

			strain_distrib_horizontal, strain_distrib_vertical = compute_distribution_OS(X_2)

			model.fit(X, [y, strain_distrib_horizontal, strain_distrib_vertical], batch_size = batch_size, epochs = epochs, shuffle = True, callbacks=[history])
			

		# Resource Clear up
		del X, y

		# log history results [losses, accuracy, epochs]
		filename = './net_images/' + str(train_id) + '.txt'
		f = open(filename, 'a')
		f.write('Sub_' + str(sub) + '\n')
		f.write('Loss Accuracy' + '\n')
		for counter_loss in range(len(history.losses)):
			curr_loss = str(history.losses[counter_loss])
			curr_acc = str(history.accuracy[counter_loss])
			print(curr_acc)
			f.write(curr_loss + ' ')
			f.write(curr_acc + '\n')
		f.close()

		# Test Time 
		test_loso_generator = create_generator_LOSO(total_list, total_labels, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase = False)


		for X, y, non_binarized_y in test_loso_generator:


			spatial_features, _, _ = model.predict(X)
			predicted_class = np.argmax(spatial_features, axis=1)

			
			non_binarized_y = non_binarized_y[0]

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


			total_samples += len(non_binarized_y)

			war = weighted_average_recall(tot_mat, classes, total_samples)
			uar = unweighted_average_recall(tot_mat, classes)
			macro_f1, weighted_f1 = sklearn_macro_f1(y_list, pred)



		weights_name = weights_path + str(sub) + '.h5'
		model.save_weights(weights_name)

		# Resource CLear up
		del X, y, non_binarized_y





	# print confusion matrix of highest f1
	print("Best Results: ")
	print(tot_mat)
	print("Micro F1: " + str(f1))
	print("Macro F1: " + str(macro_f1))
	print("WAR: " + str(war))
	print("UAR: " + str(uar))

	return f1, war, uar, tot_mat, macro_f1, weighted_f1


f1, war, uar, tot_mat, macro_f1, weighted_f1 =  train(train_shallow_alexnet_imagenet, 'test_emd', preprocessing_type=None, feature_type = 'flow', db='Combined_Dataset_Apex_Flow', spatial_size = 227, classifier_flag='softmax', tf_backend_flag = False, attention = False, freeze_flag=None, classes=5)

