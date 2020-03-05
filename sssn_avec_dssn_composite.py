import pandas as pd
import cv2, os, pydot, graphviz, json, itertools, sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
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
from networks import train_dssn_merging_with_sssn

def train(type_of_test, train_id, preprocessing_type, classes=5, feature_type = 'grayscale', db='Combined Dataset', spatial_size = 224, classifier_flag = 'svc', tf_backend_flag = False, attention=False, freeze_flag = 'last'):

	sys.setrecursionlimit(10000)
	# general variables and path
	working_dir = '/home/babeen/Documents/ME_Autoencoders/'
	root_dir = '/home/babeen/Documents/MMU_Datasets/' + db + '/'
	weights_path = '/home/babeen/Documents/MMU_Datasets/'
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
	elif feature_type == 'gray_weighted_flow':
		casme2_db = 'CASME2_Optical_Gray_Weighted'
		samm_db = 'SAMM_Optical_Gray_Weighted'
		smic_db = 'SMIC_Optical_Gray_Weighted'
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

	# labels reading
	casme2_table = loading_casme_table(root_dir, casme2_db)
	casme2_table = class_discretization(casme2_table, 'CASME_2')	
	casme_list, casme_labels = read_image(root_dir, casme2_db, casme2_table)

	# MULTI STREAM SETTINGS (TRI STREAM)
	sec_db = 'CASME2_Optical_Gray_Weighted'
	casme2_2 = loading_casme_table(root_dir, sec_db)
	casme2_2 = class_discretization(casme2_2, 'CASME_2')
	casme_list_2, casme_labels_2 = read_image(root_dir, sec_db, casme2_2)

	third_db = 'CASME2_Flow_Strain_minor'
	casme2_3 = loading_casme_table(root_dir, third_db)
	casme2_3 = class_discretization(casme2_3, 'CASME_2')
	casme_list_3, casme_labels_3 = read_image(root_dir, third_db, casme2_3)

	# SAMM DBs
	samm_1, _ = loading_samm_table(root_dir, samm_db, objective_flag=0)
	samm_1 = class_merging(samm_1)
	samm_list, samm_labels = read_image(root_dir, samm_db, samm_1)

	samm_sec_db = 'SAMM_Optical_Gray_Weighted'
	samm_2, _ = loading_samm_table(root_dir, samm_sec_db, objective_flag=0)
	samm_2 = class_merging(samm_2)
	samm_list_2, samm_labels_2 = read_image(root_dir, samm_sec_db, samm_2)

	samm_third_db = 'SAMM_Flow_Strain_minor'
	samm_3, _ = loading_samm_table(root_dir, samm_third_db, objective_flag=0)
	samm_3 = class_merging(samm_3)
	samm_list_3, samm_labels_3 = read_image(root_dir, samm_third_db, samm_3)


	# SMIC DBs
	smic_1 = loading_smic_table(root_dir, smic_db)
	smic_1 = smic_1[0]
	smic_list, smic_labels = read_image(root_dir, smic_db, smic_1)		

	smic_sec_db = 'SMIC_Optical_Gray_Weighted'
	smic_2 = loading_smic_table(root_dir, smic_sec_db)
	smic_2 = smic_2[0]
	smic_list_2, smic_labels_2 = read_image(root_dir, smic_sec_db, smic_2)		

	smic_third_db = 'SMIC_Flow_Strain_minor'
	smic_3 = loading_smic_table(root_dir, smic_third_db)
	smic_3 = smic_3[0]
	smic_list_3, smic_labels_3 = read_image(root_dir, smic_third_db, smic_3)	

	# Combining each DBs
	total_list = casme_list + samm_list + smic_list
	total_labels = casme_labels + samm_labels + smic_labels

	total_list_2 = casme_list_2 + samm_list_2 + smic_list_2
	total_labels_2 = casme_labels_2 + samm_labels_2 + smic_labels_2

	total_list_3 = casme_list_3 + samm_list_3 + smic_list_3
	total_labels_3 = casme_labels_3 + samm_labels_3 + smic_labels_3

	# training configuration
	learning_rate = 0.0001
	history = LossHistory()	
	sgd = optimizers.SGD(lr=learning_rate, decay=1e-7, momentum=0.9, nesterov=True)
	adam = optimizers.Adam(lr=learning_rate, decay=1e-7)
	stopping = EarlyStopping(monitor='loss', min_delta = 0, mode = 'min', patience=5)	
	batch_size  = 60
	epochs = 1
	total_samples = 0

	tot_mat = np.zeros((classes, classes))
	pred = []
	y_list = []


	# pre-process input images and normalization
	for sub in range(len(total_list)):
		# model
		model = type_of_test(classes = classes, freeze_flag = freeze_flag)
		model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])		

		clf = SVC(kernel = 'linear', C = 1, decision_function_shape='ovr')
		loso_generator = create_generator_LOSO(total_list, total_labels, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase='svc')
		loso_generator_2 = create_generator_LOSO(total_list_2, total_labels_2, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase='svc')
		loso_generator_3 = create_generator_LOSO(total_list_3, total_labels_3, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase='svc')


		for (alpha, beta, charlie) in zip(loso_generator, loso_generator_2, loso_generator_3):
			X, y, non_binarized_y = alpha[0], alpha[1], alpha[2]
			X_2, y_2, non_binarized_y_2 = beta[0], beta[1], beta[2]
			X_3, y_3, non_binarized_y_3 = charlie[0], charlie[1], charlie[2]
			print(".fit")
			model.fit([X, X_2, X_3], y, batch_size = batch_size, epochs = epochs, shuffle = False, callbacks=[history])
			print("after fit")

			# svm
			if classifier_flag == 'svc':
				if attention == True:
					encoder = Model(inputs = model.input, outputs = model.get_layer('softmax_activate').get_output_at(1))
					plot_model(encoder, to_file='encoder.png', show_shapes=True)
				else:
					encoder = Model(inputs = model.input, outputs = model.get_layer('softmax_activate').output)
				spatial_features = encoder.predict([X, X_2, X_3], batch_size = batch_size)

				clf.fit(spatial_features, non_binarized_y)
		

		# Resource Clear up
		del X, y

		# Test Time 
		test_loso_generator = create_generator_LOSO(total_list, total_labels, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase = False)
		test_loso_generator_2 = create_generator_LOSO(total_list_2, total_labels_2, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase = False)
		test_loso_generator_3 = create_generator_LOSO(total_list_3, total_labels_3, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase = False)

		for (alpha, beta, charlie) in zip(test_loso_generator, test_loso_generator_2, test_loso_generator_3):
			X, y, non_binarized_y = alpha[0], alpha[1], alpha[2]
			X_2, y_2, non_binarized_y_2 = beta[0], beta[1], beta[2]	
			X_3, y_3, non_binarized_y_3 = charlie[0], charlie[1], charlie[2]	

			# Spatial Encoding
			# svm
			if classifier_flag == 'svc':
				spatial_features = encoder.predict([X, X_2, X_3], batch_size = batch_size)
				if tf_backend_flag == True:
					spatial_features = np.reshape(spatial_features, (spatial_features.shape[0], spatial_features.shape[-1]))
				predicted_class = clf.predict(spatial_features)

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
			total_samples += len(non_binarized_y)
			war = weighted_average_recall(tot_mat, classes, total_samples)
			uar = unweighted_average_recall(tot_mat, classes)
			macro_f1, weighted_f1 = sklearn_macro_f1(y_list, pred)




		# save the maximum epoch only (replace with maximum f1)
		weights_name = weights_path + str(sub) + '.h5'
		model.save_weights(weights_name)

		# Resource Clear up
		del X, y, non_binarized_y


	# print confusion matrix of highest f1
	print("Best Results: ")
	print(tot_mat)
	print("Micro F1: " + str(f1))
	print("Macro F1: " + str(macro_f1))
	print("WAR: " + str(war))
	print("UAR: " + str(uar))


	return f1, war, uar, tot_mat, macro_f1, weighted_f1





f1, war, uar, tot_mat, macro_f1, weighted_f1 =  train(train_dssn_merging_with_sssn, 'sssn_avec_dssn', preprocessing_type='vgg', feature_type = 'flow', db='Combined_Dataset_Apex_Flow', spatial_size = 227, tf_backend_flag = False)



