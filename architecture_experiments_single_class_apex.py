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
from networks import train_shallow_alexnet_imagenet_with_attention, train_dual_stream_shallow_alexnet
from networks import train_shallow_alexnet_imagenet_FCN
from networks_ablation import train_shallow_inceptionv3, train_shallow_resnet50, train_shallow_vgg16
from capsule_net import capsule_net, margin_loss

def train(type_of_test, train_id, preprocessing_type, classes=5, feature_type = 'grayscale', db='Combined Dataset', spatial_size = 224, classifier_flag = 'svc', tf_backend_flag = False, attention=False, freeze_flag = 'last'):

	sys.setrecursionlimit(10000)
	# general variables and path
	working_dir = '/home/viprlab/Documents/ME_Autoencoders'
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
		casme2_db = 'CASME2_Flow_Strain_Normalized'
		timesteps_TIM = 1
	elif feature_type == 'flow_strain_224':
		casme2_db = 'CASME2_Flow_OS_224'
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
	batch_size  = 30
	epochs = 100
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

		# # # for model that requires parameters
		# model = type_of_test(classes = classes, ablation_flag=2)

		# # model
		# model = type_of_test(classes=classes)
		# model.compile(loss=margin_loss, optimizer=adam, metrics=[metrics.categorical_accuracy])



		model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])		
		f1_king = 0
		# epoch by epoch
		for epoch_counter in range(epochs_step):
			tot_mat = tot_mat_list[epoch_counter]
			pred = pred_list[epoch_counter]
			y_list = y_list_list[epoch_counter]
			print("Current Training Epoch: " + str(epochs))

			clf = SVC(kernel = 'linear', C = 1, decision_function_shape='ovr')
			loso_generator = create_generator_LOSO(total_list, total_labels, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase='svc')

			for X, y, non_binarized_y in loso_generator:

				model.fit(X, y, batch_size = batch_size, epochs = epochs, shuffle = True, callbacks=[history])

				# svm
				if classifier_flag == 'svc':
					if attention == True:
						print("attention")
						encoder = Model(inputs = model.input, outputs = model.get_layer('flatten').get_output_at(1))
						# encoder = Model(inputs = model.input, outputs = model.layers[-7].get_output_at(1))
						plot_model(encoder, to_file='encoder.png', show_shapes=True)
					else:
						encoder = Model(inputs = model.input, outputs = model.layers[-2].output)
					spatial_features = encoder.predict(X, batch_size = batch_size)
					if tf_backend_flag == True:
						spatial_features = np.reshape(spatial_features, (spatial_features.shape[0], spatial_features.shape[-1]))

					clf.fit(spatial_features, non_binarized_y)


			# Resource Clear up
			del X, y

			# Test Time 
			test_loso_generator = create_generator_LOSO(total_list, total_labels, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase = False)


			for X, y, non_binarized_y in test_loso_generator:
				# Spatial Encoding
				# svm
				if classifier_flag == 'svc':
					spatial_features = encoder.predict(X, batch_size = batch_size)
					if tf_backend_flag == True:
						spatial_features = np.reshape(spatial_features, (spatial_features.shape[0], spatial_features.shape[-1]))
					predicted_class = clf.predict(spatial_features)

				# softmax
				elif classifier_flag == 'softmax':
					spatial_features = model.predict(X)
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
	# f1 = f1_list[highest_idx]
	# macro_f1 = macro_f1_list[highest_idx]
	# war = war_list[highest_idx]
	# uar = uar_list[highest_idx]
	# tot_mat = tot_mat_list[highest_idx]
	# weighted_f1 = weighted_f1_list[highest_idx]

	# print confusion matrix of highest f1
	highest_idx = np.argmax(f1_list)
	highest_idx = epochs - 1 # take last epoch
	print("Best Results: ")
	print(tot_mat_list[highest_idx])
	print("Micro F1: " + str(f1_list[highest_idx]))
	print("Macro F1: " + str(macro_f1_list[highest_idx]))
	print("WAR: " + str(war_list[highest_idx]))
	print("UAR: " + str(uar_list[highest_idx]))

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
	batch_size  = 1
	epochs = 50
	total_samples = 0


	# backend
	if tf_backend_flag:
		K.set_image_dim_ordering('tf')	

	# pre-process input images and normalization
	for sub in range(len(total_list)):
		# model
		model = type_of_test()
		# load weights
		model.load_weights(weights_path + str(sub) + '.h5')

		model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
		clf = SVC(kernel = 'linear', C = 1, decision_function_shape='ovr')
		loso_generator = create_generator_LOSO(total_list, total_labels, classes, sub, preprocessing_type, spatial_size = spatial_size, train_phase='svc')


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

# f1, war, uar, tot_mat, macro_f1, weighted_f1 =  test(test_vgg16_imagenet, 'vgg16_fs', preprocessing_type='vgg', feature_type = 'flow_strain_224', db='Combined_Dataset_Apex_Flow', spatial_size = 224, tf_backend_flag = False)
# # f1, war, uar, tot_mat, macro_f1, weighted_f1 =  test(test_vgg19_imagenet, 'vgg19_f', feature_type = 'flow', db='Combined_Dataset_Apex_Flow', spatial_size = 224, tf_backend_flag = False)
# f1_2, war_2, uar_2, tot_mat_2, macro_f1_2, weighted_f1_2 =  test(test_res50_imagenet, 'res50_fs', preprocessing_type='res', feature_type = 'flow_strain_224', db='Combined_Dataset_Apex_Flow', spatial_size = 224, tf_backend_flag = False)
# f1_3, war_3, uar_3, tot_mat_3, macro_f1_3, weighted_f1_3 =  test(test_inceptionv3_imagenet, 'incepv3', preprocessing_type='incep', feature_type = 'flow_strain_224', db='Combined_Dataset_Apex_Flow', spatial_size = 299, tf_backend_flag = False)
# f1_4, war_4, uar_4, tot_mat_4, macro_f1_4, weighted_f1_4 =  test(test_inceptionResV2_imagenet, 'incepres_f', feature_type = 'flow', db='Combined_Dataset_Apex_Flow', spatial_size = 299, tf_backend_flag = False)
# f1, war, uar, tot_mat, macro_f1, weighted_f1 =  test(test_vgg19_imagenet, 'vgg19_g', feature_type = 'grayscale', db='Combined_Dataset_Apex', spatial_size = 224, tf_backend_flag = False)
# f1_2, war_2, uar_2, tot_mat_2, macro_f1_2, weighted_f1_2 =  test(test_mobilenet_imagenet, 'mobilenet_g', feature_type = 'grayscale', db='Combined_Dataset_Apex', spatial_size = 224, tf_backend_flag = True)
# f1_3, war_3, uar_3, tot_mat_3, macro_f1_3, weighted_f1_3 =  test(test_xception_imagenet, 'xception_g', feature_type = 'grayscale', db='Combined_Dataset_Apex', spatial_size = 299, tf_backend_flag = True)
# f1_4, war_4, uar_4, tot_mat_4, macro_f1_4, weighted_f1_4 =  test(test_inceptionResV2_imagenet, 'incepres_g', feature_type = 'grayscale', db='Combined_Dataset_Apex', spatial_size = 299, tf_backend_flag = False)

f1, war, uar, tot_mat, macro_f1, weighted_f1 =  train(train_shallow_alexnet_imagenet, 'dilated_conv', preprocessing_type=None, feature_type = 'flow', db='Combined_Dataset_Apex_Flow', spatial_size = 227, classifier_flag='softmax', tf_backend_flag = False, attention = False, freeze_flag=None, classes=5)
# f1, war, uar, tot_mat, macro_f1, weighted_f1 =  train(train_vgg16_imagenet, 'vgg16_41_fs', preprocessing_type='vgg', feature_type = 'flow_strain', db='Combined_Dataset_Apex_Flow', spatial_size = 224, tf_backend_flag = False)
# f1_2, war_2, uar_2, tot_mat_2, macro_f1_2, weighted_f1_2 =  train(train_res50_imagenet, 'res50_23_analysis', preprocessing_type = 'res', feature_type = 'flow', db='Combined_Dataset_Apex_Flow', spatial_size = 224, tf_backend_flag = False)
# f1_3, war_3, uar_3, tot_mat_3, macro_f1_3, weighted_f1_3 =  train(train_inceptionv3_imagenet, 'incepv3_41C_fs', preprocessing_type='incepv3', feature_type = 'flow_strain', db='Combined_Dataset_Apex_Flow', spatial_size = 299, tf_backend_flag = False)

# f1, war, uar, tot_mat, macro_f1, weighted_f1 =  train(train_vgg16_imagenet, 'vgg16_44', preprocessing_type='vgg', feature_type = 'flow_strain_224', db='Combined_Dataset_Apex_Flow', spatial_size = 224, tf_backend_flag = False)
# f1_2, war_2, uar_2, tot_mat_2, macro_f1_2, weighted_f1_2 =  train(train_res50_imagenet, 'res50_43B', preprocessing_type = 'res', feature_type = 'flow_strain_224', db='Combined_Dataset_Apex_Flow', spatial_size = 224, tf_backend_flag = False)
# f1_3, war_3, uar_3, tot_mat_3, macro_f1_3, weighted_f1_3 =  train(train_inceptionv3_imagenet, 'incepv3_43C', preprocessing_type='incepv3', feature_type = 'flow_strain_224', db='Combined_Dataset_Apex_Flow', spatial_size = 299, tf_backend_flag = False)
# f1, war, uar, tot_mat, macro_f1, weighted_f1 =  test(test_vgg16_finetuned, 'vgg16_43', preprocessing_type='vgg', feature_type = 'flow_strain_224', db='Combined_Dataset_Apex_Flow', spatial_size = 224, tf_backend_flag = False)

# f1, war, uar, tot_mat, macro_f1, weighted_f1 =  train(train_vgg16_imagenet, 'vgg16_44', preprocessing_type='vgg', feature_type = 'flow_strain_224', db='Combined_Dataset_Apex_Flow', spatial_size = 224, tf_backend_flag = False)
# f1_2, war_2, uar_2, tot_mat_2, macro_f1_2, weighted_f1_2 =  train(train_res50_imagenet, 'res50_44B', preprocessing_type = 'res', feature_type = 'flow_strain_224', db='Combined_Dataset_Apex_Flow', spatial_size = 224, tf_backend_flag = False)
# f1_3, war_3, uar_3, tot_mat_3, macro_f1_3, weighted_f1_3 =  train(train_inceptionv3_imagenet, 'incepv3_44C', preprocessing_type='incepv3', feature_type = 'flow_strain_224', db='Combined_Dataset_Apex_Flow', spatial_size = 299, tf_backend_flag = False)


# print("RESULTS FOR vgg_finetuning")
# print("F1: " + str(f1))
# print("war: " + str(war))
# print("uar: " + str(uar))
# print("Macro_f1: " + str(macro_f1))
# print("Weighted_f1: " + str(weighted_f1))
# print(tot_mat)	

# print("RESULTS FOR alexnet")
# print("F1: " + str(f1))
# print("war: " + str(war))
# print("uar: " + str(uar))
# print("Macro_f1: " + str(macro_f1))
# print("Weighted_f1: " + str(weighted_f1))
# print(tot_mat)

# print("RESULTS FOR res50_finetuning")
# print("F1: " + str(f1_2))
# print("war: " + str(war_2))
# print("uar: " + str(uar_2))
# print("Macro_f1: " + str(macro_f1_2))
# print("Weighted_f1: " + str(weighted_f1_2))
# print(tot_mat_2)	

# print("RESULTS FOR incepv3_finetuning_f")
# print("F1: " + str(f1_3))
# print("war: " + str(war_3))
# print("uar: " + str(uar_3))
# print("Macro_f1: " + str(macro_f1_3))
# print("Weighted_f1: " + str(weighted_f1_3))
# print(tot_mat_3)

# print("RESULTS FOR inceptionresv2_g")
# print("F1: " + str(f1_4))
# print("war: " + str(war_4))
# print("uar: " + str(uar_4))
# print("Macro_f1: " + str(macro_f1_4))
# print("Weighted_f1: " + str(weighted_f1_4))
# print(tot_mat_4)		