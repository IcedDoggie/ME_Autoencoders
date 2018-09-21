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
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet

from utilities import loading_smic_table, loading_samm_table, loading_casme_table
from utilities import class_merging, read_image, create_generator_LOSO
from utilities import LossHistory, record_loss_accuracy
from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall
from models import VGG_16, temporal_module, layer_wise_conv_autoencoder, layer_wise_autoencoder, convolutional_autoencoder



def test_res50_imagenet():
	resnet50 = ResNet50(weights = 'imagenet')
	resnet50 = Model(inputs = resnet50.input, outputs = resnet50.layers[-2].output)
	plot_model(resnet50, to_file='resnet50.png', show_shapes=True)

	return resnet50

def test_vgg16_imagenet():
	vgg16 = VGG16(weights = 'imagenet')
	vgg16 = Model(inputs = vgg16.input, outputs = vgg16.layers[-2].output)
	plot_model(vgg16, to_file='vgg16.png', show_shapes=True)

	return vgg16

def test_vgg19_imagenet():
	vgg19 = VGG19(weights = 'imagenet')
	vgg19 = Model(inputs = vgg19.input, outputs = vgg19.layers[-2].output)
	plot_model(vgg19, to_file='vgg19.png', show_shapes=True)

	return vgg19

def test_inceptionv3_imagenet():
	inceptionv3 = InceptionV3(weights = 'imagenet')
	inceptionv3 = Model(inputs = inceptionv3.input, outputs = inceptionv3.layers[-2].output)	
	plot_model(inceptionv3, to_file='inceptionv3.png', show_shapes=True)

	return inceptionv3

def test(type_of_test):
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
	batch_size = 1
	epochs = 1	

	# pre-process input images and normalization
	for sub in range(len(total_list)):

	# 	# model
		model = type_of_test()
		model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
		clf = SVC(kernel = 'linear', C = 1, decision_function_shape='ovr')
		loso_generator = create_generator_LOSO(total_list, total_labels, classes, sub, train_phase='svc')

		for X, y, non_binarized_y in loso_generator:
			# non_binarized_y = non_binarized_y[0]
			spatial_features = model.predict(X, batch_size = batch_size)

			clf.fit(spatial_features, non_binarized_y)


		# Resource Clear up
		del X, y

		# Test Time 
		test_loso_generator = create_generator_LOSO(total_list, total_labels, classes, sub, train_phase = False)
		

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
			file = open(root_dir + 'Classification/' + 'Result/'+ 'Combined Dataset' + '/f1_' + str(train_id) +  '.txt', 'a')
			file.write(str(f1) + "\n")
			file.close()
			war = weighted_average_recall(tot_mat, classes, len(non_binarized_y))
			uar = unweighted_average_recall(tot_mat, classes)
			print("war: " + str(war))
			print("uar: " + str(uar))

		# Resource CLear up
		del X, y, non_binarized_y	


	return f1, war, uar, tot_mat

f1, war, uar, tot_mat = test(test_res50_imagenet)
f1_2, war_2, uar_2, tot_mat_2 = test(test_vgg16_imagenet)
f1_3, war_3, uar_3, tot_mat_3 = test(test_vgg19_imagenet)
f1_4, war_4, uar_4, tot_mat_4 = test(test_inceptionv3_imagenet)

print("RESULTS FOR RES 50")
print("F1: " + str(f1))
print("war: " + str(war))
print("uar: " + str(uar))
print(tot_mat)

print("RESULTS FOR VGG 16")
print("F1: " + str(f1_2))
print("war: " + str(war_2))
print("uar: " + str(uar_2))
print(tot_mat_2)

print("RESULTS FOR VGG 19")
print("F1: " + str(f1_3))
print("war: " + str(war_3))
print("uar: " + str(uar_3))
print(tot_mat_3)

print("RESULTS FOR InceptionV3")
print("F1: " + str(f1_4))
print("war: " + str(war_4))
print("uar: " + str(uar_4))
print(tot_mat_4)
# test_res50_imagenet()
# test_vgg16_imagenet()
# test_vgg19_imagenet()
# test_inceptionv3_imagenet()