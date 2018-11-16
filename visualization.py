import numpy as np
import sys
import math
import operator
import csv
import glob,os
import xlrd
import cv2
import pandas as pd
import os
import glob
import matplotlib
# matplotlib.use('GTK3Agg')
# matplotlib.use('GTK3Cairo')
# matplotlib.use('MacOSX')
# matplotlib.use('Qt4Agg')
# matplotlib.use('Qt5Agg')
matplotlib.use('TkAgg')
# matplotlib.use('WX')
# matplotlib.use('WXAgg')
# matplotlib.use('Agg')
# matplotlib.use('Cairo')
# matplotlib.use('PS')
# matplotlib.use('PDF')
# matplotlib.use('SVG')
# matplotlib.use('Template')

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import confusion_matrix
import scipy.io as sio


from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import np_utils
from keras import metrics
from keras import backend as K
from keras.models import model_from_json
import keras
import pydot, graphviz
from keras.utils import np_utils, plot_model
from theano import tensor as T

from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr
import itertools
from vis import visualization as vi
from vis.utils import utils 

from networks import train_shallow_alexnet_imagenet
from utilities import loading_casme_table, class_discretization, read_image, create_generator_LOSO
from utilities import reverse_discretization
def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')


def gpu_observer():

	nvmlInit()
	for i in range(nvmlDeviceGetCount()):
		handle = nvmlDeviceGetHandleByIndex(i)
		meminfo = nvmlDeviceGetMemoryInfo(handle)
		print("%s: %0.1f MB free, %0.1f MB used, %0.1f MB total" % (
			nvmlDeviceGetName(handle),
			meminfo.free/1024.**2, meminfo.used/1024.**2, meminfo.total/1024.**2))    



def plot_scores_and_losses(result_path, train_id):

	for root, folders, files in os.walk(result_path + train_id):
		for file in files:
			# scores = np.loadtxt(root + '/' + file, dtype='str')

			# read micro
			if "micro" in file:
				score = np.loadtxt(root + '/' + file)
				plt.figure()
				plt.title('Micro F1')
				lines = plt.plot(range(100), score)
				plt.savefig(train_id + 'microf1.png')
				plt.close()
			# read micro
			elif "macro" in file:
				score = np.loadtxt(root + '/' + file)
				plt.figure()	
				plt.title('Macro F1')			
				lines = plt.plot(range(100), score)
				plt.savefig(train_id + 'macrof1.png')
				plt.close()
			# read micro
			elif "war" in file:
				score = np.loadtxt(root + '/' + file)
				plt.figure()
				plt.title('WAR')
				lines = plt.plot(range(100), score)
				plt.savefig(train_id + 'war.png')
				plt.close()
			# read micro
			elif "uar" in file:
				score = np.loadtxt(root + '/' + file)
				plt.figure()
				plt.title('UAR')
				lines = plt.plot(range(100), score)
				plt.savefig(train_id + 'uar.png')		
				plt.close()	
			# read micro
			elif "losses" in file:
				score = np.loadtxt(root + '/' + file)
				plt.figure()
				plt.title('LOSS')
				lines = plt.plot(range(100), score)
				plt.savefig(train_id + 'losses.png')	
				plt.close()													



def visualize_class_activation_maps(weights_path, model, designated_layer, img_list, img_labels, ori_img, ori_img_path):
	sys.setrecursionlimit(10000)
	no_of_subj = []
	identity_arr = []

	for root, folders, files in os.walk(weights_path):
		no_of_subj = len(files)

	# get subject and vid identity
	for root, folders, files in os.walk(ori_img_path):
		if len(root) > 115:
			identity_idx = root.split('/', 9)
			identity = str(identity_idx[-2]) + '_' + str(identity_idx[-1])
			identity_arr += [identity]


	temp_identity_counter = 0
	for counter in range(no_of_subj):
		weights_name = weights_path + str(counter) + '.h5'
		model.load_weights(weights_name)

		gen = create_generator_LOSO(img_list, img_labels, classes=5, sub=counter, net=None, spatial_size = 227, train_phase = False)
		ori_gen = create_generator_LOSO(ori_img, img_labels, classes=5, sub=counter, net=None, spatial_size = 227, train_phase = False)
		for (alpha, beta) in zip(gen, ori_gen):
			X, y, non_binarized_y = alpha[0], alpha[1], alpha[2]
			X_ori, _, _ = beta[0], beta[1], beta[2]
			print(X.shape)
			print(X_ori.shape)

			predicted_labels = np.argmax(model.predict(X), axis=1)
			non_binarized_y = non_binarized_y[0]

			# visualize CAM
			for img_counter in range(len(predicted_labels)):
				input_img = X[img_counter]
				input_img = input_img.reshape((1, input_img.shape[0], input_img.shape[1], input_img.shape[2]))
				predict = np.argmax(model.predict(input_img), axis=1)

				# utils.apply_modifications(model)
				layer_idx = utils.find_layer_idx(model, 'activation_1')
				penultimate_layer = utils.find_layer_idx(model, 'max_pooling2d_3')
				cam = vi.visualize_cam(model, layer_idx=layer_idx, filter_indices=predict, seed_input=input_img, penultimate_layer_idx=penultimate_layer, \
					backprop_modifier=None, grad_modifier=None)

				# layer_idx = utils.find_layer_idx(model, 'conv_2')
				# cam = vi.visualize_activation(model, layer_idx=layer_idx, filter_indices=None, seed_input=None, \
				# 	backprop_modifier=None, grad_modifier=None)				

				# reshape operation
				input_img = input_img.reshape((input_img.shape[1], input_img.shape[2], input_img.shape[3]))
				input_img = np.transpose(input_img, (1, 2, 0))
				gray_img = X_ori[img_counter]
				gray_img = np.transpose(gray_img, (1, 2, 0))
				# Plotting
				fig, axes = plt.subplots(1, 6, figsize=(18, 6))
				print(input_img.shape)
				txt_X = 0
				txt_Y = 60
				# Reverse Discretization
				predict = reverse_discretization(predict[0])
				label = reverse_discretization(non_binarized_y[img_counter])
				predict_str = "Predicted: " + predict
				label_str = "Label: " + label
				identity_str = identity_arr[temp_identity_counter]
				temp_identity_counter += 1
				
				plt.text(txt_X, txt_Y, identity_str, fontsize=15)
				plt.text(txt_X, txt_Y * 3, predict_str, fontsize=15)
				plt.text(txt_X, txt_Y * 4, label_str, fontsize=15)

				axes[0].imshow(np.uint8(input_img))
				axes[1].imshow(cam)
				axes[2].imshow(vi.overlay(cam, input_img))
				axes[3].imshow(np.uint8(gray_img))
				axes[4].imshow(vi.overlay(cam, gray_img))
				axes[5].imshow(input_img)


				for ax in axes:
					ax.set_xticks([])
					ax.set_yticks([])
					ax.grid(False)

				save_str = '/media/ice/OS/Datasets/Visualizations/CAM_AlexNet_25E/' + identity_str + '.png'
				plt.savefig(save_str)
				print(str(img_counter) + ' / ' + str(len(predicted_labels)))
				# plt.show()


			print("Predicted: ")
			print(predicted_labels)
			print("GroundTruth: ")
			print(non_binarized_y)

		# print(weights_name)

def visualize_activation_maps(weights_path, model, designated_layer, img_list, img_labels, ori_img, ori_img_path):	
	sys.setrecursionlimit(10000)
	no_of_subj = []
	identity_arr = []

	for root, folders, files in os.walk(weights_path):
		no_of_subj = len(files)

	# get subject and vid identity
	for root, folders, files in os.walk(ori_img_path):
		if len(root) > 115:
			identity_idx = root.split('/', 9)
			identity = str(identity_idx[-2]) + '_' + str(identity_idx[-1])
			identity_arr += [identity]


	temp_identity_counter = 0
	for counter in range(no_of_subj):
		weights_name = weights_path + str(counter) + '.h5'
		model.load_weights(weights_name)

		gen = create_generator_LOSO(img_list, img_labels, classes=5, sub=counter, net=None, spatial_size = 227, train_phase = False)
		ori_gen = create_generator_LOSO(ori_img, img_labels, classes=5, sub=counter, net=None, spatial_size = 227, train_phase = False)
		for (alpha, beta) in zip(gen, ori_gen):
			X, y, non_binarized_y = alpha[0], alpha[1], alpha[2]
			X_ori, _, _ = beta[0], beta[1], beta[2]
			# print(X.shape)
			# print(X_ori.shape)

			predicted_labels = np.argmax(model.predict(X), axis=1)
			non_binarized_y = non_binarized_y[0]

			# visualize CAM
			for img_counter in range(len(predicted_labels)):
				input_img = X[img_counter]
				input_img = input_img.reshape((1, input_img.shape[0], input_img.shape[1], input_img.shape[2]))
				predict = np.argmax(model.predict(input_img), axis=1)

				# utils.apply_modifications(model)

				layer_idx = utils.find_layer_idx(model, 'conv_1')
				layer_shape = model.layers[layer_idx].output_shape
				filter_num = layer_shape[1]
				num_rows = 4
				filter_arrange_dim = int(filter_num / num_rows)

				print(filter_num)
				print(layer_shape)
				print(model.summary())

				# Plotting
				print("Constructing plots")
				# fig, axes = plt.subplots(1, 6, figsize=(18, 6))

				fig, axes = plt.subplots(num_rows, filter_arrange_dim, figsize=(100, 100))
				print(axes.shape)
				print(axes[0, :].shape)
				for row_count in range(num_rows):
					for ax in axes[row_count, :]:
						ax.set_xticks([])
						ax.set_yticks([])
						ax.grid(False)
				# for ax in axes[]:
				# 	ax.set_xticks([])
				# 	ax.set_yticks([])
				# 	ax.grid(False)		

				# viualize each filter
				next_row_counter = 0
				curr_row_counter = 0
				for filter_count in range(filter_num):
					print("Visualizing...")
					cam = vi.visualize_activation(model, layer_idx=layer_idx, filter_indices=filter_count, seed_input=None, \
						backprop_modifier=None, grad_modifier=None)				



					print("Plotting...")
					if curr_row_counter >= filter_arrange_dim:
						next_row_counter += 1
						curr_row_counter = 0
					axes[next_row_counter, curr_row_counter].imshow(cam)
					curr_row_counter += 1
					print(str(filter_count) + ' / ' + str(filter_num))

				



				# for ax in axes[0]:
				# 	ax.set_xticks([])
				# 	ax.set_yticks([])
				# 	ax.grid(False)

				# for ax in axes[1]:
				# 	ax.set_xticks([])
				# 	ax.set_yticks([])
				# 	ax.grid(False)
				# save_str = '/media/ice/OS/Datasets/Visualizations/CAM_AlexNet_25E/' + identity_str + '.png'
				# plt.savefig(save_str)
				# print(str(img_counter) + ' / ' + str(len(predicted_labels)))
				plt.savefig('test1.png')
				plt.show()


			print("Predicted: ")
			print(predicted_labels)
			print("GroundTruth: ")
			print(non_binarized_y)

		# print(weights_name)	

def img_label_loading(root_dir, db_type):
	casme2_table = loading_casme_table(root_dir, db_type)
	casme2_table = class_discretization(casme2_table, 'CASME_2')
	casme_list, casme_labels = read_image(root_dir, db_type, casme2_table)
	# print(casme_list)
	# print(casme_labels)

	return casme_list, casme_labels

##### Simple call to plot simple graph #####
# db_path = '/media/ice/OS/Datasets/Combined_Dataset_Apex_Flow/'
# result_path = db_path + 'Classification/Result/Combined_Dataset_Apex_Flow/'
# train_id = 'Alex_25E'
# plot_scores_and_losses(result_path, train_id)

##### Simple call to visualize simple cam #####
weights_path = '/media/ice/OS/Datasets/Weights/alexnet_25E/'
feature_used = '/media/ice/OS/Datasets/Combined_Dataset_Apex_Flow/CASME2_Flow_Strain_Normalized/'
root_dir = '/media/ice/OS/Datasets/Combined_Dataset_Apex_Flow/'
ori_img = '/media/ice/OS/Datasets/Combined_Dataset_Apex/'
model = train_shallow_alexnet_imagenet(classes=5)
casme_list, casme_labels = img_label_loading(root_dir, 'CASME2_Flow_Strain_Normalized')
casme_ori, _ = img_label_loading(ori_img, 'CASME2_TIM10')
# print(casme_list)
# visualize_class_activation_maps(weights_path, model, None, casme_list, casme_labels, casme_ori, feature_used + 'CASME2_Flow_Strain_Normalized/')
visualize_activation_maps(weights_path, model, None, casme_list, casme_labels, casme_ori, feature_used + 'CASME2_Flow_Strain_Normalized/')
