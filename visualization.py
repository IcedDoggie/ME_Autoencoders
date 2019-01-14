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

from networks import train_shallow_alexnet_imagenet, train_dual_stream_shallow_alexnet
from utilities import loading_casme_table, class_discretization, read_image, create_generator_LOSO
from utilities import reverse_discretization
def plot_confusion_matrix(cm, classes,
						  normalize=True,
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
		# print(cm[i, j])
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	plt.show()


def gpu_observer():

	nvmlInit()
	for i in range(nvmlDeviceGetCount()):
		handle = nvmlDeviceGetHandleByIndex(i)
		meminfo = nvmlDeviceGetMemoryInfo(handle)
		print("%s: %0.1f MB free, %0.1f MB used, %0.1f MB total" % (
			nvmlDeviceGetName(handle),
			meminfo.free/1024.**2, meminfo.used/1024.**2, meminfo.total/1024.**2))    



def plot_scores_and_losses(result_path, train_id, range_len = 100):
	
	for root, folders, files in os.walk(result_path + train_id):
		for file in files:
			# scores = np.loadtxt(root + '/' + file, dtype='str')

			# read micro
			if "micro" in file:
				score = np.loadtxt(root + '/' + file)
				plt.figure()
				plt.title('Micro F1')
				lines = plt.plot(range(range_len), score)
				plt.savefig(train_id + 'microf1.png')
				plt.close()
			# read micro
			elif "macro" in file:
				score = np.loadtxt(root + '/' + file)
				plt.figure()	
				plt.title('Macro F1')			
				lines = plt.plot(range(range_len), score)
				plt.savefig(train_id + 'macrof1.png')
				plt.close()
			# read micro
			elif "war" in file:
				score = np.loadtxt(root + '/' + file)
				plt.figure()
				plt.title('WAR')
				lines = plt.plot(range(range_len), score)
				plt.savefig(train_id + 'war.png')
				plt.close()
			# read micro
			elif "uar" in file:
				score = np.loadtxt(root + '/' + file)
				plt.figure()
				plt.title('UAR')
				lines = plt.plot(range(range_len), score)
				plt.savefig(train_id + 'uar.png')		
				plt.close()	
			# read micro
			elif "losses" in file:
				score = np.loadtxt(root + '/' + file)
				plt.figure()
				plt.title('LOSS')
				lines = plt.plot(range(range_len), score)
				plt.savefig(train_id + 'losses.png')	
				plt.close()													



def visualize_class_activation_maps(weights_path, model, designated_layer, img_list, img_labels, ori_img, ori_img_path):
	sys.setrecursionlimit(10000)
	no_of_subj = []
	identity_arr = []

	for root, folders, files in os.walk(weights_path):
		no_of_subj = len(files)
	# print(no_of_subj)
	# get subject and vid identity
	for root, folders, files in os.walk(ori_img_path):
		if len(root) > 85:
			# print(root)

			identity_idx = root.split('/', 9)
			identity = str(identity_idx[-2]) + '_' + str(identity_idx[-1])
			identity_arr += [identity]
			# print(identity)

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
				# layer_idx = model.layers[-1]
				penultimate_layer = utils.find_layer_idx(model, 'max_pooling2d_3')
				# penultimate_layer = model.layers[-5]
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
				
				# plt.text(txt_X, txt_Y, identity_str, fontsize=15)
				# plt.text(txt_X, txt_Y * 3, predict_str, fontsize=15)
				# plt.text(txt_X, txt_Y * 4, label_str, fontsize=15)

				# axes[0].imshow(np.uint8(input_img))
				# axes[1].imshow(cam)
				# axes[2].imshow(vi.overlay(cam, input_img))
				# axes[3].imshow(np.uint8(gray_img))
				# axes[4].imshow(vi.overlay(cam, gray_img))
				# axes[5].imshow(input_img)

				plt.imshow(vi.overlay(cam, gray_img))	
				plt.tick_params(
					axis='both',          # changes apply to the x-axis
					which='both',      # both major and minor ticks are affected
					bottom=False,      # ticks along the bottom edge are off
					top=False,         # ticks along the top edge are off
					left=False,
					right=False,
					labelleft=False,
					labelbottom=False,) # labels along the bottom edge are off

				# for ax in axes:
				# 	ax.set_xticks([])
				# 	ax.set_yticks([])
				# 	ax.grid(False)

				identity_str = identity_str + "_predict_" + predict + "_label_" + label
				save_str = '/media/ice/OS/Datasets/Visualizations/CAM_AlexNet_50/' + identity_str + '.png'
				plt.savefig(save_str)
				print(str(img_counter) + ' / ' + str(len(predicted_labels)))
				# plt.show()


			print("Predicted: ")
			print(predicted_labels)
			print("GroundTruth: ")
			print(non_binarized_y)

		# print(weights_name)

def visualize_class_activation_maps_dual_stream(weights_path, model, designated_layer, img_list, img_labels, img_list2, img_labels2, ori_img, ori_img_path):
	sys.setrecursionlimit(10000)
	no_of_subj = []
	identity_arr = []

	for root, folders, files in os.walk(weights_path):
		no_of_subj = len(files)

	# get subject and vid identity
	for root, folders, files in os.walk(ori_img_path):
		if len(root) > 85:
			# print(root)

			identity_idx = root.split('/', 9)
			identity = str(identity_idx[-2]) + '_' + str(identity_idx[-1])
			identity_arr += [identity]
			# print(identity)

	temp_identity_counter = 0
	for counter in range(no_of_subj):
		weights_name = weights_path + str(counter) + '.h5'
		model.load_weights(weights_name)

		gen = create_generator_LOSO(img_list, img_labels, classes=5, sub=counter, net=None, spatial_size = 227, train_phase = False)
		gen2 = create_generator_LOSO(img_list2, img_labels2, classes=5, sub=counter, net=None, spatial_size = 227, train_phase = False)
		ori_gen = create_generator_LOSO(ori_img, img_labels, classes=5, sub=counter, net=None, spatial_size = 227, train_phase = False)
		for (alpha, beta, omega) in zip(gen, ori_gen, gen2):
			X, y, non_binarized_y = alpha[0], alpha[1], alpha[2]
			X_ori, _, _ = beta[0], beta[1], beta[2]
			X_2, _, _ = omega[0], omega[1], omega[2]
			print(X.shape)
			print(X_ori.shape)
			print(X_2.shape)

			predicted_labels = np.argmax(model.predict([X, X_2]), axis=1)
			non_binarized_y = non_binarized_y[0]

			# visualize CAM
			for img_counter in range(len(predicted_labels)):
				input_img = X[img_counter]
				input_img2 = X_2[img_counter]
				input_img = input_img.reshape((1, input_img.shape[0], input_img.shape[1], input_img.shape[2]))
				input_img2 = input_img2.reshape((1, input_img2.shape[0], input_img2.shape[1], input_img2.shape[2]))

				predict = np.argmax(model.predict([input_img, input_img2]), axis=1)

				# utils.apply_modifications(model)
				model_mag = model.get_layer('model_5')
				model_mag = Model(inputs = model_mag.get_input_at(0), outputs = model_mag.get_output_at(0))
				model_strain = model.get_layer('model_6')
				model_strain = Model(inputs = model_strain.get_input_at(0), outputs = model_strain.get_output_at(0))

				# for non-pretrained weights
				# pen_mag = utils.find_layer_idx(model_mag, 'conv_2')
				# pen_strain = utils.find_layer_idx(model_strain, 'conv_2')

				# for 2 conv pre-trained weights
				pen_mag = utils.find_layer_idx(model, 'model_5')
				pen_strain = utils.find_layer_idx(model, 'model_6')

				plot_model(model_mag, show_shapes = True, to_file = 'model_mag_extract')
				plot_model(model_strain, show_shapes = True, to_file = 'model_strain_extract')

				print(model_mag.inputs)
				print(model_strain.inputs)
				layer_idx = utils.find_layer_idx(model, 'softmax_activate')
				print(pen_mag)
				print(layer_idx)


				cam = vi.visualize_cam(model_mag, layer_idx=layer_idx, filter_indices=predict, seed_input=input_img, penultimate_layer_idx=pen_mag, \
					backprop_modifier=None, grad_modifier=None)
				cam2 = vi.visualize_cam(model_strain, layer_idx=layer_idx, filter_indices=predict, seed_input=input_img2, penultimate_layer_idx=pen_strain, \
					backprop_modifier=None, grad_modifier=None)

				# reshape operation
				input_img = input_img.reshape((input_img.shape[1], input_img.shape[2], input_img.shape[3]))
				input_img = np.transpose(input_img, (1, 2, 0))
				gray_img = X_ori[img_counter]
				gray_img = np.transpose(gray_img, (1, 2, 0))

				cam_temp = cam / 255
				cam2_temp = cam2 / 255

				# min-max
				min_cam = cam_temp.min()
				max_cam = cam_temp.max()
				print(min_cam)
				print(max_cam)

				# # vis multiply
				fused_cam = np.multiply(cam_temp, cam2_temp)
				# vis addition
				# fused_cam = np.add(cam_temp, cam2_temp)


				fused_cam = fused_cam * 255


				fused_cam = np.uint8(fused_cam)

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
				
				# plt.text(txt_X, txt_Y, identity_str, fontsize=15)
				# plt.text(txt_X, txt_Y * 3, predict_str, fontsize=15)
				# plt.text(txt_X, txt_Y * 4, label_str, fontsize=15)

				# axes[0].imshow(vi.overlay(cam, gray_img))
				# axes[1].imshow(vi.overlay(cam2, gray_img))
				# axes[1].imshow(cam)
				# axes[2].imshow(vi.overlay(cam, input_img))
				# axes[3].imshow(np.uint8(gray_img))
				# axes[4].imshow(vi.overlay(cam, gray_img))
				# axes[5].imshow(input_img)


				# # # FOR COMPO
				# plt.imshow(vi.overlay(fused_cam, gray_img))

				# FOR CAM AND CAM2
				plt.imshow(vi.overlay(cam, gray_img))



				# plt.imshow(vi.overlay(cam, gray_img))	
				plt.tick_params(
					axis='both',          # changes apply to the x-axis
					which='both',      # both major and minor ticks are affected
					bottom=False,      # ticks along the bottom edge are off
					top=False,         # ticks along the top edge are off
					left=False,
					right=False,
					labelleft=False,
					labelbottom=False,) # labels along the bottom edge are off
				# plt.show()

				# for ax in axes:
				# 	ax.set_xticks([])
				# 	ax.set_yticks([])
				# 	ax.grid(False)

				identity_str = identity_str + "_predict_" + predict + "_label_" + label
				save_str = '/media/ice/OS/Datasets/Visualizations/CAM_shallow_alexnet_multi_31J_MULTIPLY/' + identity_str + '.png'
				plt.savefig(save_str, bbox_inches='tight', pad_inches=0)

				plt.imshow(vi.overlay(cam2, gray_img))
				plt.tick_params(
					axis='both',          # changes apply to the x-axis
					which='both',      # both major and minor ticks are affected
					bottom=False,      # ticks along the bottom edge are off
					top=False,         # ticks along the top edge are off
					left=False,
					right=False,
					labelleft=False,
					labelbottom=False,) # labels along the bottom edge are off
				save_str = '/media/ice/OS/Datasets/Visualizations/CAM_shallow_alexnet_multi_31J_MULTIPLY_2/' + identity_str + '.png'
				plt.savefig(save_str, bbox_inches='tight', pad_inches=0)

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
				num_rows = 24
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
				save_str = '/media/ice/OS/Datasets/Visualizations/CAM_AlexNet_25E/' + str(temp_identity_counter) + '.png'
				temp_identity_counter += 1
				# plt.savefig(save_str)
				# print(str(img_counter) + ' / ' + str(len(predicted_labels)))
				plt.savefig('test1.png')
				# plt.show()


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

# def restructure_dual_stream(model):
# 	# model = Model( inputs = model.get_layer('input_1').get_output_at(0), outputs = model.get_layer('model_5').output )
# 	plot_model(model, to_file = 'restrcture.png', show_shapes=True)

# ##### Simple call to plot simple graph #####
# db_path = '/media/ice/OS/Datasets/Combined_Dataset_Apex_Flow/'
# result_path = db_path + 'Classification/Result/Combined_Dataset_Apex_Flow/'
# train_id = 'shallow_alexnet_multi_28B'
# plot_scores_and_losses(result_path, train_id, range_len = 100)
# train_id = 'shallow_alexnet_multi_28C'
# plot_scores_and_losses(result_path, train_id, range_len = 100)

# vis siamese
# db_path = '/media/ice/OS/Datasets/Siamese Macro-Micro/'
# result_path = db_path + 'Classification/Result/Siamese Macro-Micro/'
# train_id = 'siamese_6'
# plot_scores_and_losses(result_path, train_id)
# train_id = 'siamese_7'
# plot_scores_and_losses(result_path, train_id)
# train_id = 'siamese_9'
# plot_scores_and_losses(result_path, train_id, range_len = 80)

##### Simple call to visualize simple cam #####
weights_path = '/media/ice/OS/Datasets/Weights/shallow_alexnet_multi_31J_MULTIPLY/'
# weights_path = '/media/ice/OS/Datasets/Weights/shallow_alexnet_50/'
feature_used = '/media/ice/OS/Datasets/Combined_Dataset_Apex_Flow/CASME2_Optical/'
# feature_used = '/media/ice/OS/Datasets/Combined_Dataset_Apex_Flow/CASME2_Optical_Gray_Weighted/'
root_dir = '/media/ice/OS/Datasets/Combined_Dataset_Apex_Flow/'
ori_img = '/media/ice/OS/Datasets/Combined_Dataset_Apex/'
model = train_dual_stream_shallow_alexnet(classes=5)
# model = train_shallow_alexnet_imagenet(classes=5)
casme_list, casme_labels = img_label_loading(root_dir, 'CASME2_Optical')
casme_ori, _ = img_label_loading(ori_img, 'CASME2_TIM10')

casme2_list, casme2_labels = img_label_loading(root_dir, 'CASME2_Optical_Gray_Weighted')

# # restructure_dual_stream(model_mag)

# # print(casme_list)
visualize_class_activation_maps_dual_stream(weights_path, model, None, casme_list, casme_labels, casme2_list, casme2_labels, casme_ori, feature_used + 'CASME2_Optical/')
# visualize_class_activation_maps(weights_path, model, None, casme_list, casme_labels, casme_ori, feature_used + 'CASME2_Optical/')

