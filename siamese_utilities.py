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
from keras.preprocessing import image as img
from keras.applications.vgg16 import preprocess_input

from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr
import itertools

from augmentation_utilities_macro import over_sampling, pivot_class, binarize_labels, restructure_img
from augmentation_utilities_macro import duplicate_labels, display_image_len, test_over_sampling
from augmentation_utilities_macro import visualize_resampled_images, visualize_resampled_datapoints, plot_resampling
from augmentation_utilities_macro import load_image, export_oversampled_images
from augmentation_utilities_macro import read_image_emotion_class_based
from utilities import loading_casme_table, class_discretization, read_image

def run_over_sampling(feature_type='grayscale'):
	sys.setrecursionlimit(10000)
	root = '/home/ice/Documents/Micro-Expression/'
	root_db = '/media/ice/OS/Datasets/Siamese Macro-Micro/'
	target_path = root_db + 'Micro/'
	target_oversampled_path = root_db + 'Micro_Augmented/'

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

	casme2_table = loading_casme_table(root_db, casme2_db)
	casme2_table = class_discretization(casme2_table, 'CASME_2')	
	casme_list, casme_labels = read_image(root_db, casme2_db, casme2_table)


	temp_list = []
	for img_list in casme_list:
		for img in img_list:
			temp_list += [img]
	casme_list = temp_list
	temp_list = []
	for label_list in casme_labels:
		for label in label_list:
			temp_list += [label]
	casme_labels = temp_list
	total_emotions = np.unique(casme_labels)

	# path of image source, image separation into emotion classes for oversampling
	img_src_path = root_db + casme2_db + '/' + casme2_db + '/'
	restructure_img(casme_list, target_path, casme_labels, total_emotions, img_src_path)
	print(len(casme_list))
	print(len(casme_labels))
	casme_filename_list = casme_list

	# oversampling
	casme_list_emotion, casme_label_emotion, casme_img_list = read_image_emotion_class_based(root_db, 'Micro', casme2_table)	
	img_res_arr, labels_res_arr, biggest_filename_list, biggest_labels_list = over_sampling(casme_img_list, casme_label_emotion, target_oversampled_path, casme_list_emotion)
	export_oversampled_images(img_res_arr, labels_res_arr, target_oversampled_path, total_emotions)

	# export over_sampled images

def blurring(img_list, img_labels, data_filenames, target_path, mode='micro'):
	filename_counter = 0
	for namelist_counter in range(len(img_list)):
		img_namelist = img_list[namelist_counter]
		label_namelist = img_labels[namelist_counter]

		if mode == 'micro':
			for img_counter in range(len(img_namelist)):
				img = img_namelist[img_counter]
				label = label_namelist[img_counter]
				img = cv2.imread(img, 0)
				blurred_img = cv2.blur(img, (5, 5))


				subj = (data_filenames[filename_counter])[0]
				vid = (data_filenames[filename_counter])[1]
				new_filename = target_path + 'sub' + str(subj) + '/' + vid + '/' + str(filename_counter) + '.jpg'
				print(new_filename)
				cv2.imwrite(new_filename, blurred_img)
				filename_counter += 1		

		else:
			img = cv2.imread(img_namelist, 0)
			blurred_img = cv2.blur(img, (5, 5))
			label = label_namelist
			new_filename = target_path + str(label) + '/' + 'blurring_' + str(filename_counter) + '.jpg'

			print(new_filename)
			cv2.imwrite(new_filename, blurred_img)
			filename_counter += 1

def image_contrast(img_list, img_labels, data_filenames, target_path, contrast_count = 4, mode='micro'):
	filename_counter = 0
	for namelist_counter in range(len(img_list)):
		img_namelist = img_list[namelist_counter]
		label_namelist = img_labels[namelist_counter]

		if mode == 'micro':
			for img_counter in range(len(img_namelist)):
				img = img_namelist[img_counter]
				label = label_namelist[img_counter]
				img = cv2.imread(img, 0)

				for contrast_counter in range(contrast_count):
					contrast_img = img + (contrast_counter + 1) * 5

					subj = (data_filenames[filename_counter])[0]
					vid = (data_filenames[filename_counter])[1]
					new_filename = target_path + 'sub' + str(subj) + '/' + vid + '/' + 'contrast_' + str(contrast_counter) + '.jpg'
					
					print(new_filename)
					cv2.imwrite(new_filename, contrast_img)
				filename_counter += 1	

		else:
			img = cv2.imread(img_namelist, 0)
			label = label_namelist			
			for contrast_counter in range(contrast_count):
				contrast_img = img + (contrast_counter + 1) * 1			
				new_filename = target_path + str(label) + '/' + 'contrast_' + str(filename_counter) + '.jpg'

				print(new_filename)
				cv2.imwrite(new_filename, contrast_img)
				filename_counter += 1	

def gaussian_noise(img_list, img_labels, data_filenames, target_path, mode='micro'):
	filename_counter = 0	
	for namelist_counter in range(len(img_list)):
		img_namelist = img_list[namelist_counter]
		label_namelist = img_labels[namelist_counter]

		if mode == 'micro':
			for img_counter in range(len(img_namelist)):
				img = img_namelist[img_counter]
				label = label_namelist[img_counter]
				img = cv2.imread(img, 0)

				# gaussian config
				mu, sigma = 0, 0.1
				gaussian_noise = np.random.normal(mu, sigma, size=(img.shape[0], img.shape[1]))
				gaussian_noise_img = img + gaussian_noise

				subj = (data_filenames[filename_counter])[0]
				vid = (data_filenames[filename_counter])[1]
				new_filename = target_path + 'sub' + str(subj) + '/' + vid + '/' + 'gaussian_' + str(filename_counter) + '.jpg'
				
				print(new_filename)
				cv2.imwrite(new_filename, gaussian_noise_img)
				filename_counter += 1				

		else:
			img = cv2.imread(img_namelist, 0)
			label = label_namelist			
			# gaussian config
			mu, sigma = 0, 0.1
			gaussian_noise = np.random.normal(mu, sigma, size=(img.shape[0], img.shape[1]))
			gaussian_noise_img = img + gaussian_noise			
			
			new_filename = target_path + str(label) + '/' + 'gaussian_' + str(filename_counter) + '.jpg'

			print(new_filename)
			cv2.imwrite(new_filename, gaussian_noise_img)
			filename_counter += 1	




def run_image_averaging(feature_type = 'grayscale'):

	sys.setrecursionlimit(10000)
	root = '/home/ice/Documents/Micro-Expression/'
	root_db = '/media/ice/OS/Datasets/Siamese Macro-Micro/'
	target_path = root_db + 'Micro/'
	target_oversampled_path = root_db + 'Micro_Augmented/'
	me_path = '/media/ice/OS/Datasets/Siamese Macro-Micro/CASME2_TIM10/CASME2_TIM10/'


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

	casme2_table = loading_casme_table(root_db, casme2_db)
	casme2_table = class_discretization(casme2_table, 'CASME_2')	
	casme_list, casme_labels = read_image(root_db, casme2_db, casme2_table)
	# print(casme2_table)
	filenames = casme2_table[:, 0:2]
	filename_counter = 0

	# create subjects folder in augmentation folder
	for root, folders, files in os.walk(me_path):
		if len(root) == 74:
			new_path = root.replace('CASME2_TIM10/CASME2_TIM10/', 'Micro_Augmented/')
			if os.path.exists(new_path) == False:
				os.mkdir(new_path)

		if len(root) > 74:
			new_path = root.replace('CASME2_TIM10/CASME2_TIM10/', 'Micro_Augmented/')			
			if os.path.exists(new_path) == False:
				os.mkdir(new_path)
			

	# blurring
	blurring(casme_list, casme_labels, filenames, target_oversampled_path)



	# image contrast augmentation
	image_contrast(casme_list, casme_labels, filenames, target_oversampled_path, contrast_count=4)


	# Gaussian Noise
	gaussian_noise(casme_list, casme_labels, filenames, target_oversampled_path)

	

# run_over_sampling(feature_type='grayscale')
# run_image_averaging(feature_type='grayscale')