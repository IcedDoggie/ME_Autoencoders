import numpy as np
import sys
import math
import operator
import csv
import glob,os
import xlrd
import cv2
import pandas as pd
import os, shutil
import glob
import matplotlib.pyplot as plt
import h5py
from imblearn.over_sampling import SMOTE
import math


from sklearn.svm import SVC
from sklearn import preprocessing
from collections import Counter
from sklearn.metrics import confusion_matrix
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import np_utils
from keras import metrics
from keras import backend as K
from keras.models import model_from_json
import keras
import pydot, graphviz
from keras.utils import np_utils, plot_model
from vis.visualization import visualize_cam

def read_image_emotion_class_based(root_db, casme2_db, casme2_table):

	casme_namelist = []
	casme_labels = []
	casme_img_list = []

	data_path = root_db + casme2_db + '/'
	for root, folders, files in os.walk(data_path):
		if root[-1] != '/':
			temp_label_list = []
			temp_img_list = []
			casme_namelist += [files]

			for counter in range(len(files)):
				# read label
				temp_label_list += [root[-1]]

				# read img
				img = cv2.imread(data_path + root[-1] + '/' + files[counter], 0)
				temp_img_list += [img]

			casme_labels += [temp_label_list]
			casme_img_list += [temp_img_list]

	# print(casme_namelist)
	# print(casme_labels)
	# print(casme_img_list)
	# print(len(casme_namelist))
	# print(len(casme_labels))
	# print(len(casme_img_list))
	return casme_namelist, casme_labels, casme_img_list

def load_image(img_namelist):
	img_list = []
	for item in img_namelist:
		img = cv2.imread(item, 0)
		img_list += [img]

	return img_list


def duplicate_labels(imgs_list, labels_list):
	new_labels_list = []
	for count in range(len(imgs_list)):
		imgs = imgs_list[count]
		label = labels_list[count]
		temp_labels_list = []
		for img in imgs:
			# temp_labels_list += [label]
			new_labels_list += [label]
		# new_labels_list += [temp_labels_list]

	return new_labels_list


def display_image_len(images, labels):
	print("%i images, %i labels ck" %(len(images), len(labels)))



def restructure_img(all_images, images_path, emo_labels, total_emotions, img_src_path):

	# create list of emotion folders
	for emo in total_emotions:
		emo_path = images_path + str(emo)

		if os.path.exists(emo_path) == False: 
			os.mkdir(emo_path)

	for counter in range(len(all_images)):
		# check what's the label first before moving
		label = emo_labels[counter]
		img = all_images[counter]
		emo = total_emotions[label]

		img_str = img[len(img_src_path):].replace('/', '_')
		target_path = images_path + str(emo) + "/" + img_str
		shutil.copy(img, target_path)


def binarize_labels(labels_list):
	# np_utils.to_categorical(Train_Y, classes)
	
	lb = preprocessing.LabelBinarizer()
	lb.fit(labels_list)
	labels_list = lb.transform(labels_list)

	return labels_list	

def pivot_class(labels_list):
	list_pivot = []

	for counter in range(len(labels_list)):
		if counter == 0:
			first_label = labels_list[counter]
			list_pivot += [counter]


		else:
			curr_label = labels_list[counter]
			if curr_label != first_label:
				first_label = curr_label
				list_pivot += [counter]

	# where the pivot ends
	list_pivot += [counter]

	# Find biggest class
	king = 0
	for counter in range(len(list_pivot) - 1):
		first_piv = list_pivot[counter]
		second_piv = list_pivot[counter + 1]

		diff = second_piv - first_piv
		if diff > king:
			king = diff
			biggest_pivot = counter
			# print(counter)
	


	return list_pivot, biggest_pivot

def over_sampling(img_list, labels_list, target_path, filename_list, image_name_list=None):

	sm = SMOTE(kind = 'regular', sampling_strategy = 'all', k_neighbors = 1)
	X_resampled = []
	y_resampled = []

	img_list = np.asarray(img_list)
	labels_list = np.asarray(labels_list)

	# create a loop to SMOTE biggest class to other classes (T0D0)
	# visualization purpose
	pca = PCA(n_components = 2)	
	tsne = TSNE(n_components = 2, n_iter=250)


	# find largest class and pivoting
	biggest_class = []
	list_pivot = [0]
	pivot = 0
	maximum_class = 0
	maximum_compare = 0
	for counter in range(len(img_list)):
		if len(img_list[counter]) > maximum_compare:
			maximum_class = counter
			maximum_compare = len(img_list[counter])

	print("Maximum Class: " + str(maximum_class))
	biggest_img_list = img_list[maximum_class]
	biggest_labels_list = labels_list[maximum_class]
	biggest_filename_list = filename_list[maximum_class]


	# changed oversampling method
	img_res_list = []
	labels_res_list = []
	for counter in range(len(img_list) - 1):
		print("\n")
		img_list_temp = img_list[counter]
		labels_list_temp = labels_list[counter]
		filename_list_temp = filename_list[counter]

		img_list_temp = np.concatenate((img_list_temp, biggest_img_list), axis=0)
		img_list_temp = np.reshape(img_list_temp, (img_list_temp.shape[0], img_list_temp.shape[1] * img_list_temp.shape[2] * 1))
		labels_list_temp = np.concatenate((labels_list_temp, biggest_labels_list), axis=0)

		img_res, labels_res = sm.fit_sample(img_list_temp, labels_list_temp)

		img_res = img_res[len(labels_list_temp):]
		labels_res = labels_res[len(labels_list_temp):]

		print("Oversampled")
		# print(labels_res[len(labels_list_temp):])
		print(img_res[len(labels_list_temp):].shape)
		print(labels_res[len(labels_list_temp):].shape)

		img_res_list += [img_res]
		labels_res_list += [labels_res]

		# Vis
		label_majority = biggest_labels_list[0]
		label_minority = labels_list_temp[0]
		img_list_vis = normalize(img_list_temp)
		img_res_vis = normalize(img_res)
		img_list_vis = pca.fit_transform(img_list_vis)
		img_res_vis = pca.fit_transform(img_res_vis)
		visualize_resampled_datapoints(img_list_vis, img_res_vis, labels_list_temp, labels_res, counter, label_majority, label_minority)




	return img_res_list, labels_res_list, biggest_filename_list, biggest_labels_list

def test_over_sampling():
	X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
							   n_informative=3, n_redundant=1, flip_y=0,
							   n_features=20, n_clusters_per_class=1,
							   n_samples=80, random_state=10)

	sm = SMOTE(kind = 'regular')

	X_resampled = []
	y_resampled = []

	pca = PCA(n_components = 2)

	img_vis = pca.fit_transform(X)	

	img_res, labels_res = sm.fit_sample(X, y)
	visualize_resampled_datapoints(X, img_res, y, labels_res)



def visualize_resampled_images(img_list, img_res, labels_list):
	save_path = '/media/ice/OS/Datasets/Siamese Macro-Micro/over_sampled/'
	for counter in range(len(img_list)):
		img = img_list[counter]
		img = img.reshape(224, 224)
		label = labels_list[counter]

		img_r = img_res[counter]
		img_r = img_r.reshape(224, 224)

		# plt.figure()
		# plt.imshow(img, cmap='gray')
		# plt.show()

		plt.imsave(save_path + str(counter) + ".jpg", img, cmap='gray')
		plt.imsave(save_path + str(counter) + "res_.jpg", img_r, cmap='gray')

		print("%i / %i Done" % (counter + 1, len(img_list)))

def visualize_resampled_datapoints(img_vis, img_res_vis, labels_list, labels_res, counter, major, minor):

	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	ax2.axis("off")
	ax4.axis("off")
	c0, c1 = plot_resampling(ax1, img_vis, labels_list, 'Original', major, minor)
	c2, c3 = plot_resampling(ax3, img_res_vis, labels_res, 'Resampled', major, minor)

	ax2.legend((c0, c1), ('Class #0', 'Class #1'), loc='center',
			   ncol=1, labelspacing=0.)	
	ax4.legend((c2, c3), ('Class #0', 'Class #1'), loc='center',
			   ncol=1, labelspacing=0.)	



	plt.tight_layout()
	filename = str(counter) + '.png'
	plt.savefig(filename)
	plt.show()


def plot_resampling(ax, X, y, title, major, minor):


	c0 = ax.scatter(X[y == int(major), 0], X[y == int(major), 1], label="Class #0", alpha=0.5, c='black')
	c1 = ax.scatter(X[y == int(minor), 0], X[y == int(minor), 1], label="Class #1", alpha=0.5, c='yellow')
	
	minimum = np.amin(X)
	maximum = np.amax(X)


	ax.set_title(title)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	ax.spines['left'].set_position(('outward', 10))
	ax.spines['bottom'].set_position(('outward', 10))
	ax.set_xlim([minimum, maximum])
	ax.set_ylim([minimum, maximum])

	return c0, c1

def export_oversampled_images(img_res, label_res, target_oversampled_path, total_emotions):

	for emo in total_emotions:
		emo_path = target_oversampled_path + str(emo) + '/'
		if os.path.exists(emo_path) == False: 
			os.mkdir(emo_path)

	for counter in range(len(img_res)):
		img_list = img_res[counter]
		label_list = label_res[counter]

		for sub_counter in range(len(img_list)):
			img = img_list[sub_counter].reshape((340, 280, 1))
			label = label_list[sub_counter]

			img = np.uint8(img)
			filename = target_oversampled_path + str(label) + '/' + str(counter) + '_' + str(sub_counter) + '.jpg'
			cv2.imwrite(filename, img)
			print(filename)
			print(img.shape)
			print(label) 

		print("%i / %i Done" % (counter + 1, len(img_res)))

