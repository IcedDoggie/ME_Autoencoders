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
import h5py, tables
from imblearn.over_sampling import SMOTE
import math
from resnet_builder.resnet import ResnetBuilder


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


def read_image(X_train, X_test):
	img_train_list = []
	counter = 0
	for file in X_train:
		img = cv2.imread(file)
		img_train_list += [img]
		counter += 1
		print(str(counter) + " Done")

	print(len(img_train_list))
	return img_train_list

def restructure_img(all_images, images_path, emo_labels, total_emotions):
	# print(total_emotions)
	# create list of emotion folders
	for emo in total_emotions:
		emo_path = images_path + emo
		if os.path.exists(emo_path) == False: 
			os.mkdir(emo_path)

	for counter in range(len(all_images)):
		# check what's the label first before moving
		label = emo_labels[counter]
		img = all_images[counter]
		emo = total_emotions[label]

		# get extension of the image
		temp_img = img.split('.', 1)[1]
		if len(temp_img) > 4:
			temp_img = temp_img.split('.', 2)[2]


		target_path = images_path + emo + "/" + str(counter) + '.' + temp_img

		shutil.copy(img, target_path)

def restructure_sep_img(img_path, total_emotions, X_train, X_test, y_train_labels, y_test_labels):
	
	train_path = img_path + 'train/'
	test_path = img_path + 'test/'

	if os.path.exists(train_path) == False:
		os.mkdir(train_path)
	if os.path.exists(test_path) == False:
		os.mkdir(test_path)

	for emo in total_emotions:
		emo_path = train_path + emo
		if os.path.exists(emo_path) == False: 
			os.mkdir(emo_path)
		emo_path = test_path + emo
		if os.path.exists(emo_path) == False: 
			os.mkdir(emo_path)

	for counter in range(len(X_train)):
		label = y_train_labels[counter]
		img = X_train[counter]
		emo = total_emotions[label]

		temp_img = img.split('.', 1)[1]
		if len(temp_img) > 4:
			temp_img = temp_img.split('.', 2)[2]

		target_path = train_path + emo + "/" + str(counter) + '.' + temp_img

		shutil.copy(img, target_path)

	for counter in range(len(X_test)):
		label = y_test_labels[counter]
		img = X_test[counter]
		emo = total_emotions[label]

		temp_img = img.split('.', 1)[1]
		if len(temp_img) > 4:
			temp_img = temp_img.split('.', 2)[2]

		target_path = test_path + emo + "/" + str(counter) + '.' + temp_img

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

def over_sampling(img_list, labels_list, image_name_list=None):
	# print(image_name_list)
	sm = SMOTE(kind = 'regular', ratio = 'minority')
	X_resampled = []
	y_resampled = []

	img_list = np.asarray(img_list)
	img_list = np.reshape(img_list, (img_list.shape[0], img_list.shape[1] * img_list.shape[2]))
	
	labels_list = np.asarray(labels_list)

	# separate img_list based on classes, and find the biggest class
	list_pivot, biggest_pivot = pivot_class(labels_list)
	# print(biggest_pivot)

	# create a loop to SMOTE biggest class to other classes (T0D0)
	# visualization purpose
	pca = PCA(n_components = 2)	
	tsne = TSNE(n_components = 2, n_iter=250)
	# print(labels_list)

	biggest_img_list = img_list[ list_pivot[biggest_pivot] : list_pivot[biggest_pivot + 1] ]
	biggest_labels_list = labels_list[ list_pivot[biggest_pivot] : list_pivot[biggest_pivot + 1] ]
	# print(len(biggest_img_list))

	loop_helper = len(list_pivot)
	counter = 0
	img_res_arr = []
	labels_res_arr = []
	while counter < (loop_helper - 2):
	# for counter in range(len(list_pivot) - 2):
		# print(counter)
		if counter != biggest_pivot:
			img_list_temp = img_list[ list_pivot[counter] : list_pivot[counter + 1] ]
			# get subject identity
			seg_image_name_list = image_name_list[ list_pivot[counter] : list_pivot[counter + 1] ]
			# subject_identity = seg_image_name_list
			# print(seg_image_name_list)
			# print(len(seg_image_name_list))
			

			labels_list_temp = labels_list[ list_pivot[counter] : list_pivot[counter + 1] ]

			img_list_temp = np.concatenate((img_list_temp, biggest_img_list), axis=0)
			labels_list_temp = np.concatenate((labels_list_temp, biggest_labels_list), axis=0)

			label_majority = biggest_labels_list[0]
			label_minority = labels_list_temp[0]

			img_res, labels_res = sm.fit_sample(img_list_temp, labels_list_temp)
			
			img_res_arr += [img_res]
			labels_res_arr += [labels_res]

			# dimensionality reduction for visualization
			img_list_temp = normalize(img_list_temp)
			img_res = normalize(img_res)
			img_list_temp = pca.fit_transform(img_list_temp)
			# print("one done")
			img_res = pca.fit_transform(img_res)

			# visualize_resampled_datapoints(img_list_temp, img_res, labels_list_temp, labels_res, counter, label_majority, label_minority)

		
		else:
			loop_helper += 1

		counter += 1
	return img_res_arr, labels_res_arr, biggest_pivot

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
	print(labels_list)
	print(labels_res)
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
	# plt.show()


def plot_resampling(ax, X, y, title, major, minor):
	# print(X[0,0])
	# print(X[0,1])

	c0 = ax.scatter(X[y == int(major), 0], X[y == int(major), 1], label="Class #0", alpha=0.5, c='black')
	c1 = ax.scatter(X[y == int(minor), 0], X[y == int(minor), 1], label="Class #1", alpha=0.5, c='yellow')
	
	minimum = np.amin(X)
	maximum = np.amax(X)
	print(minimum)
	print(maximum)

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


def export_oversampled_images(img_res, label_res, img_path, biggest_pivot):
	# print(len(img_res))
	all_img_shape = (224, 224)
	after_pivot_flag = 0
	for counter in range(len(img_res)):
		img_arr = img_res[counter]
		label_arr = label_res[counter]

		for subcounter in range(len(img_arr)):
			img = img_arr[subcounter]
			label = label_arr[subcounter]

			img_dim = int(math.sqrt(img.shape[0]))
			img = img.reshape(img_dim, img_dim, 1)
			# img = duplicate_channel(img, 2)
			# plt.imshow(img, cmap= 'Greys')
			# plt.show()
			img = np.rollaxis(img, 2)
			# print(img.shape)


			if after_pivot_flag == 1:
				label = label + 2
			elif counter == biggest_pivot:
				label = label + 2
				after_pivot_flag = 1
			else:
				counter_name = counter + 1			

			if label == 0:
				label = 'surprise'
			elif label == 1:
				label = 'disgust'
			elif label == 2:
				label = 'happiness'
			elif label == 3:
				label = 'fear'
			elif label == 4:
				label = 'sadness'
			elif label == 5:
				label = 'anger'			


			out_path = img_path + label + '/' + str(counter) + "_" + str(subcounter) + '.hdf5'
			hdf5_file = h5py.File(out_path, mode='w')
			hdf5_file.create_dataset("all_img", all_img_shape)
			hdf5_file["all_img"][:] = img
		
		print("%i / %i Done" % (counter, len(img_res)))
			# print(img.shape)
			# print(label)

def convert_h5_to_image(h5_list):
	for counter in range(len(h5_list)):
		h5_temp = h5_list[counter]
		# specific situation for oversampled data
		if '_' in h5_temp[-11:-1]:
			hdf5_file = tables.open_file(h5_temp, mode='r')
			img = hdf5_file.root.all_img[:]


		else: 
			hdf5_file  = tables.open_file(h5_temp, mode='r')
			img = hdf5_file.root.all_img[0]		


		img = img.reshape(img.shape[0], img.shape[1], 1)	
		img = np.array(img, dtype=np.uint8)			
		out_path = h5_temp.replace('hdf5', 'jpg')
		print(out_path)


		cv2.imwrite(out_path, img)
		os.remove(h5_temp)
		hdf5_file.close()


