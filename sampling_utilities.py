import numpy as np
import sys
import math
import operator
import csv
import glob,os
import xlrd
import cv2
import pandas as pd
import os, re
import glob
# import matplotlib.pyplot as plt

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
from keras.applications.resnet50 import preprocess_input as res_preprocess_input

from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr
import itertools

def atoi(text):
	return int(text) if text.isdigit() else text
def natural_keys(text):
	return [ atoi(c) for c in re.split('(\d+)',text) ]

def read_image_sequence(root_dir, db, table):
	data_path = root_dir + db + "/" + db + "/"
	img_list = []
	subj_for_loso = ""
	img_list_subpartitioning = []

	label_list = []
	label_list_subpartitioning = []

	
	seq_img = []
	seq_label = []

	for item in table:
		

		if 'SAMM' in db:
			subj = (item[0])[0:3]
			if len(item[0]) < 4:
				vid = item[1]
				label = item[-1]
			else:
				vid = item[0]
				label = item[-1] - 1

		elif 'SMIC' in db:
			subj = (item[0])[1:3]
			if 's' not in subj:
				subj = "s" + subj
			else:
				subj = subj

			if len(item[0]) < 4:
				vid = item[1]
				label = int(item[-1])
			else:
				vid = item[0]
				label = int(item[-1] - 1)

		elif 'CASME' in db:
			if 'sub' not in item[0]:
				subj = "sub" + item[0]
				label = item[-1] - 1
			else:
				subj = item[0]
				label = item[-1]
			vid = item[1]


		# initialization
		if subj_for_loso == "":
			subj_for_loso = subj	

		# push in for first and 2nd subj
		if subj_for_loso != subj and len(img_list_subpartitioning) > 0:

			subj_for_loso = subj
			seq_img = sampling_original_cropped_sequence(files=seq_img, frames_to_sample=10)		
			img_list += [seq_img]
			label_list += [seq_label]
			img_list_subpartitioning = []
			label_list_subpartitioning = []


			seq_img = []
			seq_label = []


		folder_path = data_path + subj + "/" + vid + "/"
		files = os.listdir(folder_path)
		files = sorted(files, key=natural_keys)
		img_list_subpartitioning = []
		label_list_subpartitioning = []

		for file in files:
			temp = folder_path + file
			img_list_subpartitioning += [temp]
			label_list_subpartitioning += [label]


		label_list_subpartitioning = label_list_subpartitioning[0:10]
		seq_img += [img_list_subpartitioning]
		seq_label += [label_list_subpartitioning]
		
	# push in for last subj
	seq_img = sampling_original_cropped_sequence(files=seq_img, frames_to_sample=10)	
	img_list += [seq_img]
	label_list += [seq_label]

	return img_list, label_list

def sampling_original_cropped_sequence(files, frames_to_sample):
	# how to dynamically sample so that they are of equal length
	length = []
	for list_of_files in files:
		length += [len(list_of_files)]
	# print("Minimum: %i, Maximum: %i" % (np.amin(length), np.amax(length)))

	# set 10 as constant frame first
	frames_to_sample = 10

	for counter in range(len(files)):
		original_sequence = files[counter]
		interval = int( len(original_sequence) / frames_to_sample )
		sampling_sequence = original_sequence[::interval]
		sampling_sequence = sampling_sequence[:frames_to_sample]
		files[counter] = sampling_sequence


	return files









