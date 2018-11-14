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

from labelling import collectinglabel
from reordering import readinput
from evaluationmatrix import fpr
import itertools

# GTK3Agg GTK3Cairo MacOSX Qt4Agg Qt5Agg TkAgg
# ## WX WXAgg Agg Cairo PS PDF SVG Template

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


# Simple call to plot simple graph
# db_path = '/media/ice/OS/Datasets/Combined_Dataset_Apex_Flow/'
# result_path = db_path + 'Classification/Result/Combined_Dataset_Apex_Flow/'
# train_id = 'Alex_25E'
# plot_scores_and_losses(result_path, train_id)

