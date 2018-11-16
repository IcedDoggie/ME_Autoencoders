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
# from pynvml.pynvml import *

def read_image(root_dir, db, table):
	data_path = root_dir + db + "/" + db + "/"
	img_list = []
	subj_for_loso = ""
	img_list_subpartitioning = []

	label_list = []
	label_list_subpartitioning = []
	
	for item in table:

		if 'SAMM' in db:
			subj = (item[0])[0:3]
			vid = item[0]
			label = item[-1] - 1

		elif 'SMIC' in db:
			subj = (item[0])[1:3]
			subj = "sub" + subj
			vid = item[0]
			label = int(item[-1]) - 1

		elif 'CASME' in db:
			subj = "sub" + item[0]
			vid = item[1]
			label = item[-1] - 1

		# initialization
		if subj_for_loso == "":
			subj_for_loso = subj	

		# push in for first and 2nd subj
		if subj_for_loso != subj and len(img_list_subpartitioning) > 0:

			subj_for_loso = subj
			img_list += [img_list_subpartitioning]
			label_list += [label_list_subpartitioning]
			img_list_subpartitioning = []
			label_list_subpartitioning = []
			

		folder_path = data_path + subj + "/" + vid + "/"
		files = os.listdir(folder_path)
		for file in files:
			temp = folder_path + file
			img_list_subpartitioning += [temp]
			label_list_subpartitioning += [label]
		
	# push in for last subj
	img_list += [img_list_subpartitioning]
	label_list += [label_list_subpartitioning]

	return img_list, label_list

def create_generator_nonLOSO(x, y, classes, net = 'vgg', spatial_size = 224, train_phase=True):
	# Note: Test will be done separately from Training

	# Filter out only Training Images and Labels
	
	# Read and Yield
	X = []
	Y = []
	non_binarized_Y = []

	for subj_counter in range(len(x)):
		# train case
		if train_phase:

			for each_file in x[subj_counter]:
				image = img.load_img(each_file, target_size=(spatial_size, spatial_size))
				image = img.img_to_array(image)
				image = np.expand_dims(image, axis=0)
				if net == 'res':
					image = res_preprocess_input(image)
				elif net == 'vgg':
					image = preprocess_input(image) # vgg way
				X += [image]

			temp_y = np_utils.to_categorical(y[subj_counter], classes)
			for each_label in temp_y:
				# Y.append(each_label)
				Y += [each_label]

			for item in y[subj_counter]:
				non_binarized_Y += [item]					


	X = np.vstack(X)
	Y = np.vstack(Y)


	if train_phase:
		# print(non_binarized_Y)
		non_binarized_Y = np.vstack(non_binarized_Y) # for sklearn

		yield X, Y, non_binarized_Y
	else:
		non_binarized_Y = np.vstack(non_binarized_Y) # for sklearn
		yield X, Y, non_binarized_Y

def create_generator_LOSO(x, y, classes, sub, net='vgg', spatial_size=224, train_phase='true'):
	# Note: Test will be done separately from Training

	# Filter out only Training Images and Labels


	
	# Read and Yield
	X = []
	Y = []
	non_binarized_Y = []

	for subj_counter in range(len(x)):
		# train case
		if train_phase == 'true':
			if subj_counter != sub:
				for each_file in x[subj_counter]:

					image = img.load_img(each_file, target_size=(spatial_size, spatial_size))
					image = img.img_to_array(image)

					image = np.expand_dims(image, axis=0)
					if net == 'res':
						image = res_preprocess_input(image)
					elif net == 'vgg':
						image = preprocess_input(image)
					X += [image]

				temp_y = np_utils.to_categorical(y[subj_counter], classes)
				for each_label in temp_y:
					# Y.append(each_label)
					Y += [each_label]

		# for svc case
		elif train_phase == 'svc':
			if subj_counter != sub:
				for each_file in x[subj_counter]:

					image = img.load_img(each_file, target_size=(spatial_size, spatial_size))
					image = img.img_to_array(image)
					image = np.expand_dims(image, axis=0)
					if net == 'res':
						image = res_preprocess_input(image)
					elif net == 'vgg':
						image = preprocess_input(image)
					X += [image]

				temp_y = np_utils.to_categorical(y[subj_counter], classes)
				# non_binarized_Y += [y[subj_counter]]

				for item in y[subj_counter]:
					non_binarized_Y += [item]

				for each_label in temp_y:
					# Y.append(each_label)
					Y += [each_label]			
					# non_binarized_Y += [y[subj_counter]]	

		# test case
		else:
			if subj_counter == sub:
				# print(x)
				for each_file in x[subj_counter]:

					image = img.load_img(each_file, target_size=(spatial_size, spatial_size))
					# print(image)
					image = img.img_to_array(image)
					image = np.expand_dims(image, axis=0)
					if net == 'res':
						image = res_preprocess_input(image)
					elif net == 'vgg':
						image = preprocess_input(image)
					X += [image]

				temp_y = np_utils.to_categorical(y[subj_counter], classes)
				for each_label in temp_y:
					# Y.append(each_label)
					Y += [each_label]			
					non_binarized_Y += [y[subj_counter]]

	X = np.vstack(X)
	Y = np.vstack(Y)
	# print("Antoas")

	if train_phase == 'true':
		yield X, Y
	elif train_phase == 'svc':
		# non_binarized_Y = non_binarized_Y[0]
		# print(non_binarized_Y)
		# print(len(non_binarized_Y))
		# non_binarized_Y = np.vstack(non_binarized_Y) # for sklearn
		# print("svc")
		# print(non_binarized_Y)
		yield X, Y, non_binarized_Y
	else:
		non_binarized_Y = np.vstack(non_binarized_Y) # for sklearn
		yield X, Y, non_binarized_Y


def data_loader_with_LOSO(subject, SubjectPerDatabase, y_labels, subjects, classes):
	Train_X = []
	Train_Y = []


	Test_X = np.array(SubjectPerDatabase[subject])
	Test_Y = np_utils.to_categorical(y_labels[subject], classes)
	Test_Y_gt = y_labels[subject]

	########### Leave-One-Subject-Out ###############
	if subject==0:
		for i in range(1,subjects):
			Train_X.append(SubjectPerDatabase[i])
			Train_Y.append(y_labels[i])
	elif subject==subjects-1:
		for i in range(subjects-1):
			Train_X.append(SubjectPerDatabase[i])
			Train_Y.append(y_labels[i])
	else:
		for i in range(subjects):
			if subject == i:
				continue
			else:
				Train_X.append(SubjectPerDatabase[i])
				Train_Y.append(y_labels[i])	
	##################################################


	############ Conversion to numpy and stacking ###############
	Train_X=np.vstack(Train_X)
	Train_Y=np.hstack(Train_Y)
	Train_Y=np_utils.to_categorical(Train_Y, classes)
	#############################################################

	return Train_X, Train_Y, Test_X, Test_Y, Test_Y_gt


def duplicate_channel(X):

	X = np.repeat(X, 3, axis=3)
	# np.set_printoptions(threshold=np.nan)
	# print(X)
	print(X.shape)

	return X

def record_scores(workplace, dB, ct, sub, order, tot_mat, n_exp, subjects):
	if not os.path.exists(workplace+'Classification/'+'Result/'+dB+'/'):
		os.mkdir(workplace+'Classification/'+ 'Result/'+dB+'/')
		
	with open(workplace+'Classification/'+ 'Result/'+dB+'/sub_CT.txt','a') as csvfile:
			thewriter=csv.writer(csvfile, delimiter=' ')
			thewriter.writerow('Sub ' + str(sub+1))
			thewriter=csv.writer(csvfile,dialect=csv.excel_tab)
			for row in ct:
				thewriter.writerow(row)
			thewriter.writerow(order)
			thewriter.writerow('\n')
			
	if sub==subjects-1:
			# compute the accuracy, F1, P and R from the overall CT
			microAcc=np.trace(tot_mat)/np.sum(tot_mat)
			[f1,p,r]=fpr(tot_mat,n_exp)
			print(tot_mat)
			print("F1-Score: " + str(f1))
			# save into a .txt file
			with open(workplace+'Classification/'+ 'Result/'+dB+'/final_CT.txt','w') as csvfile:
				thewriter=csv.writer(csvfile,dialect=csv.excel_tab)
				for row in tot_mat:
					thewriter.writerow(row)
					
				thewriter=csv.writer(csvfile, delimiter=' ')
				thewriter.writerow('micro:' + str(microAcc))
				thewriter.writerow('F1:' + str(f1))
				thewriter.writerow('Precision:' + str(p))
				thewriter.writerow('Recall:' + str(r))			

def loading_smic_labels(root_db_path, dB):

	label_filename = "SMIC_label.xlsx"

	label_path = root_db_path + dB + "/" + label_filename
	label_file = pd.read_excel(label_path)
	label_file = label_file.dropna()

	subject = label_file[['Subject']]
	filename = label_file[['Filename']]
	label = label_file[['Label']]
	num_frames = label_file[['Frames']]

	# print(label_file)
	return subject, filename, label, num_frames

def loading_samm_labels(root_db_path, dB, objective_flag):
	label_filename = 'SAMM_Micro_FACS_Codes_v2.xlsx'

	label_path = root_db_path + dB + "/" + label_filename
	label_file = pd.read_excel(label_path, converters={'Subject': lambda x: str(x)})
	# remove class 6, 7
	# if objective_flag:
		# print(objective_flag)
		# label_file = label_file.ix[label_file['Objective Classes'] < 6]

	subject = label_file[['Subject']]
	filename = label_file[['Filename']]
	label = label_file[['Estimated Emotion']]
	objective_classes = label_file[['Objective Classes']]

	return subject, filename, label, objective_classes

def loading_casme_labels(root_db_path, dB):
	label_filename = 'CASME2_label_Ver_2.xls'

	label_path = root_db_path + dB + "/" + label_filename
	# print(label_path)
	label_file = pd.read_excel(label_path, converters={'Subject': lambda x: str(x)})

	# remove class others
	# label_file = label_file.ix[label_file['Objective Class'] < 6]
	# print(len(label_file)) # 185 samples

	subject = label_file[['Subject']]
	filename = label_file[['Filename']]
	expression_classes = label_file[['Estimated Emotion']]

	return subject, filename, expression_classes


def loading_casme_table(root_db_path, dB):
	subject, filename, expression_classes = loading_casme_labels(root_db_path, dB)
	
	subject = subject.as_matrix()
	filename = filename.as_matrix()
	expression_classes = expression_classes.as_matrix()

	table = np.transpose( np.array( [subject, filename, expression_classes] ) )

	return table



def loading_smic_table(root_db_path, dB):
	subject, filename, label, num_frames = loading_smic_labels(root_db_path, dB)
	filename = filename.as_matrix()
	label = label.as_matrix()

	table = np.transpose( np.array( [filename, label] ) )	
	return table	


def loading_samm_table(root_db_path, dB, objective_flag):	
	subject, filename, label, objective_classes = loading_samm_labels(root_db_path, dB, objective_flag)
	# print("subject:%s filename:%s label:%s objective_classes:%s" %(subject, filename, label, objective_classes))
	subject = subject.as_matrix()
	filename = filename.as_matrix()
	label = label.as_matrix()
	objective_classes = objective_classes.as_matrix()
	table = np.transpose( np.array( [filename, label] ) )
	table_objective = np.transpose( np.array( [subject, filename, objective_classes] ) )
	# print(table)
	return table, table_objective



class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
		self.accuracy = []
		self.epochs = []
	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		self.accuracy.append(logs.get('categorical_accuracy'))
		self.epochs.append(logs.get('epochs'))



def record_loss_accuracy(db_home, train_id, db, history_callback):
	file_loss = open(db_home + 'Classification/' + 'Result/' + db + '/loss_' + str(train_id) + '.txt', 'a')
	file_loss.write(str(history_callback.losses) + "\n")
	file_loss.close()

	file_loss = open(db_home + 'Classification/' + 'Result/' + db + '/accuracy_' + str(train_id) + '.txt', 'a')
	file_loss.write(str(history_callback.accuracy) + "\n")
	file_loss.close()	

	file_loss = open(db_home + 'Classification/' + 'Result/'+ db + '/epoch_' + str(train_id) +  '.txt', 'a')
	file_loss.write(str(history_callback.epochs) + "\n")
	file_loss.close()		

def epoch_analysis(db_home, train_id, db, f1, war, uar, macro_f1, weighted_f1, loss):

	result_folder = db_home + 'Classification/' + 'Result/' + db + '/' + str(train_id) + '/'
	if os.path.isdir(result_folder) == False:
		os.mkdir(result_folder)

	file_loss = open(result_folder + 'microf1_' + str(train_id) + '.txt', 'a')
	file_loss.write(str(f1) + "\n")
	file_loss.close()

	file_loss = open(result_folder + 'war_' + str(train_id) + '.txt', 'a')
	file_loss.write(str(war) + "\n")
	file_loss.close()	

	file_loss = open(result_folder + 'uar_' + str(train_id) +  '.txt', 'a')
	file_loss.write(str(uar) + "\n")
	file_loss.close()	

	file_loss = open(result_folder + 'macrof1_' + str(train_id) + '.txt', 'a')
	file_loss.write(str(macro_f1) + "\n")
	file_loss.close()	

	file_loss = open(result_folder + 'weightedf1_' + str(train_id) +  '.txt', 'a')
	file_loss.write(str(weighted_f1) + "\n")
	file_loss.close()	

	file_loss = open(result_folder + 'losses_' + str(train_id) +  '.txt', 'a')
	file_loss.write(str(loss) + "\n")
	file_loss.close()	

def sanity_check_image(X, channel, spatial_size):
	# item = X[0,:,:,:]
	item = X[0, :, :, 0]

	item = item.reshape(224, 224, channel)

	cv2.imwrite('sanity_check.png', item)



def class_merging(table):
	neg = ['repression', 'disgust', 'anger', 'contempt', 'fear', 'sadness']
	pos = ['happiness']
	other = ['other', 'others']
	rows_to_remove = []
	table = table[0]

	for counter in range(len(table)):
		item = table[counter]
		item[-1] = item[-1].lower()
		if item[-1] in neg:
			table[counter, -1] = 1
		elif item[-1] in pos:
			table[counter, -1] = 2
		elif item[-1] == 'surprise':
			table[counter, -1] = 3
		elif item[-1] in other:
			rows_to_remove += [counter]

	table = np.delete(table, rows_to_remove, 0)
	# print(table)

	return table

def class_discretization(table, db='CASME_2'):
	# neg = ['repression', 'disgust', 'anger', 'contempt', 'fear', 'sadness']
	# pos = ['happiness']
	# other = ['other', 'others']
	rows_to_remove = []
	table = table[0]

	if 'CASME' in db:
		for counter in range(len(table)):
			item = table[counter]
			item[-1] = item[-1].lower()
			if item[-1] == 'happiness':
				table[counter, -1] = 1
			elif item[-1] == 'disgust':
				table[counter, -1] = 2
			elif item[-1] == 'repression':
				table[counter, -1] = 3
			elif item[-1] == 'surprise':
				table[counter, -1] = 4
			elif item[-1] == 'others':
				table[counter, -1] = 5
			elif item[-1] == 'fear' or item[-1] == 'sadness':
				rows_to_remove += [counter]		
		table = np.delete(table, rows_to_remove, 0)	

	elif 'SAMM' in db:
		for counter in range(len(table)):
			item = table[counter]
			item[-1] = item[-1].lower()
			if item[-1] == 'anger':
				table[counter, -1] = 1
			elif item[-1] == 'contempt':
				table[counter, -1] = 2
			elif item[-1] == 'disgust':
				table[counter, -1] = 3
			elif item[-1] == 'fear':
				table[counter, -1] = 4
			elif item[-1] == 'happiness':
				table[counter, -1] = 5	
			elif item[-1] == 'other':
				table[counter, -1] = 6
			elif item[-1] == 'sadness':
				table[counter, -1] = 7		
			elif item[-1] == 'surprise':
				table[counter, -1] = 8



	# table = np.delete(table, rows_to_remove, 0)
	# print(table)

	return table

def reverse_discretization(label, db='CASME'):
	if 'CASME' in db:
		if label == 0:
			label = 'happiness'
		elif label == 1:
			label = 'disgust'
		elif label == 2:
			label = 'repression'
		elif label == 3:
			label = 'surprise'
		elif label == 4:
			label = 'others'									

	return label

# from keras.applications.vgg16 import VGG16

# from keras.layers.core import Dense

# from keras.models import Sequential, Model

# from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv3D, MaxPooling3D, ZeroPadding3D

# root_dir = '/media/ice/OS/Datasets/' + 'Combined_Dataset_Apex_Flow' + '/'
# casme2_db = 'CASME2_Optical'
# casme2_table = loading_casme_table(root_dir, casme2_db)
# casme2_table = class_discretization(casme2_table, 'CASME_2')
# casme_list, casme_labels = read_image(root_dir, casme2_db, casme2_table)

# vgg16 = VGG16(weights = 'imagenet')
# last_layer = vgg16.layers[-2].output
# dense_classifier = Dense(5, activation = 'softmax')(last_layer)
# vgg16 = Model(inputs = vgg16.input, outputs = dense_classifier)	


# # model = train_vgg16_imagenet(classes=5)
# vgg16.load_weights('13.h5')
# gen = create_generator_LOSO(casme_list, casme_labels, classes=5, sub=13, net='vgg', spatial_size=224, train_phase='svc')
# for x, y, non_binarized_y in gen:
# 	print(len(x))
# 	print(len(y))
# 	vgg16.predict(x)