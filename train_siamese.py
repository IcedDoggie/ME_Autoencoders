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

from keras.applications.vgg16 import VGG16

from utilities import loading_smic_table, loading_samm_table, loading_casme_table
from utilities import class_merging, read_image, create_generator_LOSO
from utilities import LossHistory, record_loss_accuracy
from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall
from siamese_models import euclidean_distance, eucl_dist_output_shape, contrastive_loss

def train(type_of_test, train_id, feature_type = 'grayscale', db='Combined_Dataset_Apex', spatial_size = 224, tf_backend_flag = False):

	sys.setrecursionlimit(10000)
	# /media/ice/OS/Datasets/Combined_Dataset_Apex/CASME2_TIM10/CASME2_TIM10	
	# general variables and path
	working_dir = '/home/ice/Documents/ME_Autoencoders/'
	root_dir = '/media/ice/OS/Datasets/' + db + '/'
	weights_path = '/media/ice/OS/Datasets/'
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

	classes = 5
	spatial_size = spatial_size
	channels = 3
	data_dim = 4096
	tot_mat = np.zeros((classes, classes))

