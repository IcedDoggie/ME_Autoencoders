import os, cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

path = '/home/ice/Documents/ME_Autoencoders/spottedApex_x4s_a28x84_max_max_conc_3_5_8_3x3_max_fc512'
shortvid_path = '/media/ice/OS/Datasets/Spotting/CASME2_Cropped/CASME2_Cropped/'
output_path = '/home/ice/Documents/ME_Autoencoders/spotted_apex_approx_onset/'

# CASME II Case
apex_arr = []
sample_arr = [] # there are samples not existent in DB
for root, folders, files in os.walk(path):
	files = sorted(files)

	for file in files:
		dat_path = root + '/' + file
		dat = (scipy.io.loadmat(dat_path))['YPredicted']
		apex_arr += [np.argmax(dat)]
		file_str = file.replace(".mat", "")
		file_str = file_str.replace("_", "/", 1)
		sample_arr += [file_str]

# print(sample_arr)
# print(apex_arr)

for item_count in range(len(sample_arr)):
	img_path = shortvid_path + sample_arr[item_count] + '/'
	img_outpath = output_path + sample_arr[item_count] + '/'
	if os.path.exists(img_outpath) == False:
		os.makedirs(img_outpath)
	for root, folders, files in os.walk(img_path):
		files = sorted(files)
		file = files[apex_arr[item_count]]
		onset_file = files[apex_arr[item_count] - 1]


		# if apex is the first frame
		if file == files[0]: 
			img = cv2.imread(img_path + files[apex_arr[item_count] + 1])
			img_onset = cv2.imread(img_path + file)
			outfile = img_outpath + files[apex_arr[item_count] + 1]
			outfile_onset = img_outpath + file		
			
		else:
			img = cv2.imread(img_path + file)
			img_onset = cv2.imread(img_path + onset_file)
			outfile = img_outpath + file
			outfile_onset = img_outpath + onset_file

		# outfile = img_outpath + file
		# outfile_onset = img_outpath + onset_file

		cv2.imwrite(outfile, img)
		cv2.imwrite(outfile_onset, img_onset)
		print(outfile_onset)
		print(outfile)
		print("\n")