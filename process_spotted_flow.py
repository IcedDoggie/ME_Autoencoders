import os, cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

path = '/home/ice/Documents/ME_Autoencoders/opticalFlow/'
shortvid_path = '/media/ice/OS/Datasets/Spotting/CASME2_Cropped/CASME2_Cropped/'
output_path = '/home/ice/Documents/ME_Autoencoders/spotted_flow/'

mag_path = path + 'magnitude/'
u_path = path + 'u/'
v_path = path + 'v/'

for root, folders, files in os.walk(mag_path):
	filenames = sorted(files)


for counter in range(len(filenames)):
	output_img = np.zeros((170, 140, 3))

	# Create Directory
	temp_out = filenames[counter].replace('.mat', '')
	temp_out = temp_out.replace("_", "/", 1)
	out = output_path + temp_out
	if os.path.exists(out) == False:
		os.makedirs(out)
	# print(out)


	# Putting the dat together
	mag_dat = mag_path + filenames[counter]
	u_dat = u_path + filenames[counter]
	v_dat = v_path + filenames[counter]

	mag_dat = scipy.io.loadmat(mag_dat)['magnitude']
	u_dat = scipy.io.loadmat(u_dat)['u']
	v_dat = scipy.io.loadmat(v_dat)['v']

	# min-max normalization
	mag_dat = ( mag_dat - np.min(mag_dat) ) / ( np.max(mag_dat) - np.min(mag_dat) )
	u_dat = ( u_dat - np.min(u_dat) ) / ( np.max(u_dat) - np.min(u_dat) )
	v_dat = ( v_dat - np.min(v_dat) ) / ( np.max(v_dat) - np.min(v_dat) )
	
	output_img[:, :, 0] = u_dat
	output_img[:, :, 1] = v_dat
	output_img[:, :, 2] = mag_dat

	output_img = np.uint8(output_img * 255)
	# print(output_img)
	cv2.imwrite(out + "/1.png", output_img)
	print(out + '/1.png')

