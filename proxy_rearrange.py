import shutil
import os

path = '/media/ice/OS/Datasets/Combined_Dataset_Apex_Flow/smic/'
output_path = '/media/ice/OS/Datasets/Combined_Dataset_Apex_Flow/SMIC_Flow_Strain_Normalized/'

for root, folders, files in os.walk(path):

	for file in files:
		sep = file.split('_', 1)
		subj = sep[0]
		vid_before_sep = sep[1]
		vid_sep = vid_before_sep.split('.', 1)
		vid = vid_sep[0]

		if os.path.exists(output_path + str(subj) + '/') == False:
			os.mkdir(output_path + str(subj) + '/')
		if os.path.exists(output_path + str(subj) + '/' + vid) == False:
			os.mkdir(output_path + str(subj) + '/' + vid)
		print(subj)
		print(vid)

		shutil.copy(path + file, output_path + str(subj) + '/' + vid + '/' + '0.png')