import numpy as np
import pandas as pd
import os, shutil
import cv2

# image_path = '/media/ice/OS/Datasets/CASME2_APEX/CASME2_ORI/'
# label_file = '/media/ice/OS/Datasets/CASME2_APEX/CASME2_label_Ver_2.xls'
# new_dir_path = '/media/ice/OS/Datasets/Combined_Dataset_Apex/CASME2_TIM10/CASME2_TIM10/'
# train_path = '/media/ice/OS/Datasets/CASME2_APEX/Train/'
# objective_label_file = '/media/ice/OS/Datasets/CASME2_APEX/CASME2-ObjectiveClasses.xlsx'


image_path = '/media/ice/OS/Datasets/CASME2_APEX/CASME2_ORI/'
label_file = '/media/ice/OS/Datasets/CASME2_APEX/CASME2_label_Ver_2.xls'
new_dir_path = '/media/ice/OS/Datasets/Combined_Dataset_Apex_Flow/CASME2_CROPPED_APEX/'
train_path = '/media/ice/OS/Datasets/CASME2_APEX/Train/'
objective_label_file = '/media/ice/OS/Datasets/CASME2_APEX/CASME2-ObjectiveClasses.xlsx'

# image_path = '/media/ice/OS/Datasets/SAMM/SAMM/'
# label_file = '/media/ice/OS/Datasets/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx'
# new_dir_path = '/media/ice/OS/Datasets/Combined_Dataset_Apex/SAMM_TIM10/SAMM_TIM10/'
# objective_label_file = '/media/ice/OS/Datasets/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx'


def read_label(label_file):
	table = pd.read_excel(label_file, converters={'Subject': lambda x: str(x), 'ApexFrame': lambda x: str(x)})
	# table = pd.read_excel(label_file)
	# print(table)
	filtered_table = table[['Subject', 'Filename', 'ApexFrame', 'Estimated Emotion']]
	# print(filtered_table)
	# print(table)
	return filtered_table

def read_objective_label(objective_label_file):
	table = pd.read_excel(objective_label_file, converters={'Subject': lambda x: str(x)})
	filtered_table = table[['Subject', 'Filename', 'ApexFrame', 'Objective Class']]
	filtered_table = filtered_table.ix[filtered_table['Objective Class'] < 6]
	# print(filtered_table)
	# print(table)
	return filtered_table

def get_image(image_path, filtered_table, add_sub_flag=True, add_img_flag=True, add_zero_flag = False, samm_flag=False):
	apex_images_list = []
	np_apex = filtered_table[['ApexFrame']].as_matrix()
	np_subj = filtered_table[['Subject']].as_matrix()
	np_filename = filtered_table[['Filename']].as_matrix()
	image_list = []
	counter = 0

	# making new lists
	for np_counter in range(len(np_subj)):
		temp_subj = (np_subj[np_counter])[0]
		temp_filename = (np_filename[np_counter])[0]
		temp_apex = (np_apex[np_counter])[0]

		# adding sub
		if add_sub_flag == True:
			temp_subj = 'sub' + temp_subj
		if add_img_flag == True:
			if int(temp_apex) < 10:
				temp_apex = 'img00' + temp_apex
			elif int(temp_apex) < 100:
				temp_apex = 'img0' + temp_apex				
			else:
				temp_apex = 'img' + temp_apex

		# adding zeros for file less than 100
		if add_zero_flag == True:
			if int(temp_apex) < 10:
				temp_apex = '00' + str(temp_apex)
			elif int(temp_apex) < 100:
				temp_apex = '0' + str(temp_apex)				

		# samm case only
		if samm_flag == True:
			vid_path = image_path + str(temp_subj) + "/" + str(temp_filename) + "/"
			for files in os.walk(vid_path):
				files = files[2]
				for items in files:
					if temp_apex in items:
						temp_apex = items[:-4]


		temp_path = image_path + str(temp_subj) + "/" + str(temp_filename) + "/" + str(temp_apex) + ".jpg"	
		image_list += [temp_path]

	return image_list

def get_objective_image(image_path, train_path, filtered_table):
	apex_images_list = []
	np_apex = filtered_table[['ApexFrame']].as_matrix()
	np_subj = filtered_table[['Subject']].as_matrix()
	np_filename = filtered_table[['Filename']].as_matrix()
	np_objective = filtered_table[['Objective Class']].as_matrix()
	image_list = []
	target_list = []
	counter = 0

	# making new lists
	for np_counter in range(len(np_subj)):
		temp_subj = (np_subj[np_counter])[0]
		temp_filename = (np_filename[np_counter])[0]
		temp_apex = (np_apex[np_counter])[0]
		temp_obj = (np_objective[np_counter])[0]

		# adding zero string
		if temp_apex < 10:
			temp_apex = 'img00' + str(temp_apex)
		elif temp_apex < 100:
			temp_apex = 'img0' + str(temp_apex)
		else:
			temp_apex = 'img' + str(temp_apex)

		if not os.path.isdir(train_path + str(temp_obj)):
			os.mkdir(train_path + str(temp_obj))

		temp_path = image_path + 'sub' + str(temp_subj) + "/" + str(temp_filename) + "/" + str(temp_apex) + ".jpg"	
		image_list += [temp_path]

		target_path = train_path + str(temp_obj) + "/" + str(np_counter) + '.jpg'
		target_list += [target_path]


	return image_list, target_list


def create_dir(new_dir_path, image_path, keyword='sub', position_of_keyword=-5, original_str='CASME2_ORI', new_str='CASME2_APEX', num_words_after_root=6):

	for root, folders, files in os.walk(image_path):
		print(root)
		if root[position_of_keyword:position_of_keyword + len(keyword)] == keyword:
			# print(root)
			temp_root = root.replace(original_str, new_str)
			
			if not os.path.isdir(temp_root):
				os.mkdir(temp_root)



	for root, folders, files in os.walk(image_path):
		if len(root) > len(image_path) + num_words_after_root:
			# print(root)			
			temp_root = root.replace(original_str, new_str)
			# print(temp_root)			
			if not os.path.isdir(temp_root):
				os.mkdir(temp_root)


def create_dir_samm(new_dir_path, image_path, original_str='CASME2_ORI', new_str='CASME2_APEX'):
	for root, folders, files in os.walk(image_path):
		if len(root) < 37:
			temp_root = root.replace(original_str, new_str)
			if not os.path.isdir(temp_root):
				os.mkdir(temp_root)

	for root, folders, files in os.walk(image_path):
		if len(root) > 37:
			temp_root = root.replace(original_str, new_str)
			if not os.path.isdir(temp_root):
				os.mkdir(temp_root)
			


	# for root, folders, files in os.walk(image_path):

	# 	if len(root) > len(image_path) + num_words_after_root:

def copy_apex_images(image_path, output_path, image_list, original_str='CASME2_ORI', new_str='CASME2_APEX'):
	
	for item in image_list:

		temp_item = item.replace(original_str, new_str)
		print(original_str)
		print(new_str)
		print(item)
		print(temp_item)
		shutil.copy(item, temp_item)

	# for root, folders, files in os.walk(output_path):
	# 	if len(root) > 53:	
	# 		if len(files) == 0:
	# 			shutil.rmtree(root)

def copy_apex_images_objective(image_list, target_list):
	
	for counter in range(len(image_list)):
		try:
			shutil.copy(image_list[counter], target_list[counter])
		except:
			print(str(image_list[counter]) + " not exists.")

def rearrange_in_images_based_folder_structure(image_list, labels_list, sub_list):
	print(len(image_list))
	print(len(labels_list))
	print(len(sub_list))
	# print(image_list)
	# print(sub_list)
	
	for counter in range(len(image_list)):
		subj_name = ("sub" + sub_list[counter])[0]
		label_name = str((labels_list[counter])[0])
		ori_image_name = image_list[counter]
		target_path = new_dir_path + label_name + '/'
		image_name = ori_image_name[-10:]
		
		if not os.path.isdir(target_path):
			os.mkdir(target_path)

		target_img = target_path + subj_name + '_' + image_name

		try:
			shutil.copy(ori_image_name, target_img)
		except:
			print(str(ori_image_name) + " not exists.")		
		# print(image_name)
		# print(target_img)

def read_label_onset_apex(label_file):
	table = pd.read_excel(label_file, converters={'Subject': lambda x: str(x), 'ApexFrame': lambda x: str(x), 'OnsetFrame': lambda x: str(x)})
	filtered_table = table[['Subject', 'Filename', 'OnsetFrame', 'ApexFrame', 'Estimated Emotion']]
	return filtered_table

def get_image_onset_apex(image_path, filtered_table, add_sub_flag=True, add_img_flag=True, add_zero_flag = False, samm_flag=False):
	apex_images_list = []
	np_apex = filtered_table[['ApexFrame']].as_matrix()
	np_onset = filtered_table[['OnsetFrame']].as_matrix()	
	np_subj = filtered_table[['Subject']].as_matrix()
	np_filename = filtered_table[['Filename']].as_matrix()
	image_list = []
	image_onset_list = []
	counter = 0

	# making new lists
	for np_counter in range(len(np_subj)):
		temp_subj = (np_subj[np_counter])[0]
		temp_filename = (np_filename[np_counter])[0]
		temp_apex = (np_apex[np_counter])[0]
		temp_onset = (np_onset[np_counter])[0]

		# adding sub
		if add_sub_flag == True:
			temp_subj = 'sub' + temp_subj
		if add_img_flag == True:
			if int(temp_apex) < 10:
				temp_apex = 'img00' + temp_apex

			elif int(temp_apex) < 100:
				temp_apex = 'img0' + temp_apex	

			else:
				temp_apex = 'img' + temp_apex

			if int(temp_onset) < 10:
				temp_onset = 'img00' + temp_onset

			elif int(temp_onset) < 100:
				temp_onset = 'img0' + temp_onset	
						
			else:
				temp_onset = 'img' + temp_onset

		# adding zeros for file less than 100
		if add_zero_flag == True:
			if int(temp_apex) < 10:
				temp_apex = '00' + str(temp_apex)
				temp_onset = '00' + str(temp_onset)
			elif int(temp_apex) < 100:
				temp_apex = '0' + str(temp_apex)				
				temp_onset = '0' + str(temp_onset)

		# samm case only (hasn't added temp_onset)
		if samm_flag == True:
			vid_path = image_path + str(temp_subj) + "/" + str(temp_filename) + "/"
			for files in os.walk(vid_path):
				files = files[2]
				for items in files:
					if temp_apex in items:
						temp_apex = items[:-4]
						# temp_onset = items


		temp_path = image_path + str(temp_subj) + "/" + str(temp_filename) + "/" + str(temp_apex) + ".jpg"	
		temp_path_onset = image_path + str(temp_subj) + "/" + str(temp_filename) + "/" + str(temp_onset) + ".jpg"	
		
		image_list += [temp_path]
		image_onset_list += [temp_path_onset]

	return image_list, image_onset_list

def rename_files_for_flow_compute(image_path):
	for root, folders, files in os.walk(image_path):
		if len(root) > 41:
			for item in range(len(files)):
				filename = root + '/' + '00' + str(item + 1) + '.jpg'
				ori_name = root + '/' + str(item + 1) + '.jpg'
				os.rename(ori_name, filename)

				
# def copy_onset_apex_images():



# filtered_table = read_label(label_file)
# image_list = get_image(image_path, filtered_table, add_sub_flag = True, add_img_flag = True)
# create_dir(new_dir_path, image_path, original_str='CASME2_Cropped/', new_str='Combined_Dataset_Apex/CASME2_CROPPED_APEX/')
# copy_apex_images(image_path, new_dir_path, image_list, original_str='CASME2_Cropped/', new_str='Combined_Dataset_Apex/CASME2_CROPPED_APEX/')


# filtered_table = read_label(label_file)
# image_list = get_image(image_path, filtered_table, add_sub_flag = False, add_img_flag = False, samm_flag = True)
# create_dir_samm(new_dir_path, image_path, original_str='SAMM/SAMM/', new_str='Combined_Dataset_Apex/SAMM_TIM10/SAMM_TIM10/')
# copy_apex_images(image_path, new_dir_path, image_list, original_str='SAMM/SAMM/', new_str='Combined_Dataset_Apex/SAMM_TIM10/SAMM_TIM10/')

# Onset and Apex
filtered_table = read_label_onset_apex(label_file)
image_list, image_onset_list = get_image_onset_apex(image_path, filtered_table, add_sub_flag=True, add_img_flag=True, add_zero_flag = False, samm_flag=False)
create_dir(new_dir_path, image_path, original_str='CASME2_APEX/CASME2_ORI/', new_str='Combined_Dataset_Apex_Flow/CASME2_CROPPED_APEX/')
copy_apex_images(image_path, new_dir_path, image_list, original_str='CASME2_APEX/CASME2_ORI/', new_str='Combined_Dataset_Apex_Flow/CASME2_CROPPED_APEX/')
copy_apex_images(image_path, new_dir_path, image_onset_list, original_str='CASME2_APEX/CASME2_ORI/', new_str='Combined_Dataset_Apex_Flow/CASME2_CROPPED_APEX/')

# rename the files to pass to tvl1flow computation, making life easier
# path = '/media/ice/OS/Datasets/Combined_Dataset_Apex_Flow/CASME2_TIM10/CASME2_TIM10/'
# rename_files_for_flow_compute(path)
