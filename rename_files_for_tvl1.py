import os, shutil, re, cv2



def atoi(text):

	return int(text) if text.isdigit() else text
def natural_keys(text):
	return [ atoi(c) for c in re.split('(\d+)',text) ]

def rename_files():
	path = '/media/viprlab/01D31FFEF66D5170/Ice/CASMEII_Cropped_Grayscale/CASMEII_Cropped_Grayscale/'

	counter = 1
	single_digit = '00'
	double_digit = '0'


	for root, folders, files in os.walk(path):
		helper_str = root.replace(path, '')
		helper_str = helper_str.split('/')
		if len(helper_str) > 1:
			counter = 1
			files = sorted(files, key=natural_keys)

			for file in files:
				source_filename = root + '/' + file

				if counter < 10:
					target_key_string = '00' + str(counter) + '.jpg'
				elif counter < 100:
					target_key_string = double_digit + str(counter) + '.jpg'


				target_filename = root + '/' + target_key_string
				print(target_filename)
				os.rename(source_filename, target_filename)
				counter += 1



def convert_rgb_to_grayscale():
	path = '/media/viprlab/01D31FFEF66D5170/Ice/CASMEII_Cropped_Grayscale/CASMEII_Cropped_Grayscale/'

	for root, folders, files in os.walk(path):
		helper_str = root.replace(path, '')
		helper_str = helper_str.split('/')

		if len(helper_str) > 1:
			for file in files:
				img_file = root + '/' + file
				img = cv2.imread(img_file, 0)
				cv2.imwrite(img_file, img)

# convert_rgb_to_grayscale()
rename_files()