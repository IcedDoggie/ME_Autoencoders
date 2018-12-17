from sklearn.preprocessing import normalize
from skimage import io, exposure
import os
import cv2


db_path = '/media/ice/OS/Datasets/resnet_datasets/images/'

for root, folders, files in os.walk(db_path):
	if len(root) > 46:
		for file in files:
			image_name = root + '/' + file
			image = io.imread(image_name)
			equalized_image = exposure.equalize_hist(image)

			cv2.imwrite(image_name, equalized_image)
			print(image_name)

