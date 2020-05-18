import cv2
import numpy as np
import matplotlib.pyplot as plt

def cut_algorithm_column(cut_interval, img, spatial_size):

	# for even numbered cutting
	cut_1 = img[:, ::cut_interval, :]

	# for odd numbered cutting
	odd_img = img[:, 1:, :]
	cut_2 = odd_img[:, ::cut_interval, :]
	# print(cut_1.shape)
	# print(cut_2.shape)

	cut_1 = cv2.resize(cut_1, (spatial_size, spatial_size))
	cut_2 = cv2.resize(cut_2, (spatial_size, spatial_size))

	# print(cut_1.shape)
	# print(cut_2.shape)	

	return cut_1, cut_2

def cut_algorithm_row(cut_interval, img, spatial_size):

	# for even numbered cutting
	cut_1 = img[::cut_interval, :, :]

	# for odd numbered cutting
	odd_img = img[1:, :, :]
	cut_2 = odd_img[::cut_interval, :, :]
	# print(cut_1.shape)
	# print(cut_2.shape)


	cut_1 = cv2.resize(cut_1, (spatial_size, spatial_size))
	cut_2 = cv2.resize(cut_2, (spatial_size, spatial_size))

	# print(cut_1.shape)
	# print(cut_2.shape)

	return cut_1, cut_2

def cut_algorithm_slant(cut_interval, img, spatial_size):
	# for / cutting (even)
	cut_1 = img[::cut_interval, ::cut_interval, :]

	# # for / cutting (odd)
	odd_img = img[1:, 1:, :]
	cut_2 = odd_img[::cut_interval, ::cut_interval, :]

	# # for \ cutting (even)
	# cut_3 = img[cut_interval::, ::cut_interval]

	# # for \ cutting (odd)
	# odd_img = img[1:, 1:, :]
	# cut_4 = odd_img[cut_interval::, ::cut_interval]
	# print(cut_1.shape)
	# print(cut_2.shape)

	cut_1 = cv2.resize(cut_1, (spatial_size, spatial_size))
	cut_2 = cv2.resize(cut_2, (spatial_size, spatial_size))
	# cut_3 = cv2.resize(cut_3, (spatial_size, spatial_size))
	# cut_4 = cv2.resize(cut_4, (spatial_size, spatial_size))

	# print(cut_1.shape)
	# print(cut_2.shape)


	return cut_1, cut_2

def cut_algorithm_call_all(cut_interval, spatial_size, img):

	img_list = []
	img = img[0]

	img = np.rollaxis(img, 0, 3)

	if type(cut_interval) == int:
		cut_1, cut_2 = cut_algorithm_column(cut_interval=2, img=img, spatial_size=spatial_size)
		cut_3, cut_4 = cut_algorithm_row(cut_interval=2, img=img, spatial_size=spatial_size)
		cut_5, cut_6 = cut_algorithm_slant(cut_interval=2, img=img, spatial_size=spatial_size)	
		img_list += [cut_1]	
		img_list += [cut_2]	
		img_list += [cut_3]	
		img_list += [cut_4]	
		img_list += [cut_5]	
		img_list += [cut_6]	
		


	elif type(cut_interval) == list:
		for interval in cut_interval:
			cut_1, cut_2 = cut_algorithm_column(cut_interval=interval, img=img, spatial_size=spatial_size)
			cut_3, cut_4 = cut_algorithm_row(cut_interval=interval, img=img, spatial_size=spatial_size)
			cut_5, cut_6 = cut_algorithm_slant(cut_interval=interval, img=img, spatial_size=spatial_size)	
			img_list += [cut_1]	
			img_list += [cut_2]	
			img_list += [cut_3]	
			img_list += [cut_4]	
			img_list += [cut_5]	
			img_list += [cut_6]	

	img_list += [img]
	img_list = np.asarray(img_list)
	img_list = np.rollaxis(img_list, 3, 1)



	return img_list			


# img = cv2.imread('46.jpg')
# img = cv2.resize(img, (128, 128))



# cut_1, cut_2 = cut_algorithm_column(cut_interval=4, img=img)
# cut_3, cut_4 = cut_algorithm_row(cut_interval=4, img=img)
# cut_5, cut_6, cut_7, cut_8 = cut_algorithm_slant(cut_interval=14, img=img)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cut_1 = cv2.cvtColor(cut_1, cv2.COLOR_BGR2RGB)
# cut_2 = cv2.cvtColor(cut_2, cv2.COLOR_BGR2RGB)
# cut_3 = cv2.cvtColor(cut_3, cv2.COLOR_BGR2RGB)
# cut_4 = cv2.cvtColor(cut_4, cv2.COLOR_BGR2RGB)
# cut_5 = cv2.cvtColor(cut_5, cv2.COLOR_BGR2RGB)
# cut_6 = cv2.cvtColor(cut_6, cv2.COLOR_BGR2RGB)
# cut_7 = cv2.cvtColor(cut_7, cv2.COLOR_BGR2RGB)
# cut_8 = cv2.cvtColor(cut_8, cv2.COLOR_BGR2RGB)

# fig, axes = plt.subplots(1, 9, figsize=(18, 6))

# axes[0].imshow(img)
# axes[1].imshow(cut_1)
# axes[2].imshow(cut_2)
# axes[3].imshow(cut_3)
# axes[4].imshow(cut_4)
# axes[5].imshow(cut_5)
# axes[6].imshow(cut_6)
# axes[7].imshow(cut_7)
# axes[8].imshow(cut_8)

# for ax in axes:
# 	ax.set_xticks([])
# 	ax.set_yticks([])
# 	ax.grid(False)	
# plt.show()
# plt.close()
