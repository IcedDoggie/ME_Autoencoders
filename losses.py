import keras
from keras import backend as K
import numpy as np

def e2_keras(y_argmax_et_mean_feat, y_feat):
	print("en construction")


def earth_mover_loss(y_true, y_pred):
	cdf_ytrue = K.cumsum(y_true, axis=-1)
	cdf_ypred = K.cumsum(y_pred, axis=-1)
	samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
	print(K.int_shape(samplewise_emd))
	return K.mean(samplewise_emd)


# y_true = np.random.rand(1, 256)
# y_pred = np.random.rand(1, 256)
# print(earth_mover_loss(y_true, y_pred))
# print(earth_mover_loss(y_true, y_pred).shape)

# def inter_class_subject_apperance_variation_loss_function(y, y_mean, y_predicted):

# 	### inputs:
# 	# y: spatial vector of i-th training sample of predicted class c
# 	# y_mean: mean of feature vectors of each class calculated at the beginning of epoch, first epoch is 0
# 	# y_predicted: predicted label

# 	### outputs:
# 	# e_2: inter class emotion apperance variation

# 	# initialize
# 	l2 = nn.MSELoss()
# 	beta = torch.Tensor([1]).cuda() # sharpness parameter, check if needs to be adjust dynamically

# 	# find minimum distance between two different classes
# 	# now just assume a value

# 	# TODO:
# 	# get distance between y_i and y_j for training samples.
# 	# the idea goes: how close is one predicted is further to its nearest predicted class
# 	# from the C-distances, find the minimum as d_c_min
# 	# d_c_min / 2
# 	y_predicted = y_predicted.cpu().numpy()
# 	y_mean = y_mean.detach().numpy()
# 	y_mean = y_mean[0]
# 	# print(y_mean)

# 	min_distance_between_classes = []
# 	for item_counter in range(len(y_mean)):
# 		item = y_mean[item_counter]
# 		temp_y_mean = np.delete(y_mean, item_counter)
# 		min_distance_between_classes += [ min( temp_y_mean - item ) ]


# 	# compute
# 	g_w_list = []

# 	for loss_counter in range(len(y)):
# 		curr_y_predicted = y_predicted[loss_counter]
# 		yc = torch.Tensor([y_mean[curr_y_predicted]]).cuda()
# 		feat_diff = l2(y, yc)
# 		d_c_min = ( min_distance_between_classes[curr_y_predicted] ) ** 2
# 		w_term = feat_diff - d_c_min
# 		g_w = torch.log(1 + torch.exp(beta * w_term)) / beta
# 		g_w_list += [g_w]


# 	g_w = torch.stack(g_w_list)

	
# 	e_2 = 0.5 * torch.sum(g_w)


# 	return e_2