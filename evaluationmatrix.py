import numpy as np
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


def fpr(matrix,n_exp):

	diag = []
	preMatrix = []
	reMatrix = []
	preList = []
	reList = []

	denominator_De_zero = 0.00001

	for index in range(n_exp):
		diag = matrix[index,index]
		col = sum(matrix[:,index])
		row = sum(matrix[index])
		# print(diag)
		# print(col)
		if index == 0:
			prec = diag/(col + denominator_De_zero)
			# print(prec) 
			rec = diag/(row + denominator_De_zero)

			# print(rec)
		else:
			prec = prec + diag/(col + denominator_De_zero) 
			# print(prec)
			rec = rec + diag/(row + denominator_De_zero)
			# print(rec)


	# since we are summing up all precision and recall, need to average it
	precision = prec / (n_exp)
	recall = rec /(n_exp)    
	# print(precision)
	# print(recall)
	f1 = 2 * precision * recall / (precision + recall)

		
	return [f1,precision,recall]

def sklearn_macro_f1(y, pred):
	f1 = f1_score(y, pred, average='macro')
	weighted_f1 = f1_score(y, pred, average='weighted')

	return f1, weighted_f1	



def weighted_average_recall(matrix, n_exp, total_N):
	# normal recognition accuracy
	# war = no. correct classified samples / total number of samples
	number_correct_classified = 0

	for index in range(n_exp):
		diag = matrix[index, index]
		number_correct_classified += diag

	war = number_correct_classified / total_N

	return war

def unweighted_average_recall(matrix, n_exp):
	# balanced recognition accuracy
	# uar = sum(accuracy of each class ) / number of classes
	sum_of_accuracy = 0
	for index in range(n_exp):
		diag = matrix[index, index]
		row = sum(matrix[index])
		accuracy_of_n_class = diag / row
		sum_of_accuracy += accuracy_of_n_class        

	uar = sum_of_accuracy / n_exp


	return uar

def majority_vote(predict, test_X, batch_size, timesteps_TIM):
	# For Majority Vote (make batch size divisible by 10(TIM No.))
	voted_predict = []
	i = 0
	while i < int(len(predict)/timesteps_TIM) - 1:
		fraction_of_predict = predict[i * timesteps_TIM : (i+1) * timesteps_TIM]
		# print(fraction_of_predict)
		fraction_of_predict = np.asarray(fraction_of_predict)
		frequencies = np.bincount(fraction_of_predict)
		highest_frequency = np.argmax(frequencies)
		voted_predict += [highest_frequency]

		i += 1
		if i+1 >= int(len(predict)/timesteps_TIM) :
			fraction_of_predict = predict[(i) * timesteps_TIM : len(predict)]
			fraction_of_predict = np.asarray(fraction_of_predict)
			frequencies = np.bincount(fraction_of_predict)
			highest_frequency = np.argmax(frequencies)
			voted_predict += [highest_frequency]					

	# print(voted_predict)
	predict = voted_predict	

	return predict

def temporal_predictions_averaging(predict, timesteps_TIM):
	average_predict = []
	i = 0
	while i < int(len(predict)/timesteps_TIM):
		fraction_of_predict = predict[i * timesteps_TIM : (i+1) * timesteps_TIM]
		fraction_of_predict = np.asarray(fraction_of_predict)
		prediction = np.sum(fraction_of_predict, axis=0)

		prediction = np.argmax(prediction)
		average_predict += [prediction]

		i += 1

	return average_predict