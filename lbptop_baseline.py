import numpy as np
from sklearn.svm import SVC
import scipy.io
import pandas as pd
from sklearn.metrics import confusion_matrix

from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall, sklearn_macro_f1


# filename = 'casme2_lbptopR114_N4_noTIM.mat'
# filename = 'casme2_lbptopR114_N4_noTIM_flow.mat'
# filename = 'SMIC_classes_lbptopR114_N4_TIM10.mat'
filename = 'SAMM5_classes_lbptopR114_N4_TIM10.mat'
# filename = 'casme2_lbptopR114_N4_noTIM_SMIC.mat'
# filename = 'casme2_lbptopR114_N4_noTIM_CASME.mat'
# filename = 'CASME2_5classes_lbptopR114_N4_TIM10.mat'

# csv_label = 'combined_3dbs.csv'
# csv_label = 'casme_3_db.txt'
csv_label = 'samm_3_db.txt'
# csv_label = 'casme2_5classes.txt'
# csv_label = 'smic_3_db.txt'

classes = 5
features_arr = []
vid_name_arr = []
labels_arr = []

###################### Reading labels ##########################
table = pd.read_table(csv_label, delimiter=',', header=None, names=['db', 'subj', 'vid', 'label'])
labels = table[['label']].as_matrix()
################################################################

#################### Reading LBPTOP Features #################
table = scipy.io.loadmat(filename)
feats = table['feats']
ids = table['ids']
check_entry = 'sub01'
# check_entry = 's01'
# check_entry = '006'
temp_array_for_loso = []
temp_feat_for_loso = []
labels_for_loso = []
print(len(feats))
print(len(labels))
for counter_idx in range(len(ids)):
	idx = ids[counter_idx]
	feat = feats[counter_idx]
	label = (labels[counter_idx])[0]


	sub = (idx[0])[0]
	vid = (idx[2])[0]
	entry_name = str(sub) + '/' + str(vid)
	if check_entry != sub:
		check_entry = sub
		vid_name_arr += [temp_array_for_loso]
		features_arr += [temp_feat_for_loso]
		labels_arr += [labels_for_loso]
		temp_array_for_loso = []
		temp_feat_for_loso = []
		labels_for_loso = []

	temp_array_for_loso += [entry_name]
	temp_feat_for_loso += [feat]
	labels_for_loso += [label]

vid_name_arr += [temp_array_for_loso]
features_arr += [temp_feat_for_loso]
labels_arr += [labels_for_loso]


vid_name_arr = np.asarray(vid_name_arr)
features_arr = np.asarray(features_arr)	
labels_arr = np.asarray(labels_arr)
print(vid_name_arr)
print(vid_name_arr.shape)
print(features_arr.shape)
print(labels_arr.shape)
# print(features_arr[0])
###############################################################

tot_mat = np.zeros((classes, classes))
total_samples = 0
clf = SVC(kernel = 'linear', C = 1, decision_function_shape='ovr')
pred = []
y_list = []

############# for separate db evaluation ##############
casme_results = np.zeros((classes, classes))
casme_samples = 246
casme_pred = []
casme_y_list = []

samm_results = np.zeros((classes, classes))
samm_samples = 136
samm_pred = []
samm_y_list = []

smic_results = np.zeros((classes, classes))
smic_samples = 164
smic_pred = []
smic_y_list = []

sep_flag = True
#######################################################

# print(vid_name_arr)
for sub in range(len(features_arr)):

	# Train
	training_feat = [x for i, x in enumerate(features_arr) if i != sub]
	training_label = [x for i, x in enumerate(labels_arr) if i != sub]
	# training_feat = np.asarray(training_feat)
	training_feat = np.vstack(training_feat)
	training_label = np.hstack(training_label)

	clf.fit(training_feat, training_label)

	# Test
	testing_feat = features_arr[sub]
	testing_label = labels_arr[sub]

	predicted_class = clf.predict(testing_feat)

	print(predicted_class)
	print(testing_label)

	# for sklearn macro f1 calculation
	for counter in range(len(predicted_class)):
		pred += [predicted_class[counter]]
		y_list += [testing_label[counter]]

	#################### Confusion Matrix ###########################
	ct = confusion_matrix(testing_label, predicted_class)
	order = np.unique(np.concatenate((predicted_class, testing_label)))	
	mat = np.zeros((classes, classes))
	for m in range(len(order)):
		for n in range(len(order)):
			mat[int(order[m]), int(order[n])] = ct[m, n]
		   
	# tot_mat = mat + tot_mat
	##################################################################

	#### adding tot mat to respective db ######
	if sep_flag == True:
		if ((vid_name_arr[sub])[0])[0:3] == 'sub':
			casme_results = mat + casme_results
			casme_pred += [predicted_class[counter]]
			casme_y_list += [testing_label[counter]]
		elif ((vid_name_arr[sub])[0])[0:1] == 's':
			smic_results = mat + smic_results
			smic_pred += [predicted_class[counter]]
			smic_y_list += [testing_label[counter]]			
		else:
			samm_results = mat + samm_results
			samm_pred += [predicted_class[counter]]
			samm_y_list += [testing_label[counter]]			
	else:
		tot_mat = mat + tot_mat
	###########################################

	[f1, precision, recall] = fpr(tot_mat, classes)

	total_samples += len(testing_label)
	war = weighted_average_recall(tot_mat, classes, total_samples)
	uar = unweighted_average_recall(tot_mat, classes)
	macro_f1, weighted_f1 = sklearn_macro_f1(y_list, pred)

	print(str(sub) + ' Fold Processed!')
	# print(casme_results)
	# print(smic_results)
	# print(samm_results)
	

[f1, precision, recall] = fpr(casme_results, classes)
war = weighted_average_recall(casme_results, classes, casme_samples)
uar = unweighted_average_recall(casme_results, classes)
macro_f1, weighted_f1 = sklearn_macro_f1(casme_y_list, casme_pred)

print("CASME II")
print(casme_results)
print("F1: " + str(f1))
print("war: " + str(war))
print("uar: " + str(uar))
print("Macro_f1: " + str(macro_f1))
print("Weighted_f1: " + str(weighted_f1))

# [f1, precision, recall] = fpr(smic_results, classes)
# war = weighted_average_recall(smic_results, classes, smic_samples)
# uar = unweighted_average_recall(smic_results, classes)
# macro_f1, weighted_f1 = sklearn_macro_f1(smic_y_list, smic_pred)

# print("SMIC")
# print(smic_results)
# print("F1: " + str(f1))
# print("war: " + str(war))
# print("uar: " + str(uar))
# print("Macro_f1: " + str(macro_f1))
# print("Weighted_f1: " + str(weighted_f1))

# [f1, precision, recall] = fpr(samm_results, classes)
# war = weighted_average_recall(samm_results, classes, samm_samples)
# uar = unweighted_average_recall(samm_results, classes)
# macro_f1, weighted_f1 = sklearn_macro_f1(samm_y_list, samm_pred)

# print("SAMM")
# print(samm_results)
# print("F1: " + str(f1))
# print("war: " + str(war))
# print("uar: " + str(uar))
# print("Macro_f1: " + str(macro_f1))
# print("Weighted_f1: " + str(weighted_f1))



# print("ALL")
# print(tot_mat)
# print("F1: " + str(f1))
# print("war: " + str(war))
# print("uar: " + str(uar))
# print("Macro_f1: " + str(macro_f1))
# print("Weighted_f1: " + str(weighted_f1))