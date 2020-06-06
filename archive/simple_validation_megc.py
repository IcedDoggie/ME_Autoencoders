import pandas as pd
import numpy as np
import os

filename = 'combined_3dbs.csv'
table = pd.read_table(filename, sep=',', header=None)
table = table.as_matrix()

data_path = '/media/ice/OS/Datasets/Combined_Dataset_Apex_Flow/'

casme_path = data_path + 'CASME2_Optical/CASME2_Optical/'
samm_path = data_path + 'SAMM_Optical/SAMM_Optical/'
smic_path = data_path + 'SMIC_Optical/SMIC_Optical/'

data_arr = []

for root, folders, files in os.walk(casme_path):
	if len(root) > 85:
		idx = root.split('/', 8)[-1]
		data_arr += [idx]
		# print(root)

for root, folders, files in os.walk(samm_path):
	if len(root) > 79:
		idx = root.split('/', 8)[-1]
		data_arr += [idx]
# 		print(root)

for root, folders, files in os.walk(smic_path):
	if len(root) > 79:
		idx = root.split('/', 8)[-1]
		data_arr += [idx]
# 		print(root)

combined_arr = []
for counter in range(len(table)):
	entry = table[counter, :]
	vid = entry[2].replace(' ', '')
	entry = entry[1] + '/' + vid
	combined_arr += [entry]


# for item in combined_arr:
# 	if item not in data_arr:
# 		print(item)






