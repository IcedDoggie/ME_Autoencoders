import h5py
import numpy as np

weights = np.load('bvlc_alexnet.npy', encoding='ASCII')

with h5py.File('bvlc_alexnet_keras.h5', 'w') as h5:
	h5.create_dataset('weights_data', data=weights)

print('Done')
