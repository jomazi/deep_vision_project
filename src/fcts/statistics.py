import numpy as np

from load_data import load_data

# get std and mean of data
data_train = load_data(part='train')
data_test = load_data(part='test')

data = np.vstack((data_train, data_test))

data_flattend_channelwise = data.reshape(-1, 3)

std = list(np.std(data_flattend_channelwise, axis=0))
mean = list(np.mean(data_flattend_channelwise, axis=0))

print("std: ", std)
print("mean: ", mean)