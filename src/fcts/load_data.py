import numpy as np
import os
from glob import glob
import _pickle


def load_data(dir='../../data/'):
    '''
    :param dir: relative path to directory the data is stored
    :return: data as numpy array
    '''

    # paths to data
    current_dir = os.path.dirname(os.path.realpath(__file__))
    loaded_dir = os.path.join(current_dir, dir)
    data_dir = os.path.join(loaded_dir, 'LLD-icon')

    # check if data already downloaded
    if not os.path.isdir(data_dir):
        print('please download data first')
        return

    # load Pickle files as numpy array
    else:
        # get all pickle files in data directory
        files = glob(os.path.join(data_dir, 'LLD-icon_data_*.pkl'))
        files.sort()

        # initialize data
        with open(files[0], 'rb') as f:
            data = _pickle.load(f, encoding='latin1')

        # iterate over other files and concat
        for file in files[1:]:
            with open(file, 'rb') as f:
                data_add = _pickle.load(f, encoding='latin1')
            data = np.concatenate((data, data_add))

        return data
