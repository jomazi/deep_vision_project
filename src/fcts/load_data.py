import numpy as np
import os
from glob import glob
import _pickle


def load_data(dir='../../data/', split=0.7, part='train', batchsize=128, subset=False):
    """
    :param dir: relative path to directory the data is stored
    :param split: partial of training data to split data into train and test set; float
    :param part: which part of data to retutn either train or test set; string
    :param batchsize: length of data has to be multiple of batchsize; int
    :param subset: only use subset of data; bool
    :return: data as numpy array, train and test set
    """

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
            
        # shuffle
        np.random.shuffle(data)

        # use subset if requested
        if subset:
            data = data[:(batchsize*100)]
        
        # split data in train and test
        index_split = int(len(data) * split)
        
        # return requested part
        if part == 'train':
            # make last batch have same batch size
            mod = index_split % batchsize
            index_split = index_split - mod
            return data[:index_split]
        else:
            # make last batch have same batch size
            mod = (len(data)-index_split) % batchsize
            index_split = index_split + mod
            return data[index_split:]
