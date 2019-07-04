import torch
import torch.nn as nn

from torch.utils.data.dataset import Dataset
from torchvision import transforms, models

import os.path
import numpy as np

from load_data import IconDataset

from tqdm import tqdm

########################################################################################################################
# CONFIG

# batch size
bs = 128
# subset for debugging
sub = False
# GPU or CPU
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# determine directory of this script
cur_dir = os.path.dirname(os.path.abspath(__file__))

########################################################################################################################
# DATA

# mean and std of dataset; only for complete data
mean_icon = [183.09622661656687, 174.6211668631342, 171.76230690173531]
std_icon = [89.13559012807926, 85.83242306938422, 90.71514644175413]

# define transformations
normalize = transforms.Normalize(mean=mean_icon, std=std_icon)
trafo_train = transforms.Compose([transforms.ToTensor(), normalize])
trafo_test = transforms.Compose([transforms.ToTensor(), normalize])

# data sets
train_set = IconDataset(transform=trafo_train, sub_set=sub)
test_set = IconDataset(part='test', transform=trafo_test, sub_set=sub)

# data loader
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False)

########################################################################################################################
# MODEL

# load pre-trained ResNet from PyTorch
resnet = models.resnet18(pretrained=True)

# get some sample images
samples = next(iter(train_loader))

# drop last layer to get images represented by the last layer
modified_resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))

########################################################################################################################
# HELPER FUNCTION


def get_features(part='train'):
    """
    :param part: which part to get features of (train or test); string
    :return icon_features: ResNet features of icon dataset; list[np.array]
    """

    if part == 'train':
        loader = train_loader
    elif part == 'test':
        loader = test_loader
    else:
        print('Please provide part')
        return

    # get features
    icon_features = []
    for batch_idx, data in enumerate(tqdm(loader, desc='Encode {} set'.format(part), leave=False)):
        x = data
        x = x.to(device)
        features = np.squeeze(modified_resnet(x).detach().cpu().numpy())
        icon_features.append(features)

    return icon_features

########################################################################################################################
# GET FEATURES


# check if numpy array of already encoded icons exists and save; otherwise encode
features_path = os.path.join(cur_dir, '../../data/features/resnet.npy')

if not os.path.isfile(features_path):
    # get features
    icon_features_train = get_features(part='train')
    icon_features_test = get_features(part='test')
    icon_features = icon_features_test + icon_features_train
    icon_features = np.concatenate(icon_features)
    # save features
    np.save(features_path, icon_features)
else:
    print('encoded data already exists')
