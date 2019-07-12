import torch
import torch.nn as nn

from torch.utils.data.dataset import Dataset
from torchvision import transforms

import os.path
import numpy as np

from load_data import IconDataset
from architectures import *

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

# path to trained model
model_path = os.path.join(cur_dir, '../models/dcgan32/vae_dcgan32-099.pth')

# check if features exist
assert os.path.isfile(model_path), 'Train dcgan32 first!'

########################################################################################################################
# DATA

# define transformations
trafo_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,), (.5,))])
trafo_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,), (.5,))])

# data sets
train_set = IconDataset(transform=trafo_train, sub_set=sub)
test_set = IconDataset(part='test', transform=trafo_test, sub_set=sub)

# data loader
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False)

########################################################################################################################
# MODEL

architecture = (EncoderDCGAN32, DecoderDCGAN32)

kwargs = {'encoder': architecture[0], 'decoder': architecture[1], 'n_channels': 3, 'latent_dim': 32,
          'n_filters': 64, 'img_h': 32, 'img_w': 32, 'batch_normalization': True,
          'encoder_activation': nn.LeakyReLU(), 'decoder_activation': nn.ReLU()}
model = VAE(**kwargs)

checkpoint = torch.load('../models/dcgan32/vae_dcgan32-099.pth')
model.load_state_dict(checkpoint['model_state_dict'])

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
        _, features, _ = model(x)
        features = np.squeeze(features.detach().cpu().numpy())
        icon_features.append(features)

    return icon_features

########################################################################################################################
# GET FEATURES


# check if numpy array of already encoded icons exists and save; otherwise encode
features_path = os.path.join(cur_dir, '../../data/features/dcgan32.npy')

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
