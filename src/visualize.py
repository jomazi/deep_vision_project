import torch
import torch.nn as nn

from torch.utils.data.dataset import Dataset
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

from fcts.load_data import load_data, IconDataset
from fcts.architectures import *

########################################################################################################################
# CONFIG

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print('Using ', device)

########################################################################################################################
# DATA

# trafo
trafo_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,), (.5,))])
# datasets
test_set = IconDataset(part='test', transform=trafo_test)
print('test dataset', len(test_set))
# subsets
test_subset = torch.utils.data.Subset(test_set, np.arange(10000 / 10, dtype=int).tolist())
# dataloader
test_loader = torch.utils.data.DataLoader(dataset=test_subset, batch_size=128, shuffle=False)

########################################################################################################################
# MODEL

architecture = (EncoderDCGAN32, DecoderDCGAN32)

kwargs = {'encoder': architecture[0], 'decoder': architecture[1], 'n_channels': 3, 'latent_dim': 32,
          'n_filters': 64, 'img_h': 32, 'img_w': 32, 'batch_normalization': True,
          'encoder_activation': nn.LeakyReLU(), 'decoder_activation': nn.ReLU()}
model = VAE(**kwargs)

checkpoint = torch.load('./models/dcgan32/vae_dcgan32-099.pth')
model.load_state_dict(checkpoint['model_state_dict'])

########################################################################################################################
# PLOT


def imshow(img, color=False):
    npimg = img.cpu().numpy()
    npimg -= npimg.min()
    npimg /= npimg.max()
    if color:
        npimg = np.swapaxes(np.swapaxes(npimg, 0, 1), 1, 2)
        plt.imshow(npimg)
    else:
        print(npimg.shape)
        plt.imshow(npimg[0], cmap='gray')
    plt.xticks([])
    plt.yticks([])


def show_examples(model, _train_loader, color=True):
    x = next(iter(_train_loader))
    samples = x.to(device)
    model.eval()
    samples_rec,   _, _ = model(samples)
    samples_rec = samples_rec.detach()
    plt.figure(figsize=(12,8))
    for i in range(0, 3):
        plt.subplot(3,2,2*i+1)
        plt.tight_layout()
        imshow(samples[i], color=color)
        plt.title("Ori. {}".format(i))

        plt.subplot(3, 2, 2*i+2)
        plt.tight_layout()
        imshow(samples_rec[i], color=color)
        plt.title("Rec. {}".format(i))
    plt.savefig('./plots/generated_icons/dcgan32.png')


show_examples(model, test_loader)