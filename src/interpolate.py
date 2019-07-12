import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data.dataset import Dataset
from torchvision import transforms

import argparse
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt

from fcts.load_data import load_data, IconDataset
from fcts.architectures import *

from tqdm import tqdm


##############
#   PARSER   #
##############
parser = argparse.ArgumentParser()

parser.add_argument('--path', required=True, type=str, help='path to load model from.')
parser.add_argument('--architecture', required=True, type=str, help='resnet32 | dcgan32 | baseline')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size, default=128')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network, default=32')
parser.add_argument('--nf', type=int, default=64, help='number of filters for DCGAN and RESNET architectures, default=64')
parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector, default=32')
parser.add_argument('--grayscale', type=bool, default=False, help='Use grayscale images? default=False')
parser.add_argument('--nimages', type=int, default=50000, help='Number of images to use for training. default=50 000')

opt = parser.parse_args()
print(opt)

img_size = opt.imageSize

nc = 3 if not opt.grayscale else 1

architecture_dict = {'resnet32': (EncoderRESNET32, DecoderRESNET32),
                     'dcgan32': (EncoderDCGAN32, DecoderDCGAN32),
                     'baseline': (EncoderBaseline, DecoderBaseline)}
assert opt.architecture in architecture_dict.keys(), \
    "\nEXIT. Please specify one of the given architectures. See options using 'python train.py -h'"
architecture = architecture_dict[opt.architecture]

##############
# GPU or CPU #
##############
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = 'cpu'
print('Using ', device)


#############
# LOAD DATA #
#############
trafo_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,), (.5,))])
trafo_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,), (.5,))])
# datasets
train_set = IconDataset(transform=trafo_train)
test_set = IconDataset(part='test', transform=trafo_test)
print('train dataset', len(train_set))
print('test dataset', len(test_set))


######################
# VISUALIZE RESULTS  #
######################
def imshow(img, color=True, save=False, _filepath=None):
    npimg = img.cpu().detach().numpy()
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
    if save and _filepath is not None:
        plt.savefig(_filepath)


def show_examples(_model, _train_loader, color=True, save=False, path=None):
    x = next(iter(_train_loader))
    samples = x.to(device)
    model.eval()
    samples_rec,  _, _ = _model(samples)
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
    if save and path is not None:
        plt.savefig(path)
    plt.show()


def interpolate_2(_model, _dataset, imgs=[1, 2], steps=20, subplot_rows=3):
    # get images
    x1 = _dataset[imgs[0]][None, :]
    x2 = _dataset[imgs[1]][None, :]
    # run model
    recon_x1, mean1, log_var1 = _model(x1)
    recon_x2, mean2, log_var2 = _model(x2)
    # save images
    imshow(x1[0], save=True, _filepath='src/images/x1.png')
    imshow(recon_x1[0], save=True, _filepath='src/images/recon_x1.png')
    imshow(x2[0], save=True, _filepath='src/images/x2.png')
    imshow(recon_x2[0], save=True, _filepath='src/images/recon_x2.png')
    # interpolate
    delta = (mean2-mean1)/float(steps)
    for i in range(0, steps+1):
        img = model.decoder(mean1+i*delta)
        img_list.append(img)
    # figure
    plt.figure()
    for i, img in enumerate(img_list):
        # correct img range
        npimg = img.cpu().detach().numpy()[0]
        npimg -= npimg.min()
        npimg /= npimg.max()
        npimg = np.swapaxes(np.swapaxes(npimg, 0, 1), 1, 2)
        # subplot
        plt.subplot(subplot_rows, np.ceil(len(img_list)/float(subplot_rows)), i+1)
        plt.imshow(npimg)
        plt.xticks([])
        plt.yticks([])
    plt.savefig('src/images/interpolation_01.png')

def interpolate_3(_model, _dataset, imgs=[0, 1, 2], steps=10, _to_path=None):
    # get images
    xs = []
    means = []
    for idx in imgs:
        xs.append(_dataset[idx][None, :])
        means.append(_model(xs[-1])[1])
    # interpolate
    delta01 = (output_list[1]-output_list[0])/float(steps)
    delta02 = (output_list[2]-output_list[0])/float(steps)
    for i in range(0, steps+1):
        img = model.decoder(mean1+i*delta)
        img_list.append(img)
    # figure
    plt.figure()
    for i in range(steps+1):
        for j in range(steps+1-i):
            # correct img range
            img = model.decoder(output_list[0] + i * delta02 + j * delta01)
            npimg = img.cpu().detach().numpy()[0]
            npimg -= npimg.min()
            npimg /= npimg.max()
            npimg = np.swapaxes(np.swapaxes(npimg, 0, 1), 1, 2)
            # subplot
            plt.subplot(steps, steps, i * steps + j + 1)
            plt.imshow(npimg)
            plt.xticks([])
            plt.yticks([])
    if _to_path is not None:
        plt.savefig(_to_path)

##############
# LOAD MODEL #
##############
kwargs = {'encoder': architecture[0], 'decoder': architecture[1], 'n_channels': nc, 'latent_dim': opt.nz,
          'n_filters': opt.nf, 'img_h': opt.imageSize, 'img_w': opt.imageSize,'batch_normalization': True,
          'encoder_activation': nn.LeakyReLU(), 'decoder_activation': nn.ReLU()}
model = VAE(**kwargs)
model.eval()
# check for previous trained models
try:
    previous = max(glob.glob(opt.path + '*.pth'))
    print(f'\nload previous model: {previous}')
    checkpoint = torch.load(previous)
    model.load_state_dict(checkpoint['model_state_dict'])
    epochs_trained = checkpoint['epoch']
except Exception as e:
    print('\nno model to load. Reason: ', e)
    epochs_trained = 0

#interpolate_2(model, test_set, imgs=[1, 200])
interpolate_3(model, test_set, _to_path='src/images/interpolate_3/')
#test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=opt.batchSize, shuffle=False)
#show_examples(model, test_loader, save=True, path='src/images/examples.png')
