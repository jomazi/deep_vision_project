import torch
import torch.nn as nn

from torch.utils.data.dataset import Dataset
from torchvision import transforms

import glob

import numpy as np
import matplotlib.pyplot as plt

from fcts.load_data import IconDataset
from fcts.architectures import *


##############
# GPU or CPU #
##############
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = 'cpu'
print('Using ', device)


############################
# VISUALIZATION FUNCTIONS  #
############################
def imshow(img, color=True, save=False, _filepath=None):
    # correct img range to [0, 1]
    npimg = img.cpu().detach().numpy()
    npimg -= npimg.min()
    npimg /= npimg.max()
    # plot
    if color:
        npimg = np.swapaxes(np.swapaxes(npimg, 0, 1), 1, 2)
        plt.imshow(npimg)
    else:
        print(npimg.shape)
        plt.imshow(npimg[0], cmap='gray')
    # hide ticks
    plt.xticks([])
    plt.yticks([])
    # save
    if save and _filepath is not None:
        plt.savefig(_filepath)

def make_subplot(_img, _row, _column, _item):
    # correct img range to [0, 1]
    npimg = _img.cpu().detach().numpy()[0]
    npimg -= npimg.min()
    npimg /= npimg.max()
    npimg = np.swapaxes(np.swapaxes(npimg, 0, 1), 1, 2)
    # subplot
    plt.subplot(_row, _column, _item)
    plt.imshow(npimg)
    plt.xticks([])
    plt.yticks([])


def compare_models(_model_dict, _train_loader, color=True, _to_path=None):
    # get samples
    x = next(iter(_train_loader))
    samples = x.to(device)
    # figure
    plt.figure()
    # plot originals
    for i in range(0, 3):
        plt.subplot(3,4,4*i+1)
        plt.tight_layout()
        imshow(samples[i], color=color)
        if i == 0:
            plt.title('original')
    # reconstruct and plot for different models
    for j, key in enumerate(_model_dict):
        # get model and reconstruct sample
        model = _model_dict[key]
        model.eval()
        samples_rec,  _, _ = model(samples)
        samples_rec = samples_rec.detach()
        # plot
        for i in range(3):
            plt.subplot(3, 4, 4*i+2+j)
            plt.tight_layout()
            imshow(samples_rec[i], color=color)
            if i==0:
                plt.title('{}'.format(key))
    # save
    if _to_path is not None:
        plt.savefig(_to_path)
    plt.show()


def interpolate_2(_model, _dataset, imgs=[1, 2], steps=20, subplot_rows=3, _to_path=None):
    # get latent space coordinates
    means = []
    for idx in imgs:
        means.append(_model(_dataset[idx][None, :])[1])
    # interpolate
    delta = (means[1]-means[0])/float(steps)
    # figure
    plt.figure()
    for i in range(steps+1):
        # subplot
        img = model.decoder(means[0] + i * delta)
        make_subplot(img, subplot_rows, np.ceil((steps+1)/subplot_rows), i+1)
    # save
    if _to_path is not None:
        plt.savefig(_to_path)


def interpolate_3(_model, _dataset, imgs=[0, 1, 2], steps=10, _to_path=None):
    # get latent space coordinates
    means = []
    for idx in imgs:
        means.append(_model(_dataset[idx][None, :])[1])
    # interpolate
    delta01 = (means[1]-means[0])/float(steps)
    delta02 = (means[2]-means[0])/float(steps)
    # figure
    plt.figure()
    for i in range(steps):
        for j in range(steps-i):
            # construct image from latent space vector
            img = model.decoder(means[0] + i * delta02 + j * delta01)
            # subplot
            make_subplot(img, steps, steps, i * steps + j + 1)
    # save
    if _to_path is not None:
        plt.savefig(_to_path)

if __name__ == '__main__':
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


    ###############
    # LOAD MODELS #
    ###############
    model_dict = {}
    # Baseline
    kwargs_baseline = {'encoder_layer_sizes': [32*32*3, 512, 256], 'decoder_layer_sizes': [256, 512, 32*32*3],
                       'latent_dim': 32, 'n_channels': 3}
    model_dict['baseline'] = BaselineVAE(**kwargs_baseline)

    # DCGAN
    kwargs_dcgan = {'encoder': EncoderDCGAN32, 'decoder': DecoderDCGAN32, 'n_channels': 3, 'latent_dim': 32,
                    'n_filters': 64, 'img_h': 32, 'img_w': 32,'batch_normalization': True,
                    'encoder_activation': nn.LeakyReLU(), 'decoder_activation': nn.ReLU()}
    model_dict['dcgan32'] = VAE(**kwargs_dcgan)

    # RESNET
    kwargs_resnet = {'encoder': EncoderRESNET32, 'decoder': DecoderRESNET32, 'n_channels': 3, 'latent_dim': 32,
                     'n_filters': 64, 'img_h': 32, 'img_w': 32,'batch_normalization': True,
                     'encoder_activation': nn.LeakyReLU(), 'decoder_activation': nn.ReLU()}
    model_dict['resnet32'] = VAE(**kwargs_resnet)

    def load_model(_path, _model):
        # check for previous trained models
        try:
            previous = max(glob.glob(_path + '*.pth'))
            print(f'\nload previous model: {previous}')
            checkpoint = torch.load(previous)
            _model.load_state_dict(checkpoint['model_state_dict'])
            epochs_trained = checkpoint['epoch']
        except Exception as e:
            print('\nno model to load. Reason: ', e)
            epochs_trained = 0

    for key in model_dict:
        model = model_dict[key]
        load_model('src/models/{}/'.format(key), model)
        model.to(device)
        model.eval()
        model_dict[key] = model


    #############
    # VISUALIZE #
    #############
    compare_models(model_dict,
                   torch.utils.data.DataLoader(dataset=test_set, batch_size=3, shuffle=True),
                   _to_path='src/images/model_comparison.png')
    interpolate_2(model_dict['dcgan32'], test_set, imgs=[0, 1], _to_path='src/images/interpolate_2.png')
    interpolate_3(model_dict['dcgan32'], test_set, imgs=[1, 2, 3], _to_path='src/images/interpolate_3.png')
