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

parser.add_argument('--outpath', required=True, type=str, help='path to store trained models to.')
parser.add_argument('--architecture', required=True, type=str, help='resnet32 | dcgan32 | baseline')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nf', type=int, default=100, help='number of filters for DCGAN and RESNET architectures')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--grayscale', type=bool, default=False, help='Use grayscale images? default=False')
parser.add_argument('--nimages', type=int, default=50000, help='Number of images to use for training. default=50 000')
parser.add_argument('--layerSizes', help='Size of encoder and decoder layers. Only if using baseline architecture',
                    default={'encoder': [512, 256], 'decoder': [256, 512]})

opt = parser.parse_args()
print(opt)

img_size = opt.imageSize

architecture_dict = {'resnet32': (EncoderRESNET32, DecoderRESNET32),
                     'dcgan32': (EncoderDCGAN32, DecoderDCGAN32),
                     'baseline': (EncoderBaseline, DecoderBaseline)}
nc = 3 if not opt.grayscale else 1
assert opt.architecture in architecture_dict.keys(), \
    "\nEXIT. Please specify one of the given architectures. See options using 'python train.py -h'"
architecture = architecture_dict[opt.architecture]

if opt.architecture == 'baseline':
    assert isinstance(opt.layerSizes, dict), print('layer sizes have to be in a dictionary.')
    layer_sizes = {'encoder': [nc*img_size**2, *opt.layerSizes['encoder']],
                   'decoder': [*opt.layerSizes['decoder'], nc*img_size**2]}
    print(layer_sizes)


##############
# GPU or CPU #
##############
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print('Using ', device)


##################
# LOSS FUNCTIONS #
##################
def elbo(recon_x, x, mu, log_var):
    """
    Arguments:
    :param recon_x: reconstruced input
    :param x: input,
    :param mu: means of posterior (distribution of z given x)
    :param log_var: log of variances of posterior (distribution of z given x)
    """
    x = x.view(-1, img_size*img_size)
    recon_x = recon_x.view(-1, img_size*img_size)
    sigma_g = 1.  # denominator of squared error loss
    beta = 2.  # factor to balance loss terms
    neg_elbo = 0.5 * (beta * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1) +
                      torch.sum((x - recon_x).pow(2) / sigma_g ** 2.))

    return neg_elbo


def l2_loss(recon_x, x, mu, log_var):
    """
    :param recon_x: reconstruced input
    :param x: input
    :param mu: redundant. Only to call both loss fcts w/ same arguments
    :param log_var: redundant. Only to call both loss fcts w/ same arguments
    """
    x = x.view(-1, img_size*img_size)
    recon_x = recon_x.view(-1, img_size*img_size)

    return .5 * torch.sum((x - recon_x).pow(2))


#####################
# TRAINING FUNCTION #
#####################
def train(model, epochs, path, optimizer, trainloader, testloader, test_after_epoch=False, loss_function=elbo):
    """
    :param model: model that will be trained; object
    :param epochs: number of epochs to train model; int
    :param path: path to store and load trained models; str
    :param optimizer: optimizer that is used for training
    :param trainloader: dataloader for training
    :param testloader: dataloader for testing
    :param test_after_epoch: evaluate on test set after epoch? default=False
    :param loss_function: loss to use in training. default=elbo
    """

    # check for previous trained models and resume from there if available
    try:
        previous = max(glob.glob(path + '*.pth'))
        print(f'\nload previous model: {previous}')
        checkpoint = torch.load(previous)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epochs_trained = checkpoint['epoch']
    except Exception as e:
        print('\nno model to load. Reason: ', e)
        epochs_trained = 0

    model.train()

    for epoch in np.arange(epochs_trained, epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(tqdm(trainloader, desc=f'Train Epoch {epoch}', leave=False)):
            x = data
            x = x.to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var = model(x)

            loss = loss_function(recon_batch, x, mu, log_var)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % int(len(train_loader) / 10) == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item() / len(x)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

        # save model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path + '{}.pth'.format(f'00{epoch}'[-3:]))

        # test model
        if test_after_epoch:
            test(model, testloader)


def test(model, loader):
    """
    :param model: model to test
    :param loader: loader for test data
    """

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += elbo(recon_batch, data, mu, logvar).item()

    test_loss /= len(loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


######################
# VISUALIZE RESULTS  #
######################
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
    plt.show()


#############
#    RUN    #
#############
if __name__ == "__main__":

    ##############
    # LOAD DATA  #
    ##############
    print('\nloading data')
    # define transformations
    if opt.grayscale:
        trafo_train = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1),
                                          transforms.ToTensor(), transforms.Normalize((.5,), (.5,))])
        trafo_test = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1),
                                         transforms.ToTensor(), transforms.Normalize((.5,), (.5,))])
    else:
        trafo_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,), (.5,))])
        trafo_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,), (.5,))])
    # datasets
    train_set = IconDataset(transform=trafo_train)
    test_set = IconDataset(part='test', transform=trafo_test)
    print('train dataset', len(train_set))
    print('test dataset', len(test_set))
    if opt.nimages <= len(train_set):
        # subsets
        train_subset = torch.utils.data.Subset(train_set, np.arange(opt.nimages, dtype=int).tolist())
        test_subset = torch.utils.data.Subset(test_set, np.arange(opt.nimages / 10, dtype=int).tolist())
        print('train subset', len(train_subset))
        # dataloader
        train_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=opt.batchSize, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_subset, batch_size=opt.batchSize, shuffle=False)
    else:
        print(f'number of training images {int(opt.nimages)} larger than training set. '
              f'Using all {len(train_set)} training images.')
        # dataloader
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=opt.batchSize, shuffle=False)


    #############
    # TRAIN VAE #
    #############
    # if using baseline architecture
    if opt.architecture == 'baseline':
        kwargs = {'encoder_layer_sizes': layer_sizes['encoder'], 'decoder_layer_sizes': layer_sizes['decoder'],
                  'latent_dim': opt.nz, 'n_channels': nc}
        vae = BaselineVAE(**kwargs)
    # if using dcgan32 or resnet32
    else:
        kwargs = {'encoder': architecture[0], 'decoder': architecture[1], 'n_channels': nc, 'latent_dim': opt.nz,
                  'n_filters': opt.nf, 'img_h': opt.imageSize, 'img_w': opt.imageSize,'batch_normalization': True,
                  'encoder_activation': nn.LeakyReLU(), 'decoder_activation': nn.ReLU()}
        vae = VAE(**kwargs)
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=opt.lr)

    if opt.outpath[-1] == '/':
        filepath = opt.outpath+'vae_'+opt.architecture+'-'
    else:
        filepath = opt.outpath + '/vae_' + opt.architecture + '-'
    train(vae, opt.niter, filepath, optimizer, train_loader, test_loader)
