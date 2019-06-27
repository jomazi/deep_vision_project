#!/usr/bin/env python
# coding: utf-8

# ## Import

# In[17]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data.dataset import Dataset
from torchvision import transforms, models
from torch.autograd import Variable

import os.path
import glob
import numpy as np
import matplotlib.pyplot as plt

from fcts.load_data import load_data

from tqdm import tqdm_notebook as tqdm

from sklearn.decomposition import PCA
from sklearn import cluster


# ## Configuration

# In[18]:


# batch size
bs = 128
# GPU or CPU
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


# ## Data

# In[19]:


class IconDataset(Dataset):
    def __init__(self, part='train', transform=None):
        """
        :param part: get either train or test set; string
        :param transform = transformations, that should be applied on dataset
        """
        self.data = load_data(part=part)
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        # transform image 
        if self.transform is not None:
            img_transformed = self.transform(img)
        # return transformed image
        return img_transformed

    def __len__(self):
        return self.data.shape[0]

# get std and mean of data for normalization
data_train = load_data(part='train')
data_test = load_data(part='test')
data = np.vstack((data_train, data_test))
data_flattend_channelwise = data.reshape(-1, 3)
std = list(np.std(data_flattend_channelwise, axis=0))
mean = list(np.mean(data_flattend_channelwise, axis=0))
print("std: ", std)
print("mean: ", mean)
# In[20]:


# store mean and std of dataset
mean_icon = [183.09622661656687, 174.6211668631342, 171.76230690173531]
std_icon = [89.13559012807926, 85.83242306938422, 90.71514644175413]


# In[21]:


# define transformations
normalize = transforms.Normalize(mean=mean_icon, std=std_icon)
trafo_train = transforms.Compose([transforms.ToTensor(), normalize])
trafo_test = transforms.Compose([transforms.ToTensor(), normalize])
# datasets
train_set = IconDataset(transform=trafo_train)
test_set = IconDataset(part='test', transform=trafo_test)
# dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False)


# ##### Gray version of dataset and cropped to 28 x 28

# In[22]:


# define transformations
trafo_train_gray = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1), transforms.CenterCrop(size=28), transforms.ToTensor()])
trafo_test_gray = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1), transforms.CenterCrop(size=28), transforms.ToTensor()])
# datasets
train_set_gray = IconDataset(transform=trafo_train_gray)
test_set_gray = IconDataset(part='test', transform=trafo_test_gray)
# dataloader
train_loader_gray = torch.utils.data.DataLoader(dataset=train_set_gray, batch_size=bs, shuffle=True)
test_loader_gray = torch.utils.data.DataLoader(dataset=test_set_gray, batch_size=bs, shuffle=False)


# ## Variational Autoencoder

# ### Baseline Architecture

# #### Encoder

# In[23]:


class Encoder_baseline(nn.Module):

    def __init__(self, layer_sizes, latent_dim, n_channels):
        super(Encoder_baseline, self).__init__()
        """
        :param layer_sizes: list of sizes of layers of the encoder; list[int]
        :param latent_dim: dimension of latent space, i.e. dimension out output of the encoder; int
        :param n_channels: number of channels of images; int
        """
        # store values
        self.latent_dim = latent_dim
        self.channels = n_channels
        
        # initialize layers
        layer_list = []
        for in_dim, out_dim in zip(layer_sizes, layer_sizes[1:]):
            layer_list.append(nn.Linear(in_dim, out_dim))
            layer_list.append(nn.ReLU())
        
        # store layers
        self.layers = nn.Sequential(*layer_list)
        
        # layers for latent space output
        self.out_mean = nn.Linear(layer_sizes[-1], latent_dim)
        self.out_var = nn.Linear(layer_sizes[-1], latent_dim)
    
    def forward(self, x):  
        """ 
        :param x: tensor 
        :return means: tensor of dimension 
        :return log_var: tensor of dimension 
        """
        
        # flatten x
        #print(x.size())
        #x = x.view(-1, self.channels, x.size(-1)**2)
        
        # forward 
        out = self.layers(x)
        
        # latent space output
        means = self.out_mean(out)
        log_vars = self.out_var(out)
     
        return means, log_vars
    


# #### Decoder

# In[24]:


class Decoder_baseline(nn.Module):

    def __init__(self, layer_sizes, latent_dim, n_channels):     
        super(Decoder_baseline, self).__init__()
        """
        :param layer_sizes: list of sizes of layers of the decoder; list[int]
        :param latent_dim: dimension of latent space, i.e. dimension of input of the decoder; int
        :param n_channels: number of channels of images; int
        """

        self.latent_dim = latent_dim
        self.channels = n_channels
        self.layer_sizes = layer_sizes
        
        layer_list = [nn.Linear(latent_dim, layer_sizes[0])]
        
        # initialize layers
        for in_dim, out_dim in zip(layer_sizes, layer_sizes[1:]):
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Linear(in_dim, out_dim))

        # store layers
        self.layers = nn.Sequential(*layer_list)
            
    def forward(self, z):
        """
        :param z: tensor
        :return x: mu of gaussian distribution (reconstructed image from latent code z)
        """

        # forward
        h = int(np.sqrt(self.layer_sizes[-1]))
        x = torch.sigmoid(self.layers(z)).view(-1, self.channels, h, h)

        return x


# #### Model

# In[25]:


class VAE_baseline(nn.Module):

    def __init__(self, inp_dim, encoder_layer_sizes, decoder_layer_sizes, latent_dim, n_channels):
        """
        :param inp_dim: dimension of input; int
        :param encoder_layer_sizes: sizes of the encoder layers; list
        :param decoder_layer_sizes: sizes of the decoder layers; list
        :param latent_dim: dimension of latent space/bottleneck; int
        :param n_channels: number of channels of images; int
        """
        
        super(VAE_baseline, self).__init__()
        
        self.latent_dim = latent_dim
        self.channels = n_channels
        
        self.encoder = Encoder_baseline(encoder_layer_sizes, latent_dim, n_channels)
        self.decoder = Decoder_baseline(decoder_layer_sizes, latent_dim, n_channels)
        
        
    def forward(self, x):
        """ 
        :param x: tensor of dimension 
        :return recon_x: reconstructed x
        :return means: output of encoder
        :return log_var: output of encoder (logarithm of variance)
        """
        batch_size = x.size(0)

        x = x.view(-1, self.channels, x.size(-1)**2)
        means, log_vars = self.encoder(x)
        std = torch.exp(.5*log_vars)
        eps = torch.randn_like(std)
        z = means + eps*std
        recon_x = self.decoder(z)

        return recon_x, means, log_vars
        
    def sampling(self, n=2):
        """
        :param n: amount of samples (amount of elements in the latent space); int
        :return x_sampled: n randomly sampled elements of the output distribution
        """
        
        # draw samples p(z)~N(0,1)
        z = torch.randn((n, self.latent_dim))
        # generate
        x_sampled = self.decoder(z)

        return x_sampled


# ### Advanced Architecture

# In[26]:


class VAE_advanced(nn.Module):
    def __init__(self, n_channels, height):
        """
        :param n_channels: number of channels of images; int
        :param height: height of images, expected height = width; int
        """
        super(VAE_advanced, self).__init__()
        
        # store values
        self.channels = n_channels
        self.height = height

        # Encoder
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, height, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(height)
        self.conv3 = nn.Conv2d(height, height, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(height)
        self.conv4 = nn.Conv2d(height, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(8 * 8 * 16, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc21 = nn.Linear(512, 512)
        self.fc22 = nn.Linear(512, 512)

        # Decoder
        self.fc3 = nn.Linear(512, 512)
        self.fc_bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 8 * 8 * 16)
        self.fc_bn4 = nn.BatchNorm1d(8 * 8 * 16)

        self.conv5 = nn.ConvTranspose2d(16, height, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(height)
        self.conv6 = nn.ConvTranspose2d(height, height, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(height)
        self.conv7 = nn.ConvTranspose2d(height, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, n_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()

    def encode(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, 8 * 8 * 16)

        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        return self.fc21(fc1), self.fc22(fc1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 8, 8)

        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        return self.conv8(conv7).view(-1, self.channels, height, height)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ### Loss
# Implement the loss function for the VAE
def vae_loss(recon_x, x, mu, log_var, loss_func):
    """
    :param recon_x: reconstruced input
    :param x: input
    :param mu, log_var: parameters of posterior (distribution of z given x)
    :loss_func: loss function to compare input image and constructed image
    """

    recon_loss = loss_func(recon_x, x)
    kl_loss = torch.mean(0.5 * torch.sum(
        torch.exp(log_var) + mu**2 - 1. - log_var, 1))
    return recon_loss + kl_loss
# In[27]:


# Implement the Loss function for the VAE/CVAE
def elbo(recon_x, x, mu, log_var):
    """
    :param recon_x: reconstruced input
    :param x: input,
    :param mu, log_var: parameters of posterior (distribution of z given x)
    :return neg_ELBO: neagtive ELBO
    """
    
    sigma_g = 1.
    
    neg_ELBO =0.5 * (torch.sum( mu.pow(2)+log_var.exp()-log_var-1 )+
                       torch.sum( (x-recon_x).pow(2)/sigma_g**2. ) )
    
    return neg_ELBO


# ### Test Function

# In[28]:


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
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


# ### Training Function

# In[29]:


# Training of the VAE
def train(model, epochs, path, optimizer, train_loader, test_loader):
    """
    :param model: model that will be trained; object
    :param epochs: number of epochs to train model; int
    :param path: path to store and load trained models; str
    :param optimizer: optimizer that is used for training
    :param train_/test_loader: dataloader for training and testing 
    """
    
    # check for previous trained models and resume from there if available
    try:
        previous = max(glob.glob(path + '/*.pth'))
        print('load previous model')
        checkpoint = torch.load(previous)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epochs_trained = checkpoint['epoch']
    except Exception as e:
        print('no model to load')
        epochs_trained = 0
    
    model.train()
    
    for epoch in np.arange(epochs_trained, epochs): 
 
        train_loss = 0
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f'Train Epoch {epoch}', leave=False)):
            x = data
            x = x.to(device)
            optimizer.zero_grad()

            recon_batch,  mu, log_var = model(x)
            loss = elbo(recon_batch,  x, mu, log_var)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

        # save model
        torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, path+('/vae-{}.pth').format(epoch))
        
        # test model
        test(model, test_loader)


# ## Experiments

# ### VAE Training

# #### Grayscale Images: Baseline Architecture

# In[30]:


# hyperparameters
encoder_layer_sizes_bl_gray = [28*28, 512, 256]
decoder_layer_sizes_bl_gray = [256, 512, 28*28]

latent_dim_baseline_bl_gray = 2 
vae_baseline_gray = VAE_baseline(inp_dim=(28*28), encoder_layer_sizes=encoder_layer_sizes_bl_gray, decoder_layer_sizes=decoder_layer_sizes_bl_gray, 
                                 latent_dim=latent_dim_baseline_bl_gray, n_channels=1)
vae_baseline_gray = vae_baseline_gray.to(device)
optimizer_baseline_gray = optim.Adam(vae_baseline_gray.parameters(), lr=1e-3)

epochs_baseline_gray = 15


# In[31]:


train(vae_baseline_gray, epochs_baseline_gray, './models/clustering/baseline/gray', optimizer_baseline_gray, train_loader_gray, test_loader_gray)


# #### Colored Images: Baseline Architecture

# In[32]:


# hyperparameters
encoder_layer_sizes_bl_color = [32*32, 512, 256]
decoder_layer_sizes_bl_color = [256, 512, 32*32]

latent_dim_baseline_bl_color = 2 
vae_baseline_color = VAE_baseline(inp_dim=(32*32), encoder_layer_sizes=encoder_layer_sizes_bl_color, decoder_layer_sizes=decoder_layer_sizes_bl_color, 
                                 latent_dim=latent_dim_baseline_bl_color, n_channels=3)
vae_baseline_color = vae_baseline_color.to(device)
optimizer_baseline_color = optim.Adam(vae_baseline_color.parameters(), lr=1e-3)

epochs_baseline_color = 15


# In[ ]:


train(vae_baseline_color, epochs_baseline_color, './models/clustering/baseline/color', optimizer_baseline_color, train_loader, test_loader)


# #### Grayscale Images: Advanced Architecture

# In[ ]:


# hyperparameters
latent_dim_advanced_gray = 2 
vae_advanced_gray = VAE_advanced(n_channels=1, height=28)
vae_advanced_gray = vae_advanced_gray.to(device)
optimizer_advanced_gray = torch.optim.SGD(vae_advanced_gray.parameters(), lr=0.001, momentum=0.9)

epochs_advanced_gray = 15


# In[ ]:


train(vae_advanced_gray, epochs_advanced_gray, './models/clustering/advanced/gray', optimizer_advanced_gray, train_loader_gray, test_loader_gray)


# #### Colored Images: Advanced Architecture

# In[ ]:


# hyperparameters
latent_dim_advanced_color = 2 
vae_advanced_color = VAE_advanced(n_channels=3, height=32)
vae_advanced_color = vae_advanced_color.to(device)
optimizer_advanced_color = torch.optim.SGD(vae_advanced_color.parameters(), lr=0.001, momentum=0.9)

epochs_advanced_color = 15


# In[ ]:


train(vae_advanced_color, epochs_advanced_color, './models/clustering/advanced/color', optimizer_advanced_color, train_loader, test_loader)


# ### Visualization

# In[ ]:


#path2model = max(glob.glob('./models/clustering/advanced' + '/*.pth'))
best_model = vae_baseline_gray
#best_model_dict = torch.load(path2model)
#best_model.load_state_dict(best_model_dict['model_state_dict'])


# In[ ]:


def imshow(inp):
    """Imshow for Tensor."""
    # unnomralize
    #inp = inp.mul(torch.FloatTensor(std_icon))#.add(torch.FloatTensor(mean)
    print(inp.shape)
    #inp = inp.numpy().transpose((1, 2, 0))
    inp = inp.numpy().squeeze()
    print(inp.max())
    print(inp.shape)
    print(std_icon)
    #inp = std_icon * inp + mean_icon
    print(inp.max())
    #inp = np.clip(inp, 0, 1)
    print(inp.max())
    #inp = (inp * 255).astype(np.uint8)
    print(inp.max())
    plt.imshow(inp)


# In[ ]:


_, x= next(enumerate(train_loader_gray))
samples = x.to(device)

samples_rec,   _, _ = best_model(samples)
samples_rec = samples_rec.detach().cpu()

for i in range(0, 3):
    plt.subplot(3,2,2*i+1)
    plt.tight_layout()
    #imshow((samples[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    imshow(samples[i])
    plt.title("Ori. {}".format(i))

    plt.subplot(3, 2, 2*i+2)
    plt.tight_layout()
    imshow(samples_rec[i])
    plt.title("Rec. {}".format(i))


# TODO:
# - use other architectures
# - tune hyperparameters; latent space dimensions
# - GPU