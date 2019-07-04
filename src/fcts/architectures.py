import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet


class VAE(nn.Module):

    def __init__(self, encoder, decoder, n_channels=3, latent_dim=8, n_filters=128, img_h=32, img_w=32,
                 batch_normalization=True, encoder_activation=nn.LeakyReLU(), decoder_activation=nn.ReLU()):
        """
        :param n_channels: number of color channels
        :param latent_dim: latent space vector dimensionality
        :param n_filters: filters of conv layers
        :param img_h: height
        :param img_w: width
        """
        super(VAE, self).__init__()

        # store arguments
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.img_h = img_h
        self.img_w = img_w
        self.img_size = img_h*img_w

        # encoder, decoder
        self.encoder = encoder(n_channels=n_channels, latent_dim=latent_dim, n_filters=n_filters,
                               img_h=img_h, img_w=img_w, batch_normalization=batch_normalization,
                               activation=encoder_activation)
        self.decoder = decoder(n_channels=n_channels, latent_dim=latent_dim, n_filters=n_filters,
                               img_h=img_h, img_w=img_w, batch_normalization=batch_normalization,
                               activation=decoder_activation)

    def forward(self, _x):
        """
        Forward process of whole VAE
        :param _x: batch of images, size=[batch_size, n_channels, img_w, img_h]
        :return: batch of reconstructed images, size=input.size()
        """
        # encode
        _means, _log_vars = self.encoder(_x)
        # reparameterization
        _std = torch.exp(.5 * _log_vars)
        _eps = torch.randn_like(_std)
        _z = _means + _eps * _std
        # decode
        _recon_x = self.decoder(_z)

        return _recon_x, _means, _log_vars


class EncoderDCGAN32(nn.Module):

    def __init__(self, n_channels=3, latent_dim=8, n_filters=128, img_h=32, img_w=32,
                 batch_normalization=True, activation=nn.LeakyReLU()):
        """

        :param n_channels: number of color channels
        :param latent_dim: latent space vector dimensionality
        :param n_filters: number of filters
        :param img_h: height
        :param img_w: width
        :param batch_normalization: (bool) use batch normalization? default=True
        :param activation: function, default nn.LeakyReLU()
        """
        super(EncoderDCGAN32, self).__init__()

        # store some arguments
        self.n_filters = n_filters
        self.batch_norm = batch_normalization
        self.img_h = img_h
        self.img_w = img_w
        # arguments for conv layers
        kernel = 5
        stride = 2
        padding = 2
        # activation function
        self.activation = activation
        # convolutional layers
        self.conv1 = nn.Conv2d(n_channels, n_filters, kernel_size=kernel, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(n_filters, 2*n_filters, kernel_size=kernel, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(2*n_filters, 4*n_filters, kernel_size=kernel, stride=stride, padding=padding)
        # batch norm layers
        if batch_normalization:
            self.bn1 = nn.BatchNorm2d(n_filters)
            self.bn2 = nn.BatchNorm2d(2*n_filters)
            self.bn3 = nn.BatchNorm2d(4*n_filters)
        # compute input dim of linear layer
        conv3_out_dim = img_h
        for _ in range(3):
            conv3_out_dim = int(np.floor((conv3_out_dim + 2 * padding - (kernel - 1) - 1) / stride + 1))
        self.fc_in_dim = 4*n_filters*conv3_out_dim**2
        # linear layers
        self.fc_means = nn.Linear(self.fc_in_dim, latent_dim)
        self.fc_logvars = nn.Linear(self.fc_in_dim, latent_dim)

    def forward(self, _x):
        """
        :param _x: batch of images, size=[batch_size, n_channels, img_w, img_h]
        :return: batch of means, log_vars for latent space representation
        """
        batch_size = _x.size(0)
        # convolutions
        out = self.activation(self.conv1(_x))
        out = self.activation(self.bn2(self.conv2(out))) if self.batch_norm else self.activation(self.conv2(out))
        out = self.activation(self.bn3(self.conv3(out))) if self.batch_norm else self.activation(self.conv3(out))
        # reshape
        out = out.view(batch_size, -1)
        # check dimensions
        assert out.size(1) == self.fc_in_dim, \
            print(f'Mismatching dimensions after convolution, '
                  f'out.size(1)={out.size(1)} but should be {self.fc_in_dim}.')
        # fully connected
        _means = self.fc_means(out)
        _log_vars = self.fc_logvars(out)

        return _means, _log_vars


class DecoderDCGAN32(nn.Module):
    def __init__(self, n_channels=3, latent_dim=8, n_filters=128, img_h=32, img_w=32,
                 batch_normalization=True, activation=nn.ReLU()):
        """

        :param n_channels: number of color channels
        :param latent_dim: latent space vector dimensionality
        :param n_filters: number of filters
        :param img_h: height
        :param img_w: width
        :param batch_normalization: (bool) use batch normalization? default=True
        :param activation: function, default nn.ReLU()
        """
        super(DecoderDCGAN32, self).__init__()
        # store some arguments
        self.n_filters = n_filters
        self.batch_norm = batch_normalization
        self.img_h = img_h
        self.img_w = img_w
        self.n_channels = n_channels
        # arguments for conv layers
        kernel = 5
        stride = 2
        padding = 2
        output_padding = 1
        # activation function
        self.activation = activation
        # compute output dim of linear layer
        conv_in_dim = img_h
        for _ in range(3):
            conv_in_dim = int(np.floor((conv_in_dim + 2 * padding - (kernel - 1) - 1) / stride + 1))
        self.conv_in_dim = conv_in_dim
        self.fc_out_dim = 4*n_filters*conv_in_dim**2
        # linear layer
        self.fc = nn.Linear(latent_dim, self.fc_out_dim)
        # convolutional layers
        self.deconv1 = nn.ConvTranspose2d(4*n_filters, 2*n_filters, kernel_size=kernel, stride=stride,
                                          padding=padding, output_padding=output_padding)
        self.deconv2 = nn.ConvTranspose2d(2*n_filters, n_filters, kernel_size=kernel, stride=stride,
                                          padding=padding, output_padding=output_padding)
        self.deconv3 = nn.ConvTranspose2d(n_filters, n_channels, kernel_size=kernel, stride=stride,
                                          padding=padding, output_padding=output_padding)
        # batch norm layers
        if batch_normalization:
            self.bn_fc = nn.BatchNorm1d(self.fc_out_dim)
            self.bn1 = nn.BatchNorm2d(2*n_filters)
            self.bn2 = nn.BatchNorm2d(n_filters)

    def forward(self, _z):
        """
        :param _z: batch latent space vectors, size=[batch_size, latent_dim]
        :return: batch of generated images
        """
        batch_size = _z.size(0)
        # linear
        out = self.activation(self.bn_fc(self.fc(_z))) if self.batch_norm else self.activation(self.fc(_z))
        assert out.size(1) == self.fc_out_dim, \
            print(f'Mismatching dimensions before convolution, out.size(1)={out.size()} but should be '
                  f'[{batch_size}, {4 * self.n_filters}, {self.conv_in_dim}, {self.conv_in_dim}].')
        # reshape
        out = out.view(-1, 4*self.n_filters, self.conv_in_dim, self.conv_in_dim)
        assert out.size(0) == batch_size, print(f'out.size(0)={out.size(0)} but should be {batch_size}')
        # convolutions
        out = self.activation(self.bn1(self.deconv1(out))) if self.batch_norm else self.activation(self.deconv1(out))
        out = self.activation(self.bn2(self.deconv2(out))) if self.batch_norm else self.activation(self.deconv2(out))
        out = self.deconv3(out)
        # check dimensions
        if out.size(2) != self.img_w:
            print(f'Reconstructed image is {out.size(2)}x{out.size(3)} but should be {self.img_h}x{self.img_w}. '
                  'Center crop each image.')
            out = out[:, :, int((out.size(2)-self.img_h)/2):-int((out.size(2)-self.img_h)/2),
                      int((out.size(2)-self.img_h)/2):-int((out.size(2)-self.img_h)/2)]
        # output
        out = nn.Tanh()(out)
        
        return out


class EncoderRESNET32(nn.Module):

    def __init__(self, n_channels=3, latent_dim=8, n_filters=128, img_h=32, img_w=32,
                 batch_normalization=None, activation=None):
        """

        :param n_channels: number of color channels
        :param latent_dim: latent space vector dimensionality
        :param n_filters: number of filters
        :param img_h: height
        :param img_w: width
        """
        super(EncoderRESNET32, self).__init__()

        # store some arguments
        self.n_filters = n_filters
        self.img_h = img_h
        self.img_w = img_w
        # residual blocks
        self.res_block1 = resnet.BasicBlock(
            n_channels, n_filters, downsample=nn.Sequential(resnet.conv1x1(n_channels, n_filters, 1), nn.BatchNorm2d(n_filters),))
        self.res_block2 = resnet.BasicBlock(
            n_filters, 2*n_filters, downsample=nn.Sequential(resnet.conv1x1(n_filters, 2*n_filters, 1), nn.BatchNorm2d(2*n_filters),))
        self.res_block3 = resnet.BasicBlock(
            2*n_filters, 4*n_filters, downsample=nn.Sequential(resnet.conv1x1(2*n_filters, 4*n_filters, 1), nn.BatchNorm2d(4*n_filters),))
        # linear layers
        self.fc_in_dim = 4*n_filters*img_h*img_w
        self.fc_means = nn.Linear(self.fc_in_dim, latent_dim)
        self.fc_logvars = nn.Linear(self.fc_in_dim, latent_dim)

    def forward(self, _x):
        """
        :param _x: batch of images, size=[batch_size, n_channels, img_w, img_h]
        :return: batch of means, log_vars for latent space representation
        """
        batch_size = _x.size(0)
        # convolutions
        out = self.res_block1(_x)
        out = self.res_block2(out)
        out = self.res_block3(out)
        # reshape
        out = out.view(batch_size, -1)
        # check dimensions
        assert out.size(1) == self.fc_in_dim, \
            print(f'Mismatching dimensions after residual blocks, '
                  f'out.size(1)={out.size(1)} but should be {self.fc_in_dim}.')
        # fully connected
        _means = self.fc_means(out)
        _log_vars = self.fc_logvars(out)

        return _means, _log_vars


class DecoderRESNET32(nn.Module):
    def __init__(self, n_channels=3, latent_dim=8, n_filters=128, img_h=32, img_w=32,
                 batch_normalization=None, activation=None, final_activation=nn.Tanh()):
        """

        :param n_channels: number of color channels
        :param latent_dim: latent space vector dimensionality
        :param n_filters: number of filters
        :param img_h: height
        :param img_w: width
        """
        super(DecoderRESNET32, self).__init__()
        # store some arguments
        self.n_filters = n_filters
        self.img_h = img_h
        self.img_w = img_w
        self.n_channels = n_channels
        # activation function of last layer
        self.activation = final_activation
        # compute output dim of linear layer
        self.fc_out_dim = 4*n_filters*img_h*img_w
        # linear layer
        self.fc = nn.Linear(latent_dim, self.fc_out_dim)
        # residual blocks
        self.res_block1 = resnet.BasicBlock(
            4*n_filters, 2*n_filters, downsample=nn.Sequential(resnet.conv1x1(4*n_filters, 2*n_filters, 1), nn.BatchNorm2d(2*n_filters),))
        self.res_block2 = resnet.BasicBlock(
            2*n_filters, n_filters, downsample=nn.Sequential(resnet.conv1x1(2*n_filters, n_filters, 1), nn.BatchNorm2d(n_filters),))
        self.res_block3 = resnet.BasicBlock(
            n_filters, n_channels, downsample=nn.Sequential(resnet.conv1x1(n_filters, n_channels, 1), nn.BatchNorm2d(n_channels),))

    def forward(self, _z):
        """
        :param _z: batch latent space vectors, size=[batch_size, latent_dim]
        :return: batch of generated images
        """
        batch_size = _z.size(0)
        # linear
        out = self.fc(_z)
        assert out.size(1) == self.fc_out_dim, \
            print(f'Mismatching dimensions before residual blocks, out.size(1)={out.size()} but should be '
                  f'[{batch_size}, {4 * self.n_filters}, {self.img_h}, {self.img_w}].')
        # reshape
        out = out.view(-1, 4*self.n_filters, self.img_h, self.img_w)
        assert out.size(0) == batch_size, print(f'out.size(0)={out.size(0)} but should be {batch_size}')
        # convolutions
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)
        # check dimensions
        if out.size(1) != self.n_channels:
            print(f'Reconstructed image has {out.size(1)} channels, but should have {self.n_channels}.')
        # output
        out = self.activation(out)
        
        return out


class BaselineVAE(nn.Module):

    def __init__(self, encoder_layer_sizes, decoder_layer_sizes, latent_dim, n_channels):
        """
        :param encoder_layer_sizes: sizes of the encoder layers; list
        :param decoder_layer_sizes: sizes of the decoder layers; list
        :param latent_dim: dimension of latent space/bottleneck; int
        :param n_channels: number of channels of images; int
        """

        super(BaselineVAE, self).__init__()

        self.latent_dim = latent_dim
        self.channels = n_channels

        self.encoder = EncoderBaseline(encoder_layer_sizes, latent_dim, n_channels)
        self.decoder = DecoderBaseline(decoder_layer_sizes, latent_dim, n_channels)

    def forward(self, x):
        """
        :param x: tensor of dimension
        :return recon_x: reconstructed x
        :return means: output of encoder
        :return log_var: output of encoder (logarithm of variance)
        """
        batch_size = x.size(0)

        x = x.view(-1, self.channels, x.size(-1) ** 2)
        assert x.size(0) == batch_size
        means, log_vars = self.encoder(x)
        std = torch.exp(.5 * log_vars)
        eps = torch.randn_like(std)
        z = means + eps * std
        recon_x = self.decoder(z)

        return recon_x, means, log_vars


class EncoderBaseline(nn.Module):

    def __init__(self, layer_sizes, latent_dim, n_channels):
        super(EncoderBaseline, self).__init__()
        """
        :param layer_sizes: list of sizes of layers of the encoder; list[int]
        :param latent_dim: dimension of latent space, i.e. dimension out output of the encoder; int
        :param n_channels: number of channels of images; int
        """
        # store values
        self.latent_dim = latent_dim
        self.channels = n_channels
        self.layer_sizes = layer_sizes

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
        batch_size = x.size(0)
        x = x.view(-1, self.layer_sizes[0])
        assert x.size(0) == batch_size

        # forward
        out = self.layers(x)

        # latent space output
        means = self.out_mean(out)
        log_vars = self.out_var(out)

        return means, log_vars


class DecoderBaseline(nn.Module):

    def __init__(self, layer_sizes, latent_dim, n_channels):
        super(DecoderBaseline, self).__init__()
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
        h = int(np.sqrt(self.layer_sizes[-1]/self.channels))
        x = torch.sigmoid(self.layers(z)).view(-1, self.channels, h, h)

        return x
