import numpy as np
import torch
from .layers import PeriodicConv2D


class CircUNet(torch.nn.Module):
    """ Simple UNet module for image-to-image regression
        Assumes image height and width are the same for input and output.
        Note that default constructor uses PeriodicConv2D layers !
    """
    def __init__(self, in_channels, out_channels, filters, kernels, pooling, activation, mode='circular'):
        super(CircUNet, self).__init__()
        assert not np.any(kernels == 2), 'kernel size 2 not allowed for circular padding'
        ziplist = zip([in_channels] + [f[0] for f in filters[:-1]], filters, kernels)
        self.downlayers = [Downscaler(i, f, k, pooling,  activation, mode='circular') for i,f,k in ziplist]

        # bottom layer: number of filters actually has to increase
        i, f, k, o = filters[-1][-1], [2*f for f in filters[-1]], kernels[-1], filters[-1][-1]
        self.uplayers = [Upscaler(i, f, k, pooling, o, activation, mode='circular')]
        
        ziplist = zip([2*f[-1] for f in filters[::-1]], filters[::-1], kernels[::-1], [f[-1] for f in filters[::-1][1:]])
        self.uplayers += [Upscaler(i, f, k, pooling, o, activation, mode='circular') for i,f,k,o in ziplist]

        self.downlayers, self.uplayers = torch.nn.ModuleList(self.downlayers), torch.nn.ModuleList(self.uplayers)
        
        i, f, k = 2*filters[0][0], out_channels, kernels[0][-1]
        if mode=='circular':
            self.final = PeriodicConv2D(i, f, k, padding=(k-1, k-1), padding_mode='circular')
        else:
            self.final = torch.nn.Conv2d(i, f, k, padding=k//2)

    def forward(self, x):
        outs = []
        for layer in self.downlayers:
            x, out = layer(x)
            outs += [out]
        for layer, out in zip(self.uplayers, outs[::-1]):
            x = layer(x, out)
        return self.final(x)


class Downscaler(torch.nn.Module):
    def __init__(self, in_channels, filters, kernels, pooling, activation, mode='circular'):
        super(Downscaler, self).__init__()
        ziplist = zip([in_channels] + list(filters[:-1]), filters, kernels)
        if mode=='circular':
            self.layers = [PeriodicConv2D(i, f, k, padding=(k-1, k-1),
                          padding_mode='circular') for i,f,k in ziplist]
        else:
            self.layers = [torch.nn.Conv2d(i, f, k, padding=k//2) for i,f,k in ziplist]
        self.layers = torch.nn.ModuleList(self.layers)
        self.pooling = torch.nn.MaxPool2d(pooling)
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm2d(i) for i in filters])
        self.activation = activation

    def forward(self, x):
        for layer, bn in zip(self.layers, self.bns):            
            x = bn(self.activation(layer(x)))
        return self.pooling(x), x


class Upscaler(torch.nn.Module):
    def __init__(self, in_channels, filters, kernels, pooling, out_channel, activation, mode='circular'):
        super(Upscaler, self).__init__()
        ziplist = zip([in_channels] + list(filters[:-1]), filters, kernels)
        if mode=='circular':
            self.layers = [PeriodicConv2D(i, f, k, padding=(k-1, k-1),
                          padding_mode='circular') for i, f, k in ziplist]
        else:
            self.layers = [torch.nn.Conv2d(i, f, k, padding=k//2) for i,f,k in ziplist]
        self.layers = torch.nn.ModuleList(self.layers)
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm2d(i) for i in filters])
        self.uplayer = torch.nn.ConvTranspose2d(filters[-1], out_channel, pooling, stride=2)
        self.activation = activation

    def forward(self, x, xskip):
        for layer, bn in zip(self.layers, self.bns):            
            x = bn(self.activation(layer(x)))
        x = self.uplayer(x) 
        return torch.cat((x,xskip), axis=1) # Nx(C+Cskip)xHxW 
