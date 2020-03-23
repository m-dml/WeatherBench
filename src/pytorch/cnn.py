import numpy as np
import torch
from .layers import PeriodicConv2D


class SimpleCNN(torch.nn.Module):
    """ Simple CNN module for image-to-image regression
        Assumes image height and width are the same for input and output.
        Note that default constructor uses PeriodicConv2D layers !
    """

    def __init__(self, filters, kernels, channels, activation, mode='circular'):
        super(SimpleCNN, self).__init__()
        self.layers, in_ = [], channels
        self.activation = activation
        assert not np.any(kernels == 2), 'kernel size 2 not allowed for circular padding'
        in_channels = [channels] + list(filters[:-1])
        if mode=='circular':
            self.layers = torch.nn.ModuleList([PeriodicConv2D(i, f, k, padding=(k-1, k-1),
                            padding_mode='circular') for i,f,k in zip(in_channels, filters, kernels)])
        else:
            self.layers = torch.nn.ModuleList([torch.nn.Conv2d(i, f, k, padding=k//2) 
                                               for i,f,k in zip(in_channels, filters, kernels)])
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm2d(i) for i in filters[:-1]])

    def forward(self, x):
        for layer, bn in zip(self.layers[:-1], self.bns[:-1]):
            x = bn(self.activation(layer(x)))
        x = self.layers[-1](x)
        return x
