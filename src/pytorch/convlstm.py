import torch
import torch.nn as nn
from src.pytorch.third_party.convlstm import ConvLSTMCell, ConvLSTM
from src.pytorch.layers import PeriodicConv2D

class CircConvLSTMCell(ConvLSTMCell):
    """
    ConvLSTMCell implementation from 
    https://github.com/ndrplz/ConvLSTM_pytorch
    We here allow circular 2D convolutions (padding wrapped around last input axis).
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, padding_mode):
        """
        Initialize CircConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        padding_mode: str
            How to pad the data ('circular' for wrap-around padding on last axis)
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.padding_mode = padding_mode
        self.bias = bias

        if self.padding_mode == 'circular':
            self.conv = PeriodicConv2D(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim, 
                              kernel_size=self.kernel_size, 
                              padding=[i-1 for i in self.kernel_size], 
                              bias=self.bias, 
                              padding_mode=padding_mode)
        else:
            self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

class CircConvLSTM(ConvLSTM):
    """
    ConvLSTM implementation from 
    https://github.com/ndrplz/ConvLSTM_pytorch
    We here use CircConvLSTM cells that allow circular 2D convolutions.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False, padding_mode=None):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.padding_mode = padding_mode

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(CircConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          padding_mode=self.padding_mode))

        self.cell_list = nn.ModuleList(cell_list)