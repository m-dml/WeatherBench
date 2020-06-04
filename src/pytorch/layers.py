import torch
import torch.nn.functional as F

class PeriodicConv2D(torch.nn.Conv2d):
    """ Implementing 2D convolutional layer with mixed zero- and circular padding.
    Uses circular padding along last axis (W) and zero-padding on second-last axis (H)

    """
    
    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding_circ = ( (self.padding[0] + 1) // 2, self.padding[0] // 2, 0, 0)
            expanded_padding_zero = ( 0, 0, (self.padding[1] + 1) //2, self.padding[1] // 2 )
            return F.conv2d(F.pad(F.pad(input, expanded_padding_circ, mode='circular'), 
                                  expanded_padding_zero, mode='constant'),
                            weight, self.bias, self.stride,
                            (0,0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def setup_conv(in_channels, out_channels, kernel_size, padding, bias, padding_mode, stride=1):
    """
    Select between regular and circular 2D convolutional layers.
    padding_mode='circular' returns a convolution that wraps padding around the final axis.
    """
    if padding_mode=='circular':
        return PeriodicConv2D(in_channels=in_channels,
                      out_channels=out_channels, 
                      kernel_size=kernel_size, 
                      padding=[i-1 for i in kernel_size], 
                      bias=bias, 
                      stride=stride,
                      padding_mode=padding_mode)
    else:
        return torch.nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride,
                              bias=bias)


class ResNetBlock(torch.nn.Module):
    """A residual block to construct residual networks.
    Comprises 2 conv2D operations with optional dropout and a normalization layer.
    Additive shortcut connection force input channels to match output channels.   
    
    Parameters
    ----------
    in_channels: int
        Number of channels of input tensor.
    kernel_size: list of (int, int)
        Size of the convolutional kernel for the residual layers.
    hidden_channels: int
        Number of output channels for first residual convolution. 
    out_channels: int
        Number of output channels. If not equal to in_channels, will add
        additional 1x1 convolution. 
    bias: bool
        Whether to include bias parameters in the residual-layer convolutions.
    layerNorm: function
        Normalization layer.
    activation: str
        String specifying nonlinearity.
    padding_mode: str
        How to pad the data ('circular' for wrap-around padding on last axis)
    dropout: float
        Dropout rate.        
    """
    def __init__(self, in_channels, kernel_size, 
                 hidden_channels=None, out_channels=None, 
                 bias=True, layerNorm=torch.nn.BatchNorm2d,
                 padding_mode='circular', dropout=0.1, activation="relu"):

        super(ResNetBlock, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        
        self.additive = out_channels == in_channels
        
        self.conv1 = setup_conv(in_channels=in_channels, 
                                  out_channels=hidden_channels, 
                                  kernel_size=kernel_size, 
                                  padding=(kernel_size[0] // 2, kernel_size[1] // 2), 
                                  bias=bias,
                                  padding_mode=padding_mode)

        n_out_conv2 = out_channels if self.additive else hidden_channels
        self.conv2 = setup_conv(in_channels=hidden_channels,
                                  out_channels=n_out_conv2,
                                  kernel_size=kernel_size,
                                  padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                                  bias=bias,
                                  padding_mode=padding_mode)
        
        if layerNorm is torch.nn.BatchNorm2d: 
            self.norm1 = layerNorm(num_features=hidden_channels)
            self.norm2 = layerNorm(num_features=n_out_conv2)
        elif isinstance(layerNorm, torch.nn.Identity):
            self.norm1 = self.norm2 = layerNorm
        else:
            raise NotImplementedError
                        
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        if not self.additive:
            self.conv1x1 = torch.nn.Conv2d(in_channels=in_channels+n_out_conv2,
                              out_channels=out_channels,
                              kernel_size=(1,1),
                              bias=bias)
            if layerNorm is torch.nn.BatchNorm2d: 
                self.norm1x1 = layerNorm(num_features=out_channels)
            elif isinstance(layerNorm, torch.nn.Identity):
                self.norm1x1 = layerNorm
            self.dropout1x1 = torch.nn.Dropout(dropout)

        if activation == "relu":
            self.activation =  torch.nn.functional.relu
        elif activation == "gelu":
            self.activation =  torch.nn.functional.gelu
        else:
            raise RuntimeError("activation should be relu/gelu, not {}".format(activation))        

    def forward(self, x, x_mask=None, x_key_padding_mask=None):
        """Pass the input through the encoder layer.
        
        Parameters
        ----------
        x: tensor
            The input sequence to the encoder layer.
        x_mask: tensor 
            Mask for the input sequence (optional).
        x_key_padding_mask: tensor 
            Mask for the x keys per batch (optional).
        """
        if self.additive:
            z =  self.dropout1(self.activation(self.norm1(self.conv1(x))))
            x += self.dropout2(self.activation(self.norm2(self.conv2(z))))
        else:
            z = self.dropout1(self.activation(self.norm1(self.conv1(x))))
            z = self.dropout2(self.activation(self.norm2(self.conv2(z))))
            x = self.dropout1x1(self.activation(self.norm1x1(self.conv1x1(torch.cat((x,z),axis=1)))))
            
        return x
