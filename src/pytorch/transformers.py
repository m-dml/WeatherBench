import numpy as np
import torch
from src.pytorch.layers import PeriodicConv2D

def setup_conv(in_channels, out_channels, kernel_size, padding, bias, padding_mode):
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
                      padding_mode=padding_mode)
    else:
        return torch.nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

def tensor5D_conv(conv, x, axis=0):
    """
    Convenience function to apply 2D convolutions to last two axes of 5D tensor.
    
    Parameters
    ----------
    conv: object
        Convolutional layer.
    x: tensor
        Input tensor.
    axis: int
        Onto which axis to collapse leading two axes of x. Default axis=0.
    Returns
    out: 5D tensor (if axis=0) or 4D tensor (if axis=1) resulting from conv(x.reshape(..))
    
    """
    N,T,C,H,W = batch_shape = x.shape                   # [N, T, C, H, W]

    if axis == 0:
        x = conv(x.reshape((N*T, *batch_shape[2:])))    # [N*T,  C, H, W]
        return x.reshape(N,T,-1,H,W)
    elif axis == 1:
        x = conv(x.reshape((N, T*C, *batch_shape[3:]))) # [N, T*C,  H, W]      
        return x.reshape(N,-1,H,W) # conv along axis=1 does not preserve sequence length T!
    else:
        raise NotImplementedError


class ConvMHSA(torch.nn.Module):
    """
    Implementation of multi-head self-attention with convolutions to compute queries, keys, values 
    """
    def __init__(self, D_in, D_out, N_h, D_h, D_k, kernel_size, bias, padding_mode, dropout=0.):
        """
        Initialize ConvSALayer layer.
        
        Parameters
        ----------
        D_in: int
            Number of channels of input tensor.
        D_out: int
            Number of channels of output tensor.
        N_h: int
            Number of attention heads.        
        D_h: int
            Number of output channels per attention head.
        D_k: int
            Number of channels for convolved query and key gates.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        padding_mode: str
            How to pad the data ('circular' for wrap-around padding on last axis)
        dropout: float
            Dropout rate.
        """

        super(ConvMHSA, self).__init__()

        self.D_in = D_in
        self.D_out = D_out
        self.N_h = N_h # number of heads
        self.D_k = D_k # number of filters per head
        self.D_h = D_h # output dim per head

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.padding_mode = padding_mode
        self.bias = bias
        
        self.dropout = dropout
        assert dropout == 0., 'dropout not yet implemented'

        self.conv_qk = setup_conv(in_channels=self.D_in, 
                                  out_channels=2*self.N_h * self.D_k, 
                                  kernel_size=self.kernel_size, 
                                  padding=self.padding, 
                                  bias=self.bias, 
                                  padding_mode=self.padding_mode)
        self.conv_v = setup_conv(in_channels=self.D_in,
                              out_channels=self.N_h * self.D_h, 
                              kernel_size=self.kernel_size, 
                              padding=self.padding, 
                              bias=self.bias, 
                              padding_mode=padding_mode)

        self.W_out = torch.nn.Parameter(
            torch.zeros(self.N_h * self.D_h, self.D_out)
        )
        self.b_out = torch.nn.Parameter(
            torch.zeros(1,1,self.D_out,1,1)
        )

    def forward(self, x):

        N,T,_,H,W = batch_shape = x.shape                     # [N, T, C, H, W]
        x = x.reshape((N*T, *batch_shape[2:]))                # [N*T, C, H, W]

        conv_qk = self.conv_qk(x).reshape(N,T,-1,H,W)
        gates_qk = torch.split(conv_qk, self.D_k, dim=2)      # [ [N, T, D_k, H, W] ]
        # add dropout here?

        conv_v = self.conv_v(x).reshape(N,T,-1,H,W)
        gates_v = torch.split(conv_v, self.D_h, dim=2)        # [ [N, T, D_h, H, W] ]
        # add dropout here?

        k = torch.prod(torch.tensor(gates_qk[0].shape[2:], requires_grad=False, dtype=torch.float32))
        sqrk = torch.sqrt(k)
        X_h = []
        for h in range(self.N_h): # per attention head, do
            X_q, X_k = gates_qk[2*h:2*(h+1)]                  # [N, T, D_k, H, W]
            X_v = gates_v[h]                                  # [N, T, D_h, H, W]
            A = torch.einsum('ntcij,nrcij->ntr', X_q, X_k)    # [N, T, T]
            softA = torch.nn.functional.softmax(A/sqrk,dim=2) # [N, T, T]
            SA_v = torch.einsum('ntr,nrcij->ntcij',softA,X_v) # [N, T, D_h, H, W]
            X_h.append(SA_v)

        X_h = torch.cat(X_h, axis=2)                          # [N, T, N_h*D_h, H, W]
        out = torch.einsum('ntcij,cd->ntdij', X_h,self.W_out) # [N, T, D_out, H, W]

        return out + self.b_out


class ConvTransformerEncoderBlock(torch.nn.Module):
    """TransformerEncoderBlock is made up of multi-head self-attention and a feedforward network.
    The computation of query, key and value of the self-attention, as well as the activations of
    the feedfoward network are based on 2D convolutions.
    
    Currently hard-coded to two convolutional layers for the feedforward network.
    
    Parameters
    ----------
    in_channels: int
        Number of channels of input tensor.
    D_out: int
        Number of output channels for self-attention value convolution. 
    hidden_channels: int
        Number of output channels for first residual convolution. 
    out_channels: int
        Number of channels of output tensor (output of second residual convolution).
    kernel_size: list of (int, int)
        Size of the convolutional kernel for the residual layers.
    bias: bool
        Whether to include bias parameters in the residual-layer convolutions.
    attention_kernel_size: list of (int, int)
        Size of the convolutional kernel for the self-attention layer.
    attention_bias: bool
        Whether to include bias parameters in the self-attention-layer convolutions.
    layerNorm: function
        Normalization layer.
    activation: str
        String specifying nonlinearity.
    N_h: int
        Number of attention heads.        
    D_h: int
        Number of output channels per attention head.
    D_k: int
        Number of channels for convolved query and key gates.
    padding_mode: str
        How to pad the data ('circular' for wrap-around padding on last axis)
    dropout: float
        Dropout rate.        
    """
    def __init__(self, in_channels, D_out, hidden_channels, out_channels,
                 kernel_size, N_h, D_h, D_k, attention_kernel_size,
                 bias=True, attention_bias=True, layerNorm=torch.nn.LayerNorm,
                 padding_mode='circular', dropout=0.1, activation="relu"):

        super(ConvTransformerEncoderLayer, self).__init__()

        assert in_channels == D_out, 'no up/down-sampling implemented yet'
        assert D_out == out_channels, 'no up/down-sampling implemented yet'
        self.self_attn = ConvMHSA(D_in=in_channels, D_out=D_out, 
                                  N_h=N_h, D_h=D_h, D_k=D_k,
                                  kernel_size=attention_kernel_size, bias=attention_bias, 
                                  padding_mode=padding_mode, dropout=dropout)
                    
        # Implementation of Feedforward model
        self.conv1 = setup_conv(in_channels=D_out, 
                                  out_channels=hidden_channels, 
                                  kernel_size=kernel_size, 
                                  padding=(kernel_size[0] // 2, kernel_size[1] // 2), 
                                  bias=bias, 
                                  padding_mode=padding_mode)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = setup_conv(in_channels=hidden_channels, 
                                  out_channels=out_channels, 
                                  kernel_size=kernel_size, 
                                  padding=(kernel_size[0] // 2, kernel_size[1] // 2), 
                                  bias=bias, 
                                  padding_mode=padding_mode)

        #self.norm1 = layerNorm(d_model)
        #self.norm2 = layerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

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
        x += self.dropout(self.self_attn(x))
        #x = self.norm1(x)
        
        xx = self.dropout1(self.activation(tensor5D_conv(self.conv1, x)))
        x += self.dropout2(self.activation(tensor5D_conv(self.conv2, xx)))
        #x = self.norm2(x)
        return x


class ConvTransformer(torch.nn.Module):
    """ Simple fully convolutional Transformer network with variable number of blocks.
    """
    def __init__(self,
                 seq_length,
                 in_channels,
                 out_channels,
                 filters,
                 kernel_sizes, 
                 N_h,
                 D_h,
                 D_k,
                 D_out=None, 
                 sa_kernel_sizes=None,
                 bias=True, 
                 attention_bias=True, 
                 layerNorm=torch.nn.LayerNorm,
                 padding_mode='circular', 
                 dropout=0.1, 
                 activation="relu"):
        """
        Initialize ConvSALayer layer.
        
        Parameters
        ----------
        seq_length: int
            Length of input sequences.
        in_channels: int
            Number of channels of input tensor.
        out_channels: int
            Number of channels of output tensor of self-attention.
        filters: list of (int, int)
            Number of channels of output tensor of convolutions.
        kernel_sizes: list of (int, int)
            Sizes of the convolutional kernel for the residual layers.
        N_h: int
            Number of attention heads.        
        D_h: int
            Number of output channels per attention head.
        D_k: int
            Number of channels for convolved query and key gates.
        D_out: int
            Number of output channels for self-attention value convolution. 
        kernel_sizes: list of (int, int)
            Sizes of the convolutional kernel for the self-attention layers.
        bias: bool
            Whether to include bias parameters in the residual-layer convolutions.
        attention_bias: bool
            Whether to include bias parameters in the self-attention-layer convolutions.
        layerNorm: function
            Normalization layer.
        bias: bool
            Whether or not to add the bias.
        padding_mode: str
            How to pad the data ('circular' for wrap-around padding on last axis)
        dropout: float
            Dropout rate.
        activation: str
            String specifying nonlinearity.
        """
        super(ConvTransformer, self).__init__()
    
        sa_kernel_sizes = kernel_sizes if sa_kernel_sizes is None else sa_kernel_sizes
        D_out = filters if D_out is None else D_out
        assert len(filters) == len(kernel_sizes)

        layers = []
        for sa_kernel_size, kernel_size, n_filters in zip(sa_kernel_sizes, kernel_sizes, filters):
            layers.append( ConvTransformerEncoderBlock( in_channels=in_channels, 
                                                        D_out=n_filters, 
                                                        hidden_channels=n_filters, 
                                                        out_channels=n_filters,
                                                        kernel_size=kernel_size, 
                                                        N_h=N_h, 
                                                        D_h=D_h, 
                                                        D_k=D_k, 
                                                        attention_kernel_size=sa_kernel_size,
                                                        bias=bias, 
                                                        attention_bias=attention_bias, 
                                                        layerNorm=layerNorm,
                                                        padding_mode=padding_mode, 
                                                        dropout=dropout, 
                                                        activation=activation)
                          )
            in_channels = n_filters
        self.layers = torch.nn.ModuleList(modules=layers)

        self.final = torch.nn.Conv2d(in_channels=in_channels*seq_length,
                                     out_channels=out_channels, 
                                     kernel_size=(1,1), 
                                     stride=1)

    def forward(self, x):
        """Pass the input through the network.
        
        Parameters
        ----------
        x: tensor
            The input sequence to the encoder layer.
        """
        for layer in self.layers:
            x = layer(x)

        return tensor5D_conv(self.final, x.contiguous(), axis=1)
