import numpy as np
import torch
from src.pytorch.layers import PeriodicConv2D

def setup_conv(in_channels, out_channels, kernel_size, padding, bias, padding_mode):
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
            torch.zeros(self.D_out,1,1)
        )

    def forward(self, x):

        gates_qk = torch.split(self.conv_qk(x), self.D_k, dim=1)
        # add dropout here?
        gates_v = torch.split(self.conv_v(x), self.D_h, dim=1)
        # add dropout here?

        k = torch.prod(torch.tensor(gates_qk[0].shape[1:], requires_grad=False, dtype=torch.float32))
        sqrk = torch.sqrt(k)
        X_h = []
        for h in range(self.N_h): # per attention head, do
            X_q, X_k = gates_qk[2*h:2*(h+1)]                  # [N, D_k, H, W]
            X_v = gates_v[h]                                  # [N, D_h, H, W]
            A = torch.einsum('ncij,mcij->nm', X_q, X_k)       # [N, N]
            softA = torch.nn.functional.softmax(A/sqrk,dim=1) # [N, N]
            SA_v = torch.einsum('ik,kjnc->ijnc', softA, X_v)  # [N, D_h, H, W]
            X_h.append(SA_v)

        X_h = torch.cat(X_h, axis=1)                          # [N, N_h*D_h, H, W]
        out = torch.einsum('iknc,kj->ijnc', X_h, self.W_out)  # [N, D_out, H, W]

        return out + self.b_out


class ConvTransformerEncoderLayer(torch.nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, in_channels, D_out, hidden_channels, out_channels,
                 kernel_size, N_h, D_h, D_k, attention_kernel_size,
                 bias=True, attention_bias=True, LayerNorm=torch.nn.LayerNorm,
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

        #self.norm1 = LayerNorm(d_model)
        #self.norm2 = LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)


        if activation == "relu":
            self.activation =  torch.nn.functional.relu
        elif activation == "gelu":
            self.activation =  torch.nn.functional.gelu
        else:
            raise RuntimeError("activation should be relu/gelu, not {}".format(activation))        

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        #src = self.norm1(src)
        src2 = self.conv2(self.dropout(self.activation(self.conv1(src))))
        src = src + self.dropout2(src2)
        #src = self.norm2(src)
        return src


class ConvTransformer(torch.nn.Module):
    """ Simple fully convolutional ResNet with variable number of blocks and layers
    """
    def __init__(self,
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
                 LayerNorm=torch.nn.LayerNorm,
                 padding_mode='circular', 
                 dropout=0.1, 
                 activation="relu"):
        """
        Initialize ConvSALayer layer.
        Parameters
        ----------
        in_channels: int
            Number of channels of input tensor.
        D_out: int
            Number of channels of output tensor of self-attention.
        filters: int
            Number of channels of output tensor of convolutions.
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
        super(ConvTransformer, self).__init__()
    
        sa_kernel_sizes = kernel_sizes if sa_kernel_sizes is None else sa_kernel_sizes
        D_out = filters if D_out is None else D_out
        assert len(filters) == len(kernel_sizes)

        layers = []
        for sa_kernel_size, kernel_size, n_filters in zip(sa_kernel_sizes, kernel_sizes, filters):
            layers.append( ConvTransformerEncoderLayer( in_channels=in_channels, 
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
                                                        LayerNorm=LayerNorm,
                                                        padding_mode=padding_mode, 
                                                        dropout=dropout, 
                                                        activation=activation)
                          )
            in_channels = n_filters
        self.layers = torch.nn.ModuleList(modules=layers)

        self.final = torch.nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels, 
                                     kernel_size=(1,1), 
                                     stride=1)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return self.final(x.contiguous())
