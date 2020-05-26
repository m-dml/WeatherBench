import torch
import numpy as np
from src.pytorch.util import init_torch_device
from src.pytorch.util import named_network

device = init_torch_device()

n_channels = 23
n_output = 2
seq_length = 13

kwargs = {
    # initial convolutional layers applied to each sequence member separately
    'kernel_size_init' : 3,
    'filters_ff_init' : [64,64],

    # Transformer layers with feedforward networks being convolutional residual nets
    'kernel_sizes' : [3, 3, 3],
    'filters' : [32, 32, 64],                     # n_channels for output stage of MHSA
    'filters_ff' : [[64,64], [64,64], [128,128]], # n_filters for feedforward blocks
    'N_h' : 8,
    'D_h' : 8,
    'D_k' : 16,
    'D_out' : [24, 24, 24],
    'blockType' : 'stacking',
    'stride_qk' : 2,

    # final convolutional layers combining applied to stacked channels from all sequence members
    'kernel_size_final' : 3,
    'filters_ff_final' : [128, 128, 128, n_output],

    # global setting applied to each layer in each block
    'dropout_rate' : 0.1,
    'layerNorm' : 'BN'
}
model_name = 'ConvTransformer'
model, model_forward, _ = named_network(model_name, n_channels, n_output, seq_length, **kwargs)


inputs = torch.tensor(
    np.random.normal(size=(16, seq_length, 23, 32, 64)),
    requires_grad=False,
    dtype=torch.float32)
print(model_forward(inputs).shape)

print('total #parameters: ', np.sum([np.prod(item.shape) for item in model.state_dict().values()]))

loss = torch.sum(model_forward(inputs) - inputs[:,0,:2,:,:])
loss.backward()
print('took gradient')