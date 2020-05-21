import torch
import numpy as np
from src.pytorch.util import init_torch_device
from src.pytorch.util import named_network

device = init_torch_device()

n_channels = 23
n_output = 2
seq_length = 13
dropout_rate = 0.1
filters = [32, 32, 32, 32, 32, 32, 32, 32]
n_hidden = [64, 64, 64, 64, 64, 64, 64, 64]
D_out = [32, 32, 32, 32, 32, 32, 32, 32]

kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3]
target_var_dict = [None, None]

kwargs = {
    'kernel_sizes' : kernel_sizes,
    'filters' : filters,
    'filters_ff' : n_hidden,
    'dropout_rate' : dropout_rate,
    'N_h' : 8,
    'D_h' : 3,
    'D_k' : 3,
    'D_out' : D_out,
    'layerNorm' : 'BN',
    'blockType' : 'stacking',
    'stride_qk' : 2
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