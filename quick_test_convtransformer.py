import torch
import numpy as np
from src.pytorch.util import init_torch_device
from src.pytorch.util import named_network

device = init_torch_device()

n_channels = 299
dropout_rate = 0.
filters=[299, 299, 299, 299, 299, 299]
kernel_sizes=[3, 3, 3, 3, 3, 3]

model_name = 'ConvTransformer'
model, model_forward, _ = named_network(model_name, n_channels, 2,  
                                     kernel_sizes=kernel_sizes, filters=filters, dropout_rate=dropout_rate)

inputs = torch.tensor(
    np.random.normal(size=(32, 13, 23, 32, 64)),
    requires_grad=False,
    dtype=torch.float32)
print(model_forward(inputs).shape)

print('total #parameters: ', np.sum([np.prod(item.shape) for item in model.state_dict().values()]))
