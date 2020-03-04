import numpy as np
import xarray as xr
import torch
from src.train_nn_pytorch import Dataset

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    print("CUDA not available")
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

lead_time = 3*24
var_dict = {'z': None, 't': None}
batch_size = 32

datadir = '/gpfs/work/nonnenma/data/forecast_predictability/weatherbench/5_625deg/'
res_dir = '/gpfs/work/nonnenma/results/forecast_predictability/weatherbench/5_625deg/'

z500 = xr.open_mfdataset(f'{datadir}geopotential_500/*.nc', combine='by_coords')
t850 = xr.open_mfdataset(f'{datadir}temperature_850/*.nc', combine='by_coords')
dataset_list = [z500, t850]
x = xr.merge(dataset_list, compat='override')
n_channels = len(dataset_list) # = 1 if only loading one of geopotential Z500 and temperature T850

# tbd: separating train and test datasets / loaders should be avoidable with the start/end arguments of Dataset!

dg_train = Dataset(x.sel(time=slice('2015', '2015')), var_dict, lead_time, normalize=True)
train_loader = torch.utils.data.DataLoader(
    dg_train,
    batch_size=batch_size,
    drop_last=True)

dg_test =  Dataset(x.sel(time=slice('2016', '2016')), var_dict, lead_time,
                        mean=dg_train.mean, std=dg_train.std, normalize=True)
test_loader = torch.utils.data.DataLoader(
    dg_test,
    batch_size=batch_size,
    drop_last=False)

## define model

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

class Net(torch.nn.Module):
    
    def __init__(self, filters, kernels, channels, activation):
        super(Net, self).__init__()
        self.layers, in_ = [], channels
        self.activation = activation
        assert not np.any(kernels == 2), 'kernel size 2 not allowed for circular padding'
        in_channels = [channels] + list(filters[:-1])
        self.layers = torch.nn.ModuleList([PeriodicConv2D(i, f, k, padding=(k-1, k-1),
                        padding_mode='circular') for i,f,k in zip(in_channels, filters, kernels)])
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

net = Net(filters=[64, 64, 64, 64, n_channels], kernels=[5, 5, 5, 5, 5], 
          channels=n_channels, activation=torch.nn.functional.elu)

## train model

import torch.optim as optim
optimizer = optim.Adam(net.parameters(), lr=0.01)

n_epochs = 200
epoch = 0
while True:

    i = 0
    epoch += 1
    # Train for a single epoch.
    for batch in train_loader:
        print('minibatch #' + str(i))
        optimizer.zero_grad()
        inputs, targets = batch[0].to(device), batch[1].to(device)
        loss = F.mse_loss(net.forward(inputs), targets)
        loss.backward()
        optimizer.step()        
        i += 1

    # tbd: write early stopping from convergence on vakidation data
    if epoch > n_epochs:
        break
        
torch.save(net.state_dict(), res_dir + 'test_fccnn_3d_pytorch.pt')

#net_rec = Net(filters=[64, 64, 64, 64, n_channels], kernels=[5, 5, 5, 5, 5], 
#          channels=n_channels, activation=torch.nn.functional.elu)
#net_rec.load_state_dict(torch.load(res_dir + 'test_fccnn_3d_pytorch.pt'))