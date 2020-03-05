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
x = x.chunk({'time' : np.sum(x.chunks['time']), 'lat' : x.chunks['lat'], 'lon': x.chunks['lon']})
n_channels = len(dataset_list) # = 1 if only loading one of geopotential Z500 and temperature T850

# tbd: separating train and test datasets / loaders should be avoidable with the start/end arguments of Dataset!

dg_train = Dataset(x.sel(time=slice('1979', '2015')), var_dict, lead_time, normalize=True)
train_loader = torch.utils.data.DataLoader(
    dg_train,
    batch_size=batch_size,
    drop_last=True)

dg_validation =  Dataset(x.sel(time=slice('2016', '2016')), var_dict, lead_time,
                        mean=dg_train.mean, std=dg_train.std, normalize=True)
validation_loader = torch.utils.data.DataLoader(
    dg_validation,
    batch_size=batch_size,
    drop_last=False)

## define model

from src.train_nn_pytorch import SimpleCNN

net = SimpleCNN(filters=[64, 64, 64, 64, n_channels], kernels=[5, 5, 5, 5, 5], 
          channels=n_channels, activation=torch.nn.functional.elu, mode='circular')

## train model

import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy

optimizer = optim.Adam(net.parameters(), lr=1e-4)

n_epochs, max_patience = 200, 20
losses, best_loss, patience = np.zeros(n_epochs), np.inf, max_patience
best_state_dict = {}

epoch = 0
while True:

    epoch += 1
    if epoch > n_epochs:
        break

    # Train for a single epoch.
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, targets = batch[0].to(device), batch[1].to(device)
        loss = F.mse_loss(net.forward(inputs), targets)
        loss.backward()
        optimizer.step()

    # Track convergence on validation set.
    val_loss = 0
    with torch.no_grad():
        nb = 0
        for batch in validation_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            val_loss += F.mse_loss(net.forward(inputs), targets)
            nb += 1
    val_loss /= nb
    print(f'epoch #{epoch} || loss (last batch) {loss} || validation loss {val_loss}')

    if val_loss < best_loss:
        patience = max_patience
        best_loss = val_loss
        best_state_dict = deepcopy(net.state_dict())        
        torch.save(best_state_dict, res_dir + 'test_fccnn_3d_pytorch.pt')

    else:
        patience -= 1

    if patience == 0:
        net.load_state_dict(best_state_dict)
        break

#net_rec = Net(filters=[64, 64, 64, 64, n_channels], kernels=[5, 5, 5, 5, 5], 
#          channels=n_channels, activation=torch.nn.functional.elu)
#net_rec.load_state_dict(torch.load(res_dir + 'test_fccnn_3d_pytorch.pt'))