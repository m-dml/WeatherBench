import numpy as np
import xarray as xr
import torch
from src.train_nn_pytorch import Dataset
from torch.utils.data import RandomSampler

if torch.cuda.is_available():
    print('using CUDA !')
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    print("CUDA not available")
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

datadir = '/gpfs/work/nonnenma/data/forecast_predictability/weatherbench/5_625deg/'
res_dir = '/gpfs/work/nonnenma/results/forecast_predictability/weatherbench/5_625deg/'

lead_time = 3*24
batch_size = 32

"""
# geopotential and tempearture each at 11 levels 
# regression target is Z500 (level index i=6) and T850 (level index 9, stacked vector index 9+11=20)
geop = xr.open_mfdataset(f'{datadir}geopotential/*.nc', combine='by_coords')
temp = xr.open_mfdataset(f'{datadir}temperature/*.nc', combine='by_coords')

# specific humidity, and wind component (u,v) volumes, each at 11 levels 
sphq = xr.open_mfdataset(f'{datadir}specific_humidity/*.nc', combine='by_coords')
cowu = xr.open_mfdataset(f'{datadir}u_component_of_wind/*.nc', combine='by_coords')
cowv = xr.open_mfdataset(f'{datadir}v_component_of_wind/*.nc', combine='by_coords')
"""

# indicent solar radiation and cloud cover fields, each single-level
tisr = xr.open_mfdataset(f'{datadir}toa_incident_solar_radiation/*.nc', combine='by_coords')
clou = xr.open_mfdataset(f'{datadir}total_cloud_cover/*.nc', combine='by_coords')

# constants: orography, land-sea mask, soil type, lat2d and lon2d (each single-level)
cnst = xr.open_mfdataset(f'{datadir}constants/*.nc', combine='by_coords')
template = tisr.tisr
T = len(template.time.values)
dataarrays = {}
for var in [cnst.orography, cnst.lsm, cnst.slt, cnst.lat2d, cnst.lon2d]:
    values = np.tile(var.values.reshape(1,*var.values.shape), (T, 1, 1)).astype(np.float32)
    dataarrays[var.name] = xr.DataArray(values, coords=template.coords, dims=template.dims, 
                                name=var.name,indexes=template.indexes)
cnst = xr.Dataset(data_vars=dataarrays)

# geopotential and tempearture each at target pressure levels 
z500 = xr.open_mfdataset(f'{datadir}geopotential_500/*.nc', combine='by_coords')
t850 = xr.open_mfdataset(f'{datadir}temperature_850/*.nc', combine='by_coords')

var_dict = {'z': None, 't': None, 'tisr' : None, 'tcc' : None, 
            'orography' : None, 'lsm' : None, 'slt' : None, 'lat2d' : None, 'lon2d': None}
dataset_list = [z500, t850, tisr, clou, cnst]

x = xr.merge(dataset_list, compat='override', fill_value=0) # fill_value for tisr (missing first 7h of 1979) !
x = x.chunk({'time' : np.sum(x.chunks['time']), 
             'lat' : x.chunks['lat'], 'lon': x.chunks['lon']})

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

n_channels = len(dg_train.data.level.level)
print('n_channels', n_channels)

## define model

from src.train_nn_pytorch import SimpleCNN

net = SimpleCNN(filters=[64, 64, 64, 64, 2], kernels=[5, 5, 5, 5, 5], 
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
        torch.save(best_state_dict, res_dir + '9D_fccnn_3d_pytorch.pt')

    else:
        patience -= 1

    if patience == 0:
        net.load_state_dict(best_state_dict)
        break

#net_rec = Net(filters=[64, 64, 64, 64, n_channels], kernels=[5, 5, 5, 5, 5], 
#          channels=n_channels, activation=torch.nn.functional.elu)
#net_rec.load_state_dict(torch.load(res_dir + 'test_fccnn_3d_pytorch.pt'))