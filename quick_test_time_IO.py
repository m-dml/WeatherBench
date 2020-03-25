import numpy as np
import xarray as xr
import torch
from src.train_nn_pytorch import Dataset
import time

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
var_dict = {'geopotential': ('z', [1, 10, 100, 200, 300, 400, 500, 600, 700, 850, 1000]),
           'temperature': ('t', [1, 10, 100, 200, 300, 400, 500, 600, 700, 850, 1000]),
           'u_component_of_wind': ('u', [1, 10, 100, 200, 300, 400, 500, 600, 700, 850, 1000]), 
           'v_component_of_wind': ('v', [1, 10, 100, 200, 300, 400, 500, 600, 700, 850, 1000]),
           'constants': ['lsm','orography','lat2d']
           }
"""
var_dict = {'geopotential': ('z', [500]),
           'temperature': ('t', [850])
           }

x = xr.merge(
[xr.open_mfdataset(f'{datadir}/{var}/*.nc', combine='by_coords')
 for var in var_dict.keys()],
fill_value=0  # For the 'tisr' NaNs
)
x = x.chunk({'time' : np.sum(x.chunks['time']), 'lat' : x.chunks['lat'], 'lon': x.chunks['lon']})

dg_train = Dataset(x.sel(time=slice('1979', '2015')), var_dict, lead_time, normalize=True, norm_subsample=30000)
train_loader = torch.utils.data.DataLoader(
    dg_train,
    batch_size=batch_size,
    drop_last=True)

n_channels = len(dg_train.data.level.level)
print('n_channels', n_channels)

## 

import torch.nn.functional as F

n_epochs, max_patience = 200, 20
best_loss, patience = np.inf, max_patience
best_state_dict = {}

epoch = 0
num_steps, eval_every = 0, 2000

while True:

    epoch += 1
    if epoch > n_epochs:
        break

    t = time.time()
    print(f'epoch #{epoch}')
    # Train for a single epoch.
    for batch in train_loader:
        inputs, targets = batch[0].to(device), batch[1].to(device)
        tmp1 = torch.sum(inputs) + torch.sum(targets) # just some simple computation to ensure data was loaded
        num_steps += 1

        # Track convergence on validation set.
        if np.mod(num_steps, eval_every) == 0:
            print(f'epoch #{epoch} || time {time.time() -t}s')

