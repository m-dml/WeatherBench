import numpy as np
import torch
from src.pytorch.util import init_torch_device
#import torch.multiprocessing

"""
from torch.multiprocessing import Pool, Process, set_start_method, cpu_count
try:
    set_start_method('spawn')
except RuntimeError:
    pass
"""

device = init_torch_device()

datadir = '/gpfs/work/nonnenma/data/forecast_predictability/weatherbench/5_625deg/'
res_dir = '/gpfs/work/nonnenma/results/forecast_predictability/weatherbench/5_625deg/'

model_name = 'simpleResnet' # 'simpleResnet', 'tvfcnResnet50', 'cnnbn', 'Unetbn'

lead_time = 3*24
batch_size = 32

train_years = ('1979', '2015')
validation_years = ('2016', '2016')
test_years = ('2017', '2018')

var_dict = {'geopotential': ('z', [100, 200, 500, 850, 1000]),
           'temperature': ('t', [100, 200, 500, 850, 1000]),
           'u_component_of_wind': ('u', [100, 200, 500, 850, 1000]), 
           'v_component_of_wind': ('v', [100, 200, 500, 850, 1000]),
           'constants': ['lsm','orography','lat2d']
           }

"""
var_dict = {'geopotential': ('z', [500, 850]),
           'temperature': ('t', [500, 850]),
           'u_component_of_wind': ('u', [500, 850]), 
           'v_component_of_wind': ('v', [500, 850]),
           'constants': ['lsm','orography','lat2d']
           }
"""

"""
var_dict = {'geopotential': ('z', [500]),
           'temperature': ('t', [850])}
"""


target_var_dict = {'geopotential': 500, 'temperature': 850}

filters = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
kernel_sizes = [7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

past_times = [-6, -12]
verbose = True
loss_fun = 'mse'

################################################################################

import xarray as xr
import dask
from src.pytorch.Dataset import Dataset_dask, Dataset_xr

x = xr.merge(
[xr.open_mfdataset(f'{datadir}/{var}/*.nc', combine='by_coords')
 for var in var_dict.keys()],
fill_value=0  # For the 'tisr' NaNs
)
dg_train = Dataset_dask(x.sel(time=slice(train_years[0], train_years[1])), var_dict, lead_time, 
                   normalize=False, norm_subsample=1, res_dir=res_dir, train_years=train_years,
                   target_var_dict=target_var_dict, past_times=past_times, verbose=verbose)

"""
x = xr.merge(
[xr.open_mfdataset(f'{datadir}/{var}/*.nc', combine='by_coords')
 for var in var_dict.keys()],
fill_value=0  # For the 'tisr' NaNs
)
dg_validation = Dataset_xr(x.sel(time=slice(validation_years[0], validation_years[1])), var_dict, lead_time,
                        normalize=True, norm_subsample=1, res_dir=res_dir, train_years=train_years, randomize_order=False,
                        target_var_dict=target_var_dict, past_times=past_times, verbose=verbose)
dg_test =  Dataset_xr(x.sel(time=slice(test_years[0], test_years[1])), var_dict, lead_time,
                   normalize=True, mean=dg_validation.mean, std=dg_validation.std, randomize_order=False,
                   target_var_dict=target_var_dict, past_times=past_times, verbose=verbose)
"""
print('chunks', dg_train.data.chunks)

def collate_fn(batch):
    X_stack = dask.array.stack([X for X,_ in batch], axis=0).compute()
    Y_stack = dask.array.stack([y for _,y in batch], axis=0).compute()
    X_stack = torch.as_tensor(X_stack, device='cpu')
    Y_stack = torch.as_tensor(Y_stack, device='cpu')
    """
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        storage = X_stack.storage()._new_shared(X_stack.numel())
        out = X_stack.new(storage)
        return torch.stack(batch, 0, out=out)
    """
    return (X_stack, Y_stack)

num_workers = int(train_years[1]) - int(train_years[0]) + 1

train_loader = torch.utils.data.DataLoader(
    dg_train,
    pin_memory=True,
    batch_size=batch_size,
    num_workers=num_workers,
    collate_fn=collate_fn,
    drop_last=True)

################################################################################

import time
# check I/O speed on single (empty) epoch
num_steps = 1
t = time.time()
for batch in train_loader:
    if np.mod(num_steps, 1) == 0:
        print('- #, time: ', num_steps, time.time() - t)
    inputs, targets = batch[0].to(device), batch[1].to(device)
    print(inputs.shape, targets.shape)
    num_steps +=1