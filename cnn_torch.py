#imports
import os
import numpy as np
import xarray as xr
import torch
from src.pytorch.util import named_network, init_torch_device
from src.pytorch.Dataset import Dataset
from src.pytorch.train import train_model
from torch.utils.data import RandomSampler


#directories
datadir = '/gpfs/work/nonnenma/data/forecast_predictability/weatherbench/5_625deg/'
results_dir = '/gpfs/work/nonnenma/results/forecast_predictability/weatherbench/5_625deg/'
assert os.path.exists(datadir) and os.path.exists(results_dir)

# define task
var_dict = {'geopotential': ('z', [500])}  # input variables/levels
if False:
    var_dict =  {'geopotential': ('z', [100, 200, 500, 850, 1000]),
                 'temperature': ('t', [100, 200, 500, 850, 1000]),
                 'u_component_of_wind': ('u', [100, 200, 500, 850, 1000]),
                 'v_component_of_wind': ('v', [100, 200, 500, 850, 1000]),
                 'constants': ['lsm','orography','lat2d']}

target_vars = ['geopotential']  # output variables
target_levels = [500]  # output levels

lead_time = 3 * 24  # hours in the future for predictions

train_years = ('1979', '2015')
valid_years = ('2016', '2016')
test_years = ('2017', '2018')


#architecture etc.
model_name = 'simpleResnet' # 'simpleResnet', 'tvfcnResnet50', 'cnnbn', 'Unetbn'
batch_size = 32
max_epochs = 5
max_patience = 20

#other options
test_batch_speed = False
train_again = False  # if False and saved trained network exists from a previous run, load instead of training

# check which device to use
device = init_torch_device()


#load data
ds = xr.merge([xr.open_mfdataset(f'{datadir}/{var}/*.nc', combine='by_coords')
               for var in var_dict.keys()],
              fill_value=0)  # For the 'tisr' NaNs
ds = ds.chunk({'time' : np.sum(ds.chunks['time']), 'lat' : ds.chunks['lat'], 'lon': ds.chunks['lon']})

# specify year ranges
ds_train, ds_valid, ds_test = ds.sel(time=slice(*train_years)), ds.sel(time=slice(*valid_years)), ds.sel(time=slice(*test_years))

# wrap in Dataset object for batching etc.
args_D = [var_dict, lead_time]
kwargs_D = dict(target_vars=target_vars, target_levels=target_levels, normalize=True)

D_train = Dataset(ds_train, *args_D, norm_subsample=30000, randomize_order=True, **kwargs_D)
kwargs_D.update(dict(mean=D_train.mean, std=D_train.std))
D_valid = Dataset(ds_valid, *args_D, randomize_order=False, **kwargs_D)
D_test = Dataset(ds_valid, *args_D,  randomize_order=False, **kwargs_D)


train_loader = torch.utils.data.DataLoader(D_train, batch_size=batch_size, drop_last=True)
validation_loader = torch.utils.data.DataLoader(D_valid, batch_size=batch_size, drop_last=False)


n_channels = len(D_train.data.level.level)
print(f'{n_channels} total input channels')
base_filename = f'{n_channels}D_fc{model_name}_{lead_time//24}d_pytorch' # file name for saving/loading prediction model
model_filename = base_filename + '.pt' # file name for saving/loading prediction model
training_outputs_filename = base_filename + '_training.npy'


# build the network
model, model_forward = named_network(model_name, n_channels, len(target_vars))


if test_batch_speed:
    from time import time
    t0 = time()
    for batch in train_loader:
        inputs, targets = batch[0].to(device), batch[1].to(device)
    print(f'{time() - t0} seconds per epoch')

if train_again or not os.path.exists(training_outputs_filename):

    training_outputs = train_model(model, train_loader, validation_loader, device, model_forward,
                                   max_epochs=max_epochs, max_patience=max_patience)
    torch.save(model.state_dict(), results_dir + model_filename)
    np.save(training_outputs, training_outputs_filename)

else:  # load model from disk

    model.load_state_dict(torch.load(state_dict_file, map_location=torch.device(device)))
    training_outputs = np.load(training_outputs_filename)