import numpy as np
import xarray as xr
import torch
from src.pytorch.Dataset import Dataset
from src.pytorch.util import init_torch_device

device = init_torch_device()

datadir = '/gpfs/work/nonnenma/data/forecast_predictability/weatherbench/5_625deg/'
res_dir = '/gpfs/work/nonnenma/results/forecast_predictability/weatherbench/5_625deg/'

model_name = 'simpleResnet' # 'simpleResnet', 'tvfcnResnet50', 'cnnbn', 'Unetbn'

lead_time = 3*24
batch_size = 32

train_years = ('1979', '2015')

var_dict = {'geopotential': ('z', [100, 200, 500, 850, 1000]),
           'temperature': ('t', [100, 200, 500, 850, 1000]),
           'u_component_of_wind': ('u', [100, 200, 500, 850, 1000]), 
           'v_component_of_wind': ('v', [100, 200, 500, 850, 1000]),
           'constants': ['lsm','orography','lat2d']
           }
target_vars = ['geopotential', 'temperature']
target_levels = [500, 850]

x = xr.merge(
[xr.open_mfdataset(f'{datadir}/{var}/*.nc', combine='by_coords')
 for var in var_dict.keys()],
fill_value=0  # For the 'tisr' NaNs
)
x = x.chunk({'time' : np.sum(x.chunks['time']), 'lat' : x.chunks['lat'], 'lon': x.chunks['lon']})

dg_train = Dataset(x.sel(time=slice(train_years[0], train_years[1])), var_dict, lead_time, 
                   normalize=True, norm_subsample=1, res_dir=res_dir, train_years=train_years,
                   target_vars=target_vars, target_levels=target_levels)

train_loader = torch.utils.data.DataLoader(
    dg_train,
    batch_size=batch_size,
    drop_last=True)

dg_validation =  Dataset(x.sel(time=slice('2016', '2016')), var_dict, lead_time,
                        mean=dg_train.mean, std=dg_train.std, normalize=True, randomize_order=False,
                        target_vars=target_vars, target_levels=target_levels)
validation_loader = torch.utils.data.DataLoader(
    dg_validation,
    batch_size=batch_size,
    drop_last=False)

n_channels = len(dg_train.data.level.level)
print('n_channels', n_channels)

model_fn = f'{n_channels}D_fc{model_name}_{lead_time//24}d_pytorch_lrdecay_weightdecay_normed_test2.pt' # file name for saving/loading prediction model
print('model filename', model_fn)


## define model
from src.pytorch.util import named_network

model, model_forward = named_network(model_name, n_channels, len(target_vars))

print('total #parameters: ', np.sum([np.prod(item.shape) for item in model.state_dict().values()]))
print('output shape: ', model_forward(torch.zeros((7,n_channels,32,64))).shape)


## train model
from src.pytorch.train import train_model

lr=5e-4
lr_min=1e-6
lr_decay=0.2
weight_decay=1e-5
max_lr_patience=5
max_patience=20
eval_every=2000
max_epochs=200
max_patience=20
    
training_outputs = train_model(model, train_loader, validation_loader, device, model_forward,
                weight_decay=weight_decay, max_epochs=max_epochs, max_patience=max_patience, 
                lr=lr, lr_min=lr_min, lr_decay=lr_decay, max_lr_patience=max_lr_patience, eval_every=eval_every,
                verbose=True, save_dir=res_dir + model_fn)


# evaluate model
from src.pytorch.train_nn import create_predictions
from src.score import compute_weighted_rmse, load_test_data

dg_test =  Dataset(x.sel(time=slice('2017', '2018')),
                   var_dict,
                   lead_time,
                   mean=dg_train.mean, # make sure that model was trained 
                   std=dg_train.std,   # with same data as in dg_train, 
                   normalize=True,     # or else normalization is off!
                   randomize_order=False,
                   target_vars=target_vars, 
                   target_levels=target_levels)

preds = create_predictions(model,
                           dg_test,
                           var_dict={'z' : None, 't' : None},
                           batch_size=100,
                           model_forward=model_forward,
                           verbose=True)

z500_test = load_test_data(f'{datadir}geopotential_500/', 'z')
t850_test = load_test_data(f'{datadir}temperature_850/', 't')
rmse_z = compute_weighted_rmse(preds.z, z500_test.isel(time=slice(lead_time, None))).load()
rmse_t = compute_weighted_rmse(preds.t, t850_test.isel(time=slice(lead_time, None))).load()
print('RMSE z', rmse_z.values); print('RMSE t', rmse_t.values)

print('saving RMSE results to ' + res_dir + model_fn[:-3] + '_RMSE_zt.npy')
np.save(res_dir + model_fn[:-3] + '_RMSE_zt',  np.hstack((rmse_z.values, rmse_t.values)))