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

use_fields = 'z_t_unilevel_const' # 'z_t', 'z_t_unilevel_const'
model_name = 'tvfcnResnet50' # 'tv_fcn_resnet50', 'cnnbn', 'Unetbn'

lead_time = 3*24
batch_size = 32

"""
# geopotential and temperature each at 11 levels 
# regression target is Z500 (level index i=6) and T850 (level index 9, stacked vector index 9+11=20)
geop = xr.open_mfdataset(f'{datadir}geopotential/*.nc', combine='by_coords')
temp = xr.open_mfdataset(f'{datadir}temperature/*.nc', combine='by_coords')

# specific humidity, and wind component (u,v) volumes, each at 11 levels 
sphq = xr.open_mfdataset(f'{datadir}specific_humidity/*.nc', combine='by_coords')
cowu = xr.open_mfdataset(f'{datadir}u_component_of_wind/*.nc', combine='by_coords')
cowv = xr.open_mfdataset(f'{datadir}v_component_of_wind/*.nc', combine='by_coords')
"""

# geopotential and temperature each at target pressure levels 
z500 = xr.open_mfdataset(f'{datadir}geopotential_500/*.nc', combine='by_coords')
t850 = xr.open_mfdataset(f'{datadir}temperature_850/*.nc', combine='by_coords')

# incident solar radiation and cloud cover fields, each single-level
tisr = xr.open_mfdataset(f'{datadir}toa_incident_solar_radiation/*.nc', combine='by_coords')
clou = xr.open_mfdataset(f'{datadir}total_cloud_cover/*.nc', combine='by_coords')

# constants: orography, land-sea mask, soil type, lat2d and lon2d (each single-level)
cnst = xr.open_mfdataset(f'{datadir}constants/*.nc', combine='by_coords')
template = tisr.tisr
T = len(template.time.values)
dataarrays = {}
for var in [cnst.orography, cnst.lsm, cnst.slt, cnst.lat2d, cnst.lon2d]:
    # manipulating stride would be preferable over np.stride, but unsure if xarray accepts that
    values = np.tile(var.values.reshape(1,*var.values.shape), (T, 1, 1)).astype(np.float32)
    dataarrays[var.name] = xr.DataArray(values, coords=template.coords, dims=template.dims, 
                                name=var.name,indexes=template.indexes)
cnst = xr.Dataset(data_vars=dataarrays)

if use_fields == 'z_t_unilevel_const':
    # merging different fields into single dataset (this can take long, and a lot of RAM!)
    x = xr.merge([z500, t850, tisr, clou, cnst], compat='override', fill_value=0) # fill_value for tisr !
    x = x.chunk({'time' : np.sum(x.chunks['time']), 
                 'lat' : x.chunks['lat'], 'lon': x.chunks['lon']})

    # dictionary of used variables and their levels for Dataset() objects
    var_dict = {'z': None,          # target
                't': None,          # target
                'tisr' : None,      # extra field
                'tcc' : None,       # extra field
                'orography' : None, # constant
                'lsm' : None,       # constant
                'slt' : None,       # constant
                'lat2d' : None,     # constant
                'lon2d': None}      # constant
    
elif use_fields == 'z_t':

    # merging different fields into single dataset (this can take long, and a lot of RAM!)
    x = xr.merge([z500, t850], compat='override', fill_value=0) # fill_value for tisr !
    x = x.chunk({'time' : np.sum(x.chunks['time']), 
                 'lat' : x.chunks['lat'], 'lon': x.chunks['lon']})

    # dictionary of used variables and their levels for Dataset() objects
    var_dict = {'z': None,          # target
                't': None}          # target

else:
    raise NotImplementedError()

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

model_fn = f'{n_channels}D_fc{model_name}_{lead_time//24}d_pytorch.pt' # file name for saving/loading prediction model
print('model filename', model_fn)


## define model


if model_name == 'cnnbn':
    from src.train_nn_pytorch import SimpleCNN

    filters = [64, 64, 64, 64, 2] # last '2' for Z500, T850
    kernels = [5, 5, 5, 5, 5]
    activation = torch.nn.functional.elu
    mode='circular'

    model = SimpleCNN(filters=filters,
                      kernels=kernels,
                      channels=n_channels, 
                      activation=activation, 
                      mode=mode)

    def model_forward(input):
        return model.forward(input)

elif model_name == 'Unetbn':
    from src.train_nn_pytorch import CircUNet

    filters =  [ [32], [32], [32], [32]] 
    kernels =  [ [5],  [5], [5], [5] ]
    pooling = 2

    activation = torch.nn.functional.elu
    mode='circular'

    model = CircUNet(in_channels=n_channels,
                     filters=filters,
                     kernels=kernels,
                     pooling=pooling,
                     activation=activation, 
                     out_channels=2,
                     mode=mode)
    
    def model_forward(input):
        return model.forward(input)

elif model_name == 'tvfcnResnet50':
    import torchvision
    k = 3
    
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=False)

    # modify input layer (torchvision ResNet expects 3 input channels)
    model._modules['backbone']['conv1'] = torch.nn.Conv2d(
                                              in_channels=n_channels, out_channels=64,
                                              kernel_size=(k,k), stride=1, padding=(k+1)//2
                                              )
    # modify output layer (torchvision ResNet predicts 21 output channels)
    model._modules['classifier'][-1] = torch.nn.Conv2d(
                                              in_channels=512, out_channels=2,
                                              kernel_size=(k,k), stride=1, padding=(k+1)//2
                                              )    

    def model_forward(input):
        return model.forward(input)['out'] # because reasons...
        
else: 
    raise NotImplementedError()

print('total #parameters: ', np.sum([np.prod(item.shape) for item in model.state_dict().values()]))
print('output shape: ', model_forward(torch.zeros((7,n_channels,32,64))).shape)


## train model

import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy

optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs, max_patience = 200, 20
best_loss, patience = np.inf, max_patience
best_state_dict = {}

epoch = 0
num_steps, eval_every = 0, 10000
model.train()
while True:

    epoch += 1
    if epoch > n_epochs:
        break

    print(f'epoch #{epoch}')
    # Train for a single epoch.
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, targets = batch[0].to(device), batch[1].to(device)
        loss = F.mse_loss(model_forward(inputs), targets)
        loss.backward()
        optimizer.step()
        num_steps += 1

        # Track convergence on validation set.
        if np.mod(num_setps, eval_every) == 0:
            val_loss = 0
            with torch.no_grad():
                nb = 0
                for batch in validation_loader:
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                    val_loss += F.mse_loss(model_forward(inputs), targets)
                    nb += 1
            val_loss /= nb
            print(f'epoch #{epoch} || loss (last batch) {loss} || validation loss {val_loss}')

        if val_loss < best_loss:
            patience = max_patience
            best_loss = val_loss
            best_state_dict = deepcopy(model.state_dict()) # during early training will save every epoch
            torch.save(best_state_dict, res_dir + model_fn)

        else:
            patience -= 1

    if patience == 0:
        model.load_state_dict(best_state_dict)
        break

torch.save(best_state_dict, res_dir + model_fn) # create savefile in case we never beat initial loss...

# model evaluation
dg_test =  Dataset(x.sel(time=slice('2017', '2018')),
                   var_dict,
                   lead_time,
                   mean=dg_train.mean, # make sure that model was trained 
                   std=dg_train.std,   # with same data as in dg_train, 
                   normalize=True,     # or else normalization is off!
                   randomize_order=False)

test_loader = torch.utils.data.DataLoader(
    dg_test, batch_size=batch_size, drop_last=False
)

model.eval() # e.g. for batch normalization layerspreds = []
preds = []
with torch.no_grad():
    for i,batch in enumerate(test_loader):
        #print('batch #' + str(i))
        inputs = batch[0]
        pred = model_forward(inputs)
        preds += [pred]
idx = dg_test._target_idx
preds = np.concatenate(preds, axis=0) * dg_test.std.values[None,idx,None,None] + dg_test.mean.values[None,idx,None,None]

from src.score import compute_weighted_rmse, load_test_data
z500_test = load_test_data(f'{datadir}geopotential_500/', 'z')
t850_test = load_test_data(f'{datadir}temperature_850/', 't')
rmse_z = compute_weighted_rmse(preds[:,0,:,:], z500_test.isel(time=slice(lead_time, None))).load()
rmse_t = compute_weighted_rmse(preds[:,1,:,:], t850_test.isel(time=slice(lead_time, None))).load()
print('RMSE z', rmse_z.values); print('RMSE t', rmse_t.values)

print('saving RMSE results to ' + res_dir + model_fn[:-3] + '_RMSE_zt.npy')
np.save(res_dir + model_fn[:-3] + '_RMSE_zt',  np.hstack((rmse_z.values, rmse_t.values)))