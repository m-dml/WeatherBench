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

model_name = 'simpleResnet' # 'simpleResnet', 'tvfcnResnet50', 'cnnbn', 'Unetbn'

lead_time = 3*24
batch_size = 32

var_dict = {'geopotential': ('z', [100, 200, 500, 850, 1000]),
           'temperature': ('t', [100, 200, 500, 850, 1000]),
           'u_component_of_wind': ('u', [100, 200, 500, 850, 1000]), 
           'v_component_of_wind': ('v', [100, 200, 500, 850, 1000]),
           'constants': ['lsm','orography','lat2d']
           }

x = xr.merge(
[xr.open_mfdataset(f'{datadir}/{var}/*.nc', combine='by_coords')
 for var in var_dict.keys()],
fill_value=0  # For the 'tisr' NaNs
)
x = x.chunk({'time' : np.sum(x.chunks['time']), 'lat' : x.chunks['lat'], 'lon': x.chunks['lon']})

dg_train = Dataset(x.sel(time=slice(train_years[0], train_years[1])), var_dict, lead_time, 
                   normalize=True, norm_subsample=1, res_dir=res_dir, train_years=train_years)
train_loader = torch.utils.data.DataLoader(
    dg_train,
    batch_size=batch_size,
    drop_last=True)

dg_validation =  Dataset(x.sel(time=slice('2016', '2016')), var_dict, lead_time,
                        mean=dg_train.mean, std=dg_train.std, normalize=True, randomize_order=False)
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
    from src.pytorch.cnn import SimpleCNN

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
    from src.pytorch.unet import CircUNet

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


elif model_name == 'simpleResnet':
    """
    from src.pytorch.resnet import FCNResNet
    from torchvision.models.resnet import Bottleneck
    model = FCNResNet(in_channels=n_channels,
                      out_channels=2,
                      block=Bottleneck, # basic ResNet block. 'Bottleneck' is 1x1 -> 3x3 -> 1x1 convs stacked  
                      replace_stride_with_dilation=[True, True, True], # assures stride=1 through all layers
                      layers=[4], # number of blocks per layer. len(layers) gives number of layers !
                      nfilters = [64, 64, 128, 256, 512], # number of filters per layer
                      kernel_size = 3 # kernel size for first conv layer
                     )
    """    
    from src.pytorch.resnet import FCNResNet, CircBlock
    layers = [13]
    model = FCNResNet(in_channels=n_channels,
                      out_channels=2,
                      block=CircBlock, # basic ResNet block. 'Bottleneck' is 1x1 -> 3x3 -> 1x1 convs stacked  
                      #replace_stride_with_dilation=[True, True, True], # assures stride=1 through all layers
                      layers=layers, # number of blocks per layer. len(layers) gives number of layers !
                      nfilters = [128, 128], # number of filters per layer
                      kernel_size = 7, # kernel size for first conv layer
                      dropout_rate = 0.1, 
                      padding_mode='circular'
                     )    
    def model_forward(input):
        return model.forward(input)
    
    model_name += '_' + str(2 + 2 * np.sum(layers)) # add layer count to model name

else: 
    raise NotImplementedError()

print('total #parameters: ', np.sum([np.prod(item.shape) for item in model.state_dict().values()]))
print('output shape: ', model_forward(torch.zeros((7,n_channels,32,64))).shape)
#print(model)


## train model

import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy

optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs, max_patience = 200, 20
best_loss, patience = np.inf, max_patience
best_state_dict = {}

epoch = 0
num_steps, eval_every = 0, 2000
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
        if np.mod(num_steps, eval_every) == 0:
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
from src.train_nn_pytorch import create_predictions

dg_test =  Dataset(x.sel(time=slice('2017', '2018')),
                   var_dict,
                   lead_time,
                   mean=dg_train.mean, # make sure that model was trained 
                   std=dg_train.std,   # with same data as in dg_train, 
                   normalize=True,     # or else normalization is off!
                   randomize_order=False)

preds = create_predictions(model,
                           dg_test,
                           var_dict={'z' : None, 't' : None},
                           batch_size=100,
                           model_forward=model_forward,
                           verbose=True)

from src.score import compute_weighted_rmse, load_test_data
z500_test = load_test_data(f'{datadir}geopotential_500/', 'z')
t850_test = load_test_data(f'{datadir}temperature_850/', 't')
rmse_z = compute_weighted_rmse(preds.z, z500_test.isel(time=slice(lead_time, None))).load()
rmse_t = compute_weighted_rmse(preds.t, t850_test.isel(time=slice(lead_time, None))).load()
print('RMSE z', rmse_z.values); print('RMSE t', rmse_t.values)

print('saving RMSE results to ' + res_dir + model_fn[:-3] + '_RMSE_zt.npy')
np.save(res_dir + model_fn[:-3] + '_RMSE_zt',  np.hstack((rmse_z.values, rmse_t.values)))