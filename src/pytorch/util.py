import numpy as np
import torch
import torchvision
from .resnet import FCNResNet, CircBlock
from .cnn import SimpleCNN
from .unet import CircUNet
from .Dataset import Dataset
import xarray as xr


def init_torch_device():
    if torch.cuda.is_available():
        print('using CUDA !')
        device = torch.device("cuda")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        print("CUDA not available")
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")
    return device


def load_data(var_dict, lead_time, batch_size, train_years, validation_years, 
              target_var_dict, datadir, res_dir=None): 

    x = xr.merge(
    [xr.open_mfdataset(f'{datadir}/{var}/*.nc', combine='by_coords')
     for var in var_dict.keys()],
    fill_value=0  # For the 'tisr' NaNs
    )
    x = x.chunk({'time' : np.sum(x.chunks['time']), 'lat' : x.chunks['lat'], 'lon': x.chunks['lon']})

    dg_train = Dataset(x.sel(time=slice(train_years[0], train_years[1])), var_dict, lead_time, 
                       normalize=True, norm_subsample=1, res_dir=res_dir, train_years=train_years,
                       target_var_dict=target_var_dict)

    train_loader = torch.utils.data.DataLoader(
        dg_train,
        batch_size=batch_size,
        drop_last=True)

    dg_validation =  Dataset(x.sel(time=slice(validation_years[0], validation_years[1])), var_dict, lead_time,
                            normalize=True, mean=dg_train.mean, std=dg_train.std, randomize_order=False,
                            target_var_dict=target_var_dict)
    validation_loader = torch.utils.data.DataLoader(
        dg_validation,
        batch_size=batch_size,
        drop_last=False)

    return train_loader, validation_loader, dg_train, dg_validation


def named_network(model_name, n_input_channels, n_output_channels, **kwargs):
    if model_name == 'cnnbn':

        model = SimpleCNN(filters=[64, 64, 64, 64, n_output_channels],  # last '2' for Z500, T850
                          kernels=[5, 5, 5, 5, 5],
                          channels=n_input_channels,
                          activation=torch.nn.functional.elu,
                          mode='circular')

        def model_forward(input):
            return model.forward(input)

    elif model_name == 'Unetbn':

        model = CircUNet(in_channels=n_input_channels,
                         filters=[[32], [32], [32], [32]],
                         kernels=[[5], [5], [5], [5]],
                         pooling=2,
                         activation=torch.nn.functional.elu,
                         out_channels=n_output_channels,
                         mode='circular')

        def model_forward(input):
            return model.forward(input)

    elif model_name == 'tvfcnResnet50':

        k = 3

        model = torchvision.models.segmentation.fcn_resnet50(pretrained=False)

        # modify input layer (torchvision ResNet expects 3 input channels)
        model._modules['backbone']['conv1'] = torch.nn.Conv2d(
            in_channels=n_input_channels, out_channels=64,
            kernel_size=(k, k), stride=1, padding=(k + 1) // 2
        )
        # modify output layer (torchvision ResNet predicts 21 output channels)
        model._modules['classifier'][-1] = torch.nn.Conv2d(
            in_channels=512, out_channels=n_output_channels,
            kernel_size=(k, k), stride=1, padding=(k + 1) // 2
        )

        def model_forward(input):
            return model.forward(input)['out']  # because reasons...

    elif model_name == 'simpleResnet':

        """
        from src.pytorch.resnet import FCNResNet
        from torchvision.models.resnet import Bottleneck
        model = FCNResNet(in_channels=n_channels,
                          out_channels=n_output_channels
                          block=Bottleneck, # basic ResNet block. 'Bottleneck' is 1x1 -> 3x3 -> 1x1 convs stacked  
                          replace_stride_with_dilation=[True, True, True], # assures stride=1 through all layers
                          layers=[4], # number of blocks per layer. len(layers) gives number of layers !
                          nfilters = [64, 64, 128, 256, 512], # number of filters per layer
                          kernel_size = 3 # kernel size for first conv layer
                         )
        """

        model = FCNResNet(in_channels=n_input_channels,
                          out_channels=n_output_channels,
                          block=CircBlock,  # basic ResNet block. 'Bottleneck' is 1x1 -> 3x3 -> 1x1 convs stacked
                          padding_mode='circular',
                          **kwargs
                          )

        def model_forward(input):
            return model.forward(input)

    else:
        raise NotImplementedError()

    return model, model_forward