import numpy as np
import torch
import torchvision
from .resnet import FCNResNet, CircBlock
from .cnn import SimpleCNN
from .unet import CircUNet
from .Dataset import Dataset_memmap, Dataset_xr
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


def load_data(var_dict, lead_time, train_years, validation_years, test_years, 
              target_var_dict, datadir, mmap_mode, past_times=[], verbose=False): 

    filedir = datadir + '5_625deg_all_zscored.npy'
    leveldir = datadir + '5_625deg_all_level_names.npy'

    if mmap_mode=='None' or mmap_mode is None:
        mmap_mode = None
        print('WARNING: will load entire dataset into memory!')
        print('filedir:', filedir)

    x = xr.merge( # lazy: use xr.merge to get hour counts per year from chunk size
    [xr.open_mfdataset(f'{datadir}/{var}/*.nc', combine='by_coords')
     for var in var_dict.keys()],
    fill_value=0  # For the 'tisr' NaNs
    )

    def get_year_idx(yrs):
        idx = [np.sum(x.chunks['time'][:(int(yr)-1979+i)]) for i,yr in enumerate(yrs)]
        idx = [int(idx[0] - np.min(past_times+[0])), int(idx[1] - lead_time)] # ensure valid times
        return idx

    start, end = get_year_idx(train_years)
    dg_train = Dataset_memmap(filedir=filedir, leveldir=leveldir, 
                              var_dict=var_dict, lead_time=lead_time,
                              start=start, end=end, randomize_order=True,
                              target_var_dict=target_var_dict, mmap_mode=mmap_mode,
                              dtype=np.float32, past_times=past_times, verbose=verbose)

    start, end = get_year_idx(validation_years)
    dg_validation = Dataset_memmap(filedir=dg_train.data, leveldir=dg_train.level_names, 
                              var_dict=var_dict, lead_time=lead_time,
                              start=start, end=end, randomize_order=False,
                              target_var_dict=target_var_dict, mmap_mode=mmap_mode,
                              dtype=np.float32, past_times=past_times, verbose=verbose)

    start, end = get_year_idx(test_years)
    dg_test = Dataset_memmap(filedir=dg_train.data, leveldir=dg_train.level_names, 
                              var_dict=var_dict, lead_time=lead_time,
                              start=start, end=end, randomize_order=False,
                              target_var_dict=target_var_dict, mmap_mode=mmap_mode,
                              dtype=np.float32, past_times=past_times, verbose=verbose)

    return dg_train, dg_validation, dg_test


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