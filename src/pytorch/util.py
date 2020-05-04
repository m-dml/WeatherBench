import numpy as np
import torch
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
              target_var_dict, datadir, mmap_mode, past_times=[], past_times_own_axis=False,
              verbose=False): 

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
    dg_meta = { # meta information on axes for re-construction of xarrays later on
       'lat' : x.lat,
       'lon' : x.lon,
       'time' : x.time,
       'level' : x.level
    }

    def get_year_idx(yrs):
        idx = [np.sum(x.chunks['time'][:(int(yr)-1979+i)]) for i,yr in enumerate(yrs)]
        idx = [int(idx[0] - np.min(past_times+[0])), int(idx[1] - lead_time)] # ensure valid times
        return idx

    start, end = get_year_idx(train_years)
    dg_train = Dataset_memmap(filedir=filedir, leveldir=leveldir, 
                              var_dict=var_dict, lead_time=lead_time,
                              start=start, end=end, randomize_order=True,
                              target_var_dict=target_var_dict, mmap_mode=mmap_mode,
                              dtype=np.float32, past_times=past_times, 
                              past_times_own_axis=past_times_own_axis, verbose=verbose)

    start, end = get_year_idx(validation_years)
    dg_validation = Dataset_memmap(filedir=dg_train.data, leveldir=dg_train.level_names, 
                              var_dict=var_dict, lead_time=lead_time,
                              start=start, end=end, randomize_order=False,
                              target_var_dict=target_var_dict, mmap_mode=mmap_mode,
                              dtype=np.float32, past_times=past_times, 
                              past_times_own_axis=past_times_own_axis, verbose=verbose)

    start, end = get_year_idx(test_years)
    dg_test = Dataset_memmap(filedir=dg_train.data, leveldir=dg_train.level_names, 
                              var_dict=var_dict, lead_time=lead_time,
                              start=start, end=end, randomize_order=False,
                              target_var_dict=target_var_dict, mmap_mode=mmap_mode,
                              dtype=np.float32, past_times=past_times, 
                              past_times_own_axis=past_times_own_axis, verbose=verbose)

    return dg_train, dg_validation, dg_test, dg_meta


def named_network(model_name, n_input_channels, n_output_channels, **kwargs):
    if model_name == 'cnnbn':

        from .cnn import SimpleCNN

        model = SimpleCNN(filters=[64, 64, 64, 64, n_output_channels],  # last '2' for Z500, T850
                          kernels=[5, 5, 5, 5, 5],
                          channels=n_input_channels,
                          activation=torch.nn.functional.elu,
                          mode='circular')

        def model_forward(input):
            return model.forward(input)

        past_times_own_axis = False

    elif model_name == 'Unetbn':

        from .unet import CircUNet

        model = CircUNet(in_channels=n_input_channels,
                         filters=[[32], [32], [32], [32]],
                         kernels=[[5], [5], [5], [5]],
                         pooling=2,
                         activation=torch.nn.functional.elu,
                         out_channels=n_output_channels,
                         mode='circular')

        def model_forward(input):
            return model.forward(input)

        past_times_own_axis = False

    elif model_name == 'tvfcnResnet50':

        import torchvision
        
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

        past_times_own_axis = False

    elif model_name == 'simpleResnet':

        from .resnet import FCNResNet, CircBlock

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

        past_times_own_axis = False

    elif model_name == 'ConvLSTM':
        
        from .convlstm import CircConvLSTM

        num_layers = len(kwargs['kernel_sizes'])
        kernel_size = [(i,i) for i in kwargs['kernel_sizes']]
        hidden_dim = kwargs['filters']
        kwargs = { # note we're overwriting the kwargs argument !
            'num_layers' : num_layers,
            'kernel_size' : kernel_size,
            'hidden_dim' : hidden_dim
        }
        
        assert kwargs['hidden_dim'][-1] == n_output_channels, 'final hidden dim is overall output dim of network!'
        
        model = CircConvLSTM(input_dim=n_input_channels,
                             padding_mode='circular',                             
                             **kwargs
                            )

        def model_forward(input):
            _, last_states = model.forward(input)
            return last_states[-1][0] # [0] for final hidden state h, [1] for final memory state c

        past_times_own_axis = True


    elif model_name == 'ConvTransformer':

        from src.pytorch.transformers import ConvTransformer

        model = ConvTransformer(
                         in_channels=n_input_channels,
                         out_channels=n_output_channels,
                         filters=kwargs['filters'],
                         kernel_sizes=[(i,i) for i in kwargs['kernel_sizes']], 
                         N_h=8,
                         D_h=8,
                         D_k=16,
                         D_out=None, 
                         sa_kernel_sizes=None,
                         bias=True, 
                         attention_bias=True, 
                         LayerNorm=torch.nn.LayerNorm,
                         padding_mode='circular', 
                         dropout=0., 
                         activation="relu")

        def model_forward(input):
            batch_shape = input.shape
            out =  model.forward(input.reshape((input.shape[0], -1, *input.shape[3:])))
            return out.reshape(batch_shape[0], -1, *batch_shape[3:])

        past_times_own_axis = True

    else:
        raise NotImplementedError()

    return model, model_forward, past_times_own_axis