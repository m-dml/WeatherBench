import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.resnet import ResNet
import xarray as xr

class Dataset(torch.utils.data.IterableDataset):
    r"""A class representing a :class:`Dataset`.
    
    Base on DataGenerator() object written by S. Rasp (for tensorflow v1.x): 
    https://github.com/pangeo-data/WeatherBench/blob/ced939e20da0432bc816d64c34344e72f9b4cd17/src/train_nn.py#L18
        
    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs a index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __init__(self, ds, var_dict, lead_time, mean=None, std=None, load=False, 
                 start=None, end=None, normalize=False, randomize_order=True):
        self.ds = ds
        self.var_dict = var_dict
        self.lead_time = lead_time
        self.normalize = normalize
        self.randomize_order = randomize_order

        # indexing for __getitem__ and __iter__ to find targets Z500, T850
        assert list(var_dict.keys())[:2] == ['z','t']
        var_levels = list(var_dict.items())
        if var_levels[0][1] is None and var_levels[1][1] is None:
            # single level for z and t
            self._target_idx = [0,1]
        elif len(var_levels[0][1].values)==11:
            assert var_levels[0][1].values[6] == 500.
            assert var_levels[1][1].values[9] == 850.
            self._target_idx = [6,20]
        else:
            raise NotImplementedError

        if start is None or end is None:
            start = 0
            end = self.ds.time.isel(time=slice(0, -self.lead_time)).values.shape[0]
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

        self.data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in var_dict.items():
            try:
                self.data.append(ds[var].sel(level=levels))
            except ValueError:
                self.data.append(ds[var].expand_dims({'level': generic_level}, 1))
        self.data = xr.concat(self.data, 'level')#.transpose('time', 'lat', 'lon', 'level')

        if self.normalize:
            self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
            self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std
            # for constants, compute std across space rather than time (which would be zero...)
            idx = np.where(self.std.values==0)
            if len(idx) > 0:
                self.std.values[idx] = self.data.std(('lat','lon')).mean('time').compute()[idx]
            self.data = (self.data - self.mean) / self.std

        self.valid_time = self.data.isel(time=slice(lead_time, None)).time
        
        # According to S. Rasp, this has to go after computation of self.mean, self.std:
        if load: print('Loading data into RAM'); self.data.load()

    def __getitem__(self, index):
        """ Generate one batch of data """
        idx = np.asarray(index)
        X = self.data.isel(time=idx).values
        y = self.data.isel(time=idx + self.lead_time, level=self._target_idx).values
        return X, y

    # for large batch-sizes, this is orders of magnitures faster than for non-iterable Dataset()
    # __iter__() based on Example 1 in documentation of torch.utils.data.IterableDataset
    def __iter__(self):
        """ Return iterable over data in random order """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        if self.randomize_order:
            idx = torch.randperm(iter_end-iter_start).cpu() + iter_start # torch for seed control
        else: 
            idx = torch.arange(iter_start, iter_end).cpu()
        X = self.data.isel(time=idx).values
        y = self.data.isel(time=idx + self.lead_time, level=self._target_idx).values
        return zip(X, y)
    
    def __len__(self):
        return self.data.isel(time=slice(0, -self.lead_time)).shape[0]


class PeriodicConv2D(torch.nn.Conv2d):
    """ Implementing 2D convolutional layer with mixed zero- and circular padding.
    Uses circular padding along last axis (W) and zero-padding on second-last axis (H)

    """
    
    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding_circ = ( (self.padding[0] + 1) // 2, self.padding[0] // 2, 0, 0)
            expanded_padding_zero = ( 0, 0, (self.padding[1] + 1) //2, self.padding[1] // 2 )
            return F.conv2d(F.pad(F.pad(input, expanded_padding_circ, mode='circular'), 
                                  expanded_padding_zero, mode='constant'),
                            weight, self.bias, self.stride,
                            (0,0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class SimpleCNN(torch.nn.Module):
    """ Simple CNN module for image-to-image regression
        Assumes image height and width are the same for input and output.
        Note that default constructor uses PeriodicConv2D layers !
    """

    def __init__(self, filters, kernels, channels, activation, mode='circular'):
        super(SimpleCNN, self).__init__()
        self.layers, in_ = [], channels
        self.activation = activation
        assert not np.any(kernels == 2), 'kernel size 2 not allowed for circular padding'
        in_channels = [channels] + list(filters[:-1])
        if mode=='circular':
            self.layers = torch.nn.ModuleList([PeriodicConv2D(i, f, k, padding=(k-1, k-1),
                            padding_mode='circular') for i,f,k in zip(in_channels, filters, kernels)])
        else:
            self.layers = torch.nn.ModuleList([torch.nn.Conv2d(i, f, k, padding=k//2) 
                                               for i,f,k in zip(in_channels, filters, kernels)])
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm2d(i) for i in filters[:-1]])

    def forward(self, x):
        for layer, bn in zip(self.layers[:-1], self.bns[:-1]):
            x = bn(self.activation(layer(x)))
        x = self.layers[-1](x)
        return x


class CircUNet(torch.nn.Module):
    """ Simple UNet module for image-to-image regression
        Assumes image height and width are the same for input and output.
        Note that default constructor uses PeriodicConv2D layers !
    """
    def __init__(self, in_channels, out_channels, filters, kernels, pooling, activation, mode='circular'):
        super(CircUNet, self).__init__()
        assert not np.any(kernels == 2), 'kernel size 2 not allowed for circular padding'
        ziplist = zip([in_channels] + [f[0] for f in filters[:-1]], filters, kernels)
        self.downlayers = [Downscaler(i, f, k, pooling,  activation, mode='circular') for i,f,k in ziplist]

        # bottom layer: number of filters actually has to increase
        i, f, k, o = filters[-1][-1], [2*f for f in filters[-1]], kernels[-1], filters[-1][-1]
        self.uplayers = [Upscaler(i, f, k, pooling, o, activation, mode='circular')]
        
        ziplist = zip([2*f[-1] for f in filters[::-1]], filters[::-1], kernels[::-1], [f[-1] for f in filters[::-1][1:]])
        self.uplayers += [Upscaler(i, f, k, pooling, o, activation, mode='circular') for i,f,k,o in ziplist]

        self.downlayers, self.uplayers = torch.nn.ModuleList(self.downlayers), torch.nn.ModuleList(self.uplayers)
        
        i, f, k = 2*filters[0][0], out_channels, kernels[0][-1]
        if mode=='circular':
            self.final = PeriodicConv2D(i, f, k, padding=(k-1, k-1), padding_mode='circular')
        else:
            self.final = torch.nn.Conv2d(i, f, k, padding=k//2)

    def forward(self, x):
        outs = []
        for layer in self.downlayers:
            x, out = layer(x)
            outs += [out]
        for layer, out in zip(self.uplayers, outs[::-1]):
            x = layer(x, out)
        return self.final(x)


class Downscaler(torch.nn.Module):
    def __init__(self, in_channels, filters, kernels, pooling, activation, mode='circular'):
        super(Downscaler, self).__init__()
        ziplist = zip([in_channels] + list(filters[:-1]), filters, kernels)
        if mode=='circular':
            self.layers = [PeriodicConv2D(i, f, k, padding=(k-1, k-1),
                          padding_mode='circular') for i,f,k in ziplist]
        else:
            self.layers = [torch.nn.Conv2d(i, f, k, padding=k//2) for i,f,k in ziplist]
        self.layers = torch.nn.ModuleList(self.layers)
        self.pooling = torch.nn.MaxPool2d(pooling)
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm2d(i) for i in filters])
        self.activation = activation

    def forward(self, x):
        for layer, bn in zip(self.layers, self.bns):            
            x = bn(self.activation(layer(x)))
        return self.pooling(x), x


class Upscaler(torch.nn.Module):
    def __init__(self, in_channels, filters, kernels, pooling, out_channel, activation, mode='circular'):
        super(Upscaler, self).__init__()
        ziplist = zip([in_channels] + list(filters[:-1]), filters, kernels)
        if mode=='circular':
            self.layers = [PeriodicConv2D(i, f, k, padding=(k-1, k-1),
                          padding_mode='circular') for i, f, k in ziplist]
        else:
            self.layers = [torch.nn.Conv2d(i, f, k, padding=k//2) for i,f,k in ziplist]
        self.layers = torch.nn.ModuleList(self.layers)
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm2d(i) for i in filters])
        self.uplayer = torch.nn.ConvTranspose2d(filters[-1], out_channel, pooling, stride=2)
        self.activation = activation

    def forward(self, x, xskip):
        for layer, bn in zip(self.layers, self.bns):            
            x = bn(self.activation(layer(x)))
        x = self.uplayer(x) 
        return torch.cat((x,xskip), axis=1) # Nx(C+Cskip)xHxW 


class FCNResNet(ResNet):
    """ Simple fully convolutional ResNet with variable number of blocks and layers
    Based on torchvision ResNet implementation
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L124
    """
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, in_channels=1, out_channels=1, 
                 nfilters=[64, 64, 128, 256, 512], kernel_size=3):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.inplanes = nfilters[0]
        self.conv1 = torch.nn.Conv2d(in_channels, nfilters[0], kernel_size=kernel_size, 
                                     stride=1, padding=(kernel_size-1)//2, bias=True)
        self.bn1 = norm_layer(nfilters[0])
        self.relu = torch.nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, nfilters[1], layers[0])
        if len(layers) > 2:
            self.layer2 = self._make_layer(block, nfilters[2], layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0])
        if len(layers) > 3:
            self.layer3 = self._make_layer(block, nfilters[3], layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1])
        if len(layers) > 4:
            self.layer4 = self._make_layer(block, nfilters[4], layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])

        self.final = torch.nn.Conv2d(in_channels=4*nfilters[len(layers)-1], # why 4x?
                                     out_channels=out_channels, 
                                     kernel_size=(1,1), 
                                     stride=1) 

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    torch.nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    torch.nn.init.constant_(m.bn2.weight, 0)    

    def _forward_impl(self, x):
        # See note [TorchScript super()]

        # initial conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # actual ResNet layers (multiple ResNet blocks per layer)
        x = self.layer1(x)
        if hasattr(self, 'layer2'):
            x = self.layer2(x)
        if hasattr(self, 'layer3'):
            x = self.layer3(x)
        if hasattr(self, 'layer4'):
            x = self.layer4(x)

        # final 1x1 convolution
        x = self.final(x)
        
        return x


def create_predictions(model, dg, var_dict={'z' : None, 't' : None}):
    """Create non-iterative predictions
    Base on create_predictions() function written by S. Rasp (for tensorflow v1.x): 
    https://github.com/pangeo-data/WeatherBench/blob/ced939e20da0432bc816d64c34344e72f9b4cd17/src/train_nn.py#L113    
    
    We introduce the extra var_dict argument in case dg has more fields than just 'z' and 't'.
    """

    preds = model.forward(torch.tensor(dg[np.arange(dg.__len__())][0])).detach().numpy()

    # Unnormalize
    if dg.normalize:
        idx = dg._target_idx
        preds = preds * dg.std.values[None,idx,None,None] + dg.mean.values[None,idx,None,None]

    das = []
    lev_idx = 0
    var_dict = dg.var_dict if var_dict is None else var_dict
    for var, levels in var_dict.items():
        if levels is None:
            das.append(xr.DataArray(
                preds[:, lev_idx, :, :],
                dims=['time', 'lat', 'lon'],
                coords={'time': dg.valid_time, 'lat': dg.ds.lat, 'lon': dg.ds.lon},
                name=var
            ))
            lev_idx += 1
        else:
            nlevs = len(levels)
            das.append(xr.DataArray(
                preds[:, lev_idx:lev_idx+nlevs, :, :],
                dims=['time', 'level' 'lat', 'lon'],
                coords={'time': dg.valid_time, 'level': dg.ds.level, 'lat': dg.ds.lat, 'lon': dg.ds.lon},
                name=var
            ))
            lev_idx += nlevs
    return xr.merge(das, compat='override')
