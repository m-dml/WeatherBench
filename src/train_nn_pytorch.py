import numpy as np
import torch
import torch.nn.functional as F
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
                 start=None, end=None, normalize=False):
        self.ds = ds
        self.var_dict = var_dict
        self.lead_time = lead_time
        self.normalize=normalize

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
        # Normalize
        self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std
        if self.normalize:
            self.data = (self.data - self.mean) / self.std
        self.valid_time = self.data.isel(time=slice(lead_time, None)).time
        
        # According to S. Rasp, this has to go after computation of self.mean, self.std:
        if load: print('Loading data into RAM'); self.data.load()

    def __getitem__(self, index):
        'Generate one batch of data'
        idx = np.asarray(index)
        X = self.data.isel(time=idx).values
        y = self.data.isel(time=idx + self.lead_time, level=[6,20]).values
        return X, y

    # for large batch-sizes, this is orders of magnitures faster than for non-iterable Dataset()
    # __iter__() based on Example 1 in documentation of torch.utils.data.IterableDataset
    def __iter__(self):
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
        idx = torch.randperm(iter_end-iter_start).cpu() + iter_start # torch for seed control
        X = self.data.isel(time=idx).values
        y = self.data.isel(time=idx + self.lead_time, level=[6,20]).values
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
            
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

def create_predictions(model, dg):
    """Create non-iterative predictions"""
    preds = model.forward(torch.tensor(dg[np.arange(dg.__len__())][0])).detach().numpy()
    # Unnormalize
    if dg.normalize:
        preds = preds * dg.std.values[None,:,None,None] + dg.mean.values[None,:,None,None]
    das = []
    lev_idx = 0
    for var, levels in dg.var_dict.items():
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