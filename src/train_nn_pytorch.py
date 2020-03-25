import numpy as np
import torch
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


def create_predictions(model, dg, var_dict={'z' : None, 't' : None}, batch_size=32, verbose=False, model_forward=None):
    """Create non-iterative predictions
    Base on create_predictions() function written by S. Rasp (for tensorflow v1.x): 
    https://github.com/pangeo-data/WeatherBench/blob/ced939e20da0432bc816d64c34344e72f9b4cd17/src/train_nn.py#L113    
    
    We introduce the extra var_dict argument in case dg has more fields than just 'z' and 't'.
    """

    model_forward = model.forward if model_forward is None else model_forward
    
    #preds = model.forward(torch.tensor(dg[np.arange(dg.__len__())][0])).detach().numpy()

    test_loader = torch.utils.data.DataLoader(dg, batch_size=batch_size, drop_last=False)
    model.eval() # e.g. for batch normalization layers
    preds = []
    with torch.no_grad():
        for i,batch in enumerate(test_loader):
            if verbose:
                print('batch #' + str(i))
            inputs = batch[0]
            pred = model_forward(inputs)
            preds += [pred]
    preds = np.concatenate(preds, axis=0)    

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