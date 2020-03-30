import numpy as np
import torch
import xarray as xr

def load_mean_std(res_dir, var_dict, train_years):

    mean, std, level, level_names = [], [], [], []
    for var in var_dict.keys():

        m = xr.open_mfdataset(f'{res_dir}/{var}/mean_{train_years[0]}_{train_years[1]}.nc', combine='by_coords')
        s = xr.open_mfdataset(f'{res_dir}/{var}/std_{train_years[0]}_{train_years[1]}.nc', combine='by_coords')

        short_name, lvls = ('lat2d', var_dict[var]) if var=='constants' else var_dict[var]
        idx =[]
        for lvl in lvls:
            idx.append(np.where(lvl == m.level.values)[0])
            level_names.append(lvl if var=='constants' else short_name + '_' + str(lvl))
            level.append(1 if var=='constants' else lvl)
        idx = np.concatenate(idx)
        mean.append(m[short_name].values[idx])
        std.append(s[short_name].values[idx])
    mean, std  = np.concatenate(mean), np.concatenate(std)
    return mean, std, level, level_names


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
                 start=None, end=None, normalize=False, norm_subsample=1, randomize_order=True,
                 target_vars=['geopotential', 'temperature'],
                 target_levels=[500, 850], dtype=np.float32, res_dir=None, train_years=None,
                 past_times=[]):

        self.ds = ds
        self.var_dict = var_dict
        self.lead_time = lead_time
        self.past_times = past_times
        self.normalize = normalize
        self.randomize_order = randomize_order

        # indexing for __getitem__ and __iter__ to find targets Z500, T850
        assert np.all(var in var_dict.keys() for var in target_vars)
        assert np.all(level in var_dict[var][1] for level, var in zip(target_levels, target_vars))
        
        max_input_lag = -np.min(self.past_times) if len(self.past_times) > 0 else 0
        if start is None or end is None:
            start = np.max([0, max_input_lag])
            end = self.ds.time.isel(time=slice(0, -self.lead_time)).values.shape[0]
        assert end > start, "this example code only works with end >= start"
        assert start >= max_input_lag
        self.start, self.end = start, end

        self.data = []
        self.level_names = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for long_var, params in var_dict.items():
            if long_var == 'constants':
                for var in params:
                    self.data.append(ds[var].expand_dims(
                        {'level': generic_level, 'time': ds.time}, (1, 0)
                    ).astype(dtype))
                    self.level_names.append(var)
            else:
                var, levels = params
                try:
                    self.data.append(ds[var].sel(level=levels))
                    self.level_names += [f'{var}_{level}' for level in levels]
                except ValueError:
                    self.data.append(ds[var].expand_dims({'level': generic_level}, 1))
                    self.level_names.append(var)        

        self.data = xr.concat(self.data, 'level')  # .transpose('time', 'lat', 'lon', 'level')
        self.data['level_names'] = xr.DataArray(
            self.level_names, dims=['level'], coords={'level': self.data.level})        
        self.output_idxs = range(len(self.data.level))

        if self.normalize:
            if mean is None or std is None:
                try:
                    print('Loading means and standard deviations from disk')
                    mean, std, level, level_names = load_mean_std(res_dir, var_dict, train_years)
                    assert np.all( level_names == self.level_names )
                    self.mean = xr.DataArray(mean, coords={'level': level}, dims=['level'])
                    self.std = xr.DataArray(std, coords={'level': level}, dims=['level'])
                except:
                    print('WARNING! Could not load means and stds. Computing. Can take a while !')
                    self.mean = self.data.isel(time=slice(0, None, norm_subsample)).mean(
                        ('time', 'lat', 'lon')).compute() if mean is None else mean
                    self.std = self.data.isel(time=slice(0, None, norm_subsample)).std(
                        ('time', 'lat', 'lon')).compute() if std is None else std
            else:
                self.mean, self.std = mean, std
            self.data = (self.data - self.mean) / self.std

        self.valid_time = self.data.isel(time=slice(lead_time, None)).time

        self._target_idx = []
        for level, var in zip(target_levels, target_vars):
            target_name = var_dict[var][0] + '_' + str(level)
            self._target_idx += [np.where(np.array(self.level_names) == target_name)[0][0]]

        # According to S. Rasp, this has to go after computation of self.mean, self.std:
        if load: print('Loading data into RAM'); self.data.load()

    def __getitem__(self, index):
        """ Generate one batch of data """
        assert np.min(index) >= self.start
        idx = np.asarray(index)
        X = self.data.isel(time=idx).values
        if max_input_lag > 0:
            Xl = []
            for l in self.past_times:
                Xl.append(self.data.isel(time=idx+self.past_times[l]).values)
            X = np.concatenate(Xls, axis=1) # stack past time points along channel dimension            
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
            idx = torch.randperm(iter_end - iter_start).cpu() + iter_start  # torch for seed control
        else: 
            idx = torch.arange(iter_start, iter_end).cpu()
        X = self.data.isel(time=idx).values
        if max_input_lag > 0:
            Xl = []
            for l in self.past_times:
                Xl.append(self.data.isel(time=idx+self.past_times[l]).values)
            X = np.concatenate(Xls, axis=1) # stack past time points along channel dimension        
        y = self.data.isel(time=idx + self.lead_time, level=self._target_idx).values
        return zip(X, y)
    
    def __len__(self):
        return self.data.isel(time=slice(0, -self.lead_time)).shape[0]
