import numpy as np
import torch
import xarray as xr
from src.pytorch.util import init_torch_device
from src.pytorch.Dataset import collate_fn_memmap

def create_predictions(model, dg, var_dict={'z' : None, 't' : None}, batch_size=32, verbose=False, 
                       model_forward=None, device=None, past_times_own_axis=False, mean=None, std=None,
                       return_xarray=True, dg_meta=None):
    """Create non-iterative predictions
    Base on create_predictions() function written by S. Rasp (for tensorflow v1.x): 
    https://github.com/pangeo-data/WeatherBench/blob/ced939e20da0432bc816d64c34344e72f9b4cd17/src/train_nn.py#L113    
    
    We introduce the extra var_dict argument in case dg has more fields than just 'z' and 't'.
    """

    model_forward = model.forward if model_forward is None else model_forward
    device = init_torch_device() if device is None else device

    def collate_fn(batch):
        return collate_fn_memmap(batch, dg, past_times_own_axis=past_times_own_axis)
    test_loader = torch.utils.data.DataLoader(dg, batch_size=batch_size, drop_last=False, collate_fn=collate_fn)

    model.eval() # e.g. for batch normalization layers
    preds = []
    with torch.no_grad():
        for i,batch in enumerate(test_loader):
            if verbose:
                print('batch #' + str(i))
            inputs = batch[0].to(device)
            pred = model_forward(inputs).detach().cpu().numpy()
            preds += [pred]
    preds = np.concatenate(preds, axis=0)    

    # Unnormalize
    if not mean is None and not std is None:
        preds = preds * std[None,:,None,None] + mean[None,:,None,None]

    if return_xarray:
        assert not dg_meta is None        
        das = []
        lev_idx = 0
        var_dict = dg.var_dict if var_dict is None else var_dict
        for var, levels in var_dict.items():
            if levels is None:
                das.append(xr.DataArray(
                    preds[:, lev_idx, :, :],
                    dims=['time', 'lat', 'lon'],
                    coords={'time': dg_meta['valid_time'], 'lat': dg_meta['lat'], 'lon': dg_meta['lon']},
                    name=var
                ))
                lev_idx += 1
            else:
                nlevs = len(levels)
                das.append(xr.DataArray(
                    preds[:, lev_idx:lev_idx+nlevs, :, :],
                    dims=['time', 'level', 'lat', 'lon'],
                    coords={'time': dg_meta['valid_time'], 'level': dg_meta['level'], 
                            'lat': dg_meta['lat'], 'lon': dg_meta['lon']},
                    name=var
                ))
                lev_idx += nlevs
        return xr.merge(das, compat='override')
    else:
        return preds

    
