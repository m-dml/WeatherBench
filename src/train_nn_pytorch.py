import numpy as np
import torch
import xarray as xr


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
