import numpy as np
import torch
from src.pytorch.util import init_torch_device, load_data, named_network
from src.pytorch.Dataset import collate_fn_memmap, load_mean_std
from src.pytorch.train import train_model, loss_function
from src.pytorch.train_nn import create_predictions
from src.score import compute_weighted_rmse, load_test_data
from configargparse import ArgParser
import ast
import subprocess

import os
def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

def run_exp(exp_id, datadir, res_dir, mmap_mode, model_name, 
            lead_time, test_years, train_years, validation_years,
            loss_fun, var_dict, past_times, past_times_own_axis,
            weight_decay, batch_size, max_epochs, eval_every, max_patience,
            lr, lr_min, lr_decay, max_lr_patience, only_eval, **net_kwargs):

    device = init_torch_device()
    target_var_dict={'geopotential': 500, 'temperature': 850}
    mean, std, _, _ = load_mean_std(res_dir, {'geopotential': ['z', [500]], 'temperature': ['t', [850]]}, train_years)   

    fetch_commit = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    commit_id = fetch_commit.communicate()[0].strip().decode("utf-8")
    fetch_commit.kill()

    # load data
    dg_train, dg_validation, dg_test, dg_meta = load_data(
        var_dict=var_dict, lead_time=lead_time,
        train_years=(train_years[0], train_years[1]), 
        validation_years=(validation_years[0], validation_years[1]), 
        test_years=(test_years[0], test_years[1]),
        target_var_dict=target_var_dict, datadir=datadir, 
        mmap_mode=mmap_mode, past_times=past_times, past_times_own_axis=past_times_own_axis
    )

    def collate_fn(batch):
        return collate_fn_memmap(batch, dg_train, past_times_own_axis=past_times_own_axis)

    validation_loader = torch.utils.data.DataLoader(
        dg_validation, batch_size=batch_size, collate_fn=collate_fn, drop_last=False,
        num_workers=0 #int(train_years[1]) - int(train_years[0]) + 1
    )
    train_loader = torch.utils.data.DataLoader(
        dg_train, batch_size=batch_size, collate_fn=collate_fn, drop_last=True,
        num_workers=0 #int(train_years[1]) - int(train_years[0]) + 1
    )

    n_channels = len(dg_train._var_idx) if past_times_own_axis else len(dg_train._var_idx) * len(dg_train.past_times)
    print('n_channels', n_channels)
    model_fn = f'{exp_id}_{n_channels}D_fc{model_name}_{lead_time}h.pt'
    print('model filename', model_fn)


    ## define model
    print('net_kwargs', net_kwargs)
    model, model_forward, _ = named_network(model_name, 
                                            n_channels, 
                                            len(target_var_dict), 
                                            len(dg_train.past_times),
                                            **net_kwargs)
    print('total #parameters: ', np.sum([np.prod(item.shape) for item in model.state_dict().values()]))
    if past_times_own_axis:
        print('output shape: ', model_forward(torch.zeros((7,len(past_times)+1,n_channels,32,64))).shape)        
    else:
        print('output shape: ', model_forward(torch.zeros((7,n_channels,32,64))).shape)


    ## train model
    save_dir = res_dir + 'models/' + exp_id + '/'
    if only_eval:
        print('loading model from disk')
        model.load_state_dict(torch.load(save_dir + model_fn, map_location=torch.device(device)))
    else: # actually train

        mkdir_p(save_dir)
        print('saving model state_dict to ' + save_dir + model_fn)
        open(save_dir + commit_id + '.txt', 'w')

        loss_fun = loss_function(loss_fun, extra_args={'lat': np.load(datadir+'5_625deg_lat_values.npy')})
        training_outputs = train_model(
            model, train_loader, validation_loader, device, model_forward, loss_fun=loss_fun,
            weight_decay=weight_decay, max_epochs=max_epochs, max_patience=max_patience, 
            lr=lr, lr_min=lr_min, lr_decay=lr_decay, max_lr_patience=max_lr_patience,
            eval_every=eval_every, verbose=True, save_dir=save_dir + model_fn
        )
        print('saving full model to ' + save_dir+model_fn[:-3] + '_full_model.pt')
        torch.save(model, save_dir+model_fn[:-3] + '_full_model.pt')
        print('saving training outputs to ' + save_dir +  '_training_outputs.npy')
        np.save(save_dir + '_training_outputs', training_outputs)


    # evaluate model
    valid_test_time = dg_meta['time'].sel(time=slice(test_years[0], test_years[1]))
    dg_meta['valid_time'] = valid_test_time.isel(time=slice(dg_test.lead_time+dg_test.max_input_lag, None)).time
    preds = create_predictions(model, dg_test, var_dict={'z' : None, 't' : None}, device=device,
                               batch_size=batch_size, model_forward=model_forward, mean=mean, std=std,
                               past_times_own_axis=past_times_own_axis, verbose=True, dg_meta=dg_meta)
    z500_test = load_test_data(f'{datadir}geopotential_500/', 'z')
    t850_test = load_test_data(f'{datadir}temperature_850/', 't')
    rmse_z = compute_weighted_rmse(preds.z, z500_test.isel(time=slice(lead_time+dg_test.max_input_lag, None))).load()
    rmse_t = compute_weighted_rmse(preds.t, t850_test.isel(time=slice(lead_time+dg_test.max_input_lag, None))).load()
    print('RMSE z', rmse_z.values); print('RMSE t', rmse_t.values)

    print('saving RMSE results to ' + save_dir + model_fn[:-3] + '_RMSE_zt.npy')
    np.save(save_dir + model_fn[:-3] + '_RMSE_zt',  np.hstack((rmse_z.values, rmse_t.values)))

def setup(conf_exp=None):
    p = ArgParser()
    p.add_argument('-c', '--conf-exp', is_config_file=True, help='config file path', default=conf_exp)
    p.add_argument('--exp_id', type=str, required=True, help='experiment id')
    p.add_argument('--datadir', type=str, required=True, help='path to data')
    p.add_argument('--res_dir', type=str, required=True, help='path to results')
    p.add_argument('--mmap_mode', type=str, default='r', help='memmap data read mode')    
    p.add_argument('--only_eval', type=bool, default=False, help='if to evaulate saved model (=False for training)')

    p.add_argument('--lead_time', type=int, required=True, help='forecast lead time')
    p.add_argument('--train_years', type=str, nargs='+', default=('1979', '2015'), help='years for training')
    p.add_argument('--validation_years', type=str, nargs='+', default=('2016', '2016'), help='years for validation')
    p.add_argument('--test_years', type=str, nargs='+', default=('2017', '2018'), help='years for testing')

    p.add_argument('--var_dict', required=True, help='dictionary of fields to use for prediction')
    #p.add_argument('--target_var_dict', help='dictionary of fields to predict')
    p.add_argument('--past_times', type=int, nargs='+', default=[], help='additional time points as input')
    p.add_argument('--past_times_own_axis', type=bool, default=False, help='if additional input times are on own axis')

    p.add_argument('--layerNorm', type=str, default='BN', help='normalization layer for some network architectures')
    p.add_argument('--N_h', type=int, default=8, help='number of attention heads for self-attention architectures')
    p.add_argument('--D_h', type=int, default=8, help='output dim per attention heads for self-attention architectures')
    p.add_argument('--D_k', type=int, default=16, help='query/key dim for self-attention architectures')
    p.add_argument('--D_out', type=int, default=None, nargs='+', help='output dimensionality for self-attention architectures')
    p.add_argument('--filters_ff', type=int, default=None, nargs='+', help='filters for feed-forward convs in TransformerBlock')
    p.add_argument('--blockType', default='adding', type=str, help='type of Transformer block for convTransformer model')

    p.add_argument('--loss_fun', type=str, default='mse', help='loss function for model training')
    p.add_argument('--batch_size', type=int, default=64, help='batch-size')
    p.add_argument('--max_epochs', type=int, default=2000, help='epochs')
    p.add_argument('--max_patience', type=int, default=None, help='patience for early stopping')
    p.add_argument('--eval_every', type=int, default=None, help='frequency for checking convergence (in minibatches)')
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    p.add_argument('--lr_min', type=float, default=1e-6, help='minimal learning rate after which stop reducing')
    p.add_argument('--lr_decay', type=float, default=1., help='learning rate decay factor')
    p.add_argument('--max_lr_patience', type=int, default=None, help='patience per learning rate plateau')
        
    p.add_argument('--model_name', type=str, default='simpleResnet', help='designator for neural network')
    p.add_argument('--filters', type=int, nargs='+', required=True, help='filter count per layer or block')
    p.add_argument('--kernel_sizes', type=int, nargs='+', default=None, help='kernel sizes per layer or block')
    p.add_argument('--weight_decay', type=float, default=0., help='weight decay (L2 norm)')
    p.add_argument('--dropout_rate', type=float, default=0, help='Dropout')

    args = p.parse_args() if conf_exp is None else p.parse_args(args=[])
    args.var_dict = ast.literal_eval(args.var_dict)
    return vars(args)