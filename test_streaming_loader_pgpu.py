import numpy as np
import torch
from src.pytorch.util import init_torch_device
#import torch.multiprocessing

"""
from torch.multiprocessing import Pool, Process, set_start_method, cpu_count
try:
    set_start_method('spawn')
except RuntimeError:
    pass
"""

#device = init_torch_device()
print("NOT using CUDA !")
device = torch.device("cpu")
torch.set_default_tensor_type("torch.FloatTensor")


datadir = '/gpfs/work/nonnenma/data/forecast_predictability/weatherbench/5_625deg/'
res_dir = '/gpfs/work/nonnenma/results/forecast_predictability/weatherbench/5_625deg/'

model_name = 'simpleResnet' # 'simpleResnet', 'tvfcnResnet50', 'cnnbn', 'Unetbn'

lead_time = 3*24
batch_size = 32

train_years = ('1979', '2015')
validation_years = ('2016', '2016')
test_years = ('2017', '2018')

var_dict = {'geopotential': ('z', [100, 200, 500, 850, 1000]),
           'temperature': ('t', [100, 200, 500, 850, 1000]),
           'u_component_of_wind': ('u', [100, 200, 500, 850, 1000]), 
           'v_component_of_wind': ('v', [100, 200, 500, 850, 1000]),
           'constants': ['lsm','orography','lat2d']
           }

target_var_dict = {'geopotential': 500, 'temperature': 850}

filters = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
kernel_sizes = [7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

past_times = [-6, -12]
verbose = True
loss_fun = 'mse'

################################################################################

from src.pytorch.Dataset import Dataset_memmap, collate_fn_memmap

filedir = datadir + '5_625deg_all_zscored.npy'
leveldir = datadir + '5_625deg_all_level_names.npy'

dg_train = Dataset_memmap(filedir=filedir, leveldir=leveldir, 
                          var_dict=var_dict, lead_time=lead_time,
                          start=None, end=None, randomize_order=True,
                          target_var_dict=target_var_dict, 
                          dtype=np.float32, past_times=past_times, verbose=verbose)

def collate_fn(batch):
    return collate_fn_memmap(batch, dg_train)

train_loader = torch.utils.data.DataLoader(
    dg_train,
    collate_fn=collate_fn,
    batch_size=batch_size,
    num_workers=0,
    drop_last=True)  

################################################################################

import time

max_steps = 1000
print_every = 100

def do_dummy_epoch(train_loader, t = None):
    # check I/O speed on single (empty) epoch
    num_steps = 1
    t = time.time() if t is None else t
    for batch in train_loader:
        inputs, targets = batch[0].to(device), batch[1].to(device)
        if np.mod(num_steps, print_every) == 0 or num_steps == 1:
            print(f"- batch #{num_steps}, time: {'{0:.2f}'.format(time.time() - t)}")
            #print(inputs.shape, targets.shape)
        num_steps +=1
        if num_steps > max_steps:
            break

test_case = 'datapoint'

                  
if test_case == 'epoch':
    t = time.time()
    do_dummy_epoch(train_loader, t)
                  
elif test_case == 'buffer':
    idx = np.random.choice(350000, 100*batch_size)
    mmap = dg_train.data    # load everything at once
    print('testing loading full buffer')
    t = time.time()
    l = len(idx)
    out = mmap[(idx + dg_train._past_idx).flatten().reshape(-1,1), dg_train._var_idx, :, :]
    out = out.reshape((len(idx), -1, *out.shape[2:]))    
    print('total time: ', time.time() - t)
    print('elements loaded: ', l)
    print('output shape of individual load:', out.shape)

elif test_case == 'minibatch':
    idx = np.random.choice(350000, 100*batch_size)
    mmap = dg_train.data    # load everything at once
    print('testing loading full minibatches')
    t = time.time()
    l = 0
    for i in range(100):    
        idx_i = idx[i*batch_size:(i+1)*batch_size]
        l += len(idx_i) 
        out = mmap[(idx_i + dg_train._past_idx).flatten().reshape(-1,1), dg_train._var_idx, :, :]
        out = out.reshape((len(idx_i), -1, *out.shape[2:]))    
    print('total time: ', time.time() - t)
    print('elements loaded: ', l)
    print('output shape of individual load:', out.shape)

elif test_case == 'datapoint':
    idx = np.random.choice(350000, 100*batch_size)
    mmap = dg_train.data    # load everything at once
    print('testing loading individual time points')
    t = time.time()
    l = 0
    for i in range(3200):    
        idx_i = idx[i:(i+1)]
        l += len(idx_i) 
        out = mmap[(idx_i + dg_train._past_idx).flatten().reshape(-1,1), dg_train._var_idx, :, :]
        out = out.reshape((len(idx_i), -1, *out.shape[2:]))    
    print('total time: ', time.time() - t)
    print('elements loaded: ', l)
    print('output shape of individual load:', out.shape)