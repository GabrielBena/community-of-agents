import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision import *

from metrics.masks import compute_mask_metric
from metrics.bottleneck import compute_bottleneck_metrics
from metrics.correlation import compute_correlation_metric
from community.data.datasets import DoubleMNIST, MultiDataset
from community.common.training import compute_trained_communities

import warnings

import wandb

#warnings.filterwarnings('ignore')

if __name__ == "__main__": 

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    batch_size = 256
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    kwargs = train_kwargs, test_kwargs
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    fix_asym = False
    permute = True

    root = os.path.expanduser("~/Code/ANNs/Data/MNIST")
    seed = np.random.randint(1000)

    single_datasets = [datasets.MNIST(root, train=t, transform=transform) for t in [True, False]]
    single_datasets_fashion = [datasets.FashionMNIST(root, train=t, transform=transform, download=True) for t in [True, False]]
    
    double_datasets = [DoubleMNIST(root, fix_asym=fix_asym, permute=permute, train=t, seed=seed) for t in [True, False]]
    multi_datasets = [MultiDataset([s1, s2]) for (s1, s2) in zip(single_datasets, single_datasets_fashion)]

    single_loaders = [torch.utils.data.DataLoader(d, **k) for d, k in zip(single_datasets, kwargs)]
    double_loaders = [torch.utils.data.DataLoader(d, **k) for d, k in zip(double_datasets, kwargs)]
    multi_loaders = [torch.utils.data.DataLoader(d, **k) for d, k in zip(multi_datasets, kwargs)]

    for l in double_loaders : l.n_characters = 10
    for l in multi_loaders : l.n_characters = 10

    dataset_config = {'batch_size' : batch_size, 
                      'use_cuda' : use_cuda, 
                      'fix_asym' : fix_asym, 
                      'permute_dataset' : permute, 
                      'seed' : seed, 
                      'data_type' : 'multi'
    }

    loaders = [single_loaders, double_loaders, multi_loaders][['single', 'double', 'multi'].index(dataset_config['data_type'])]
    
    agents_params_dict = {'n_agents' : 2,
                         'n_in' : 784,
                         'n_ins' : None,
                         'n_hid' : 100,
                         'n_layer' : 1,
                         'n_out' : 10,
                         'train_in_out': (True, False),
                         'use_readout': True,
                         'cell_type': str(nn.RNN),
                         'use_bottleneck': False,
                         'dropout': 0}


    p_cons_params = (1/agents_params_dict['n_hid']**2, 0.999, 10)
    p_cons = np.geomspace(p_cons_params[0], p_cons_params[1], p_cons_params[2]).round(4)
    p_masks = [0.1]
    
    lr, gamma = 1e-3, 0.9
    params_dict = {'lr' : lr, 'gamma' : gamma}

    l1, gdnoise, lr, gamma, cooling = 1e-4, 1e-4, 0.1, 0.95, 0.95
    deepR_params_dict = {'l1' : l1, 'gdnoise' : gdnoise, 'lr' : lr, 'gamma' : gamma, 'cooling' : cooling}

    config = {
        'model_params' : {
            'agents_params' : agents_params_dict, 
            'use_deepR' : True, 
            'global_rewire' : False
            }, 
        'datasets' : dataset_config,
        'optimization' : {
            'agents' : params_dict,
            'connections' : deepR_params_dict,
        }, 
        'training' : {
            'decision_params' : ('last', 'max'),
            'n_epochs' : 15, 
            'n_tests' : 10, 
            'inverse_task' : False, 
            'early_stop' : True
        },       
        'task' : 'rotation_conflict_4',
        'p_cons_params' : p_cons_params
    }

    # WAndB tracking : 
    wandb.init(project='Spec_vs_Sparsity', entity='gbena', config=config)
    wandb.run.log_code(".")
    run_dir = wandb.run.dir 

    community_save_path = run_dir + '/state_dicts/'
    metrics_path = run_dir + '/metrics/'
    community_save_name = f'Community_State_Dicts_{agents_params_dict["n_out"]}' + '_Bottleneck'*agents_params_dict['use_bottleneck']

    config['saves'] = {'models_save_path' : community_save_path, 
                      'metrics_save_path' : metrics_path, 
                      'models_save_name' : community_save_name
    }
    wandb.config.update(config)

    #wandb.config.update({ 'model_save_path' : community_state_path, 'metric_save_path' : metrics_pat
    #"""
    #compute_trained_communities(p_cons, loaders, config=config)
    #"""    
    #compute_bottleneck_metrics(p_cons, community_state_path, model_params, double_loaders, runs_path+'Bottleneck Metric', device=device)
    
    #compute_mask_metric(p_cons, community_state_path, model_params, double_loaders, p_masks, lr=0.1, save_name=runs_path+'Weight Mask Metric', device=device)
    
    #compute_correlation_metric(p_cons, loaders, save_name='Correlations', device=device, config=config)
