import torch
import numpy as np
import torch.nn as nn
from torchvision import *

from community.data.datasets import get_datasets
from community.funcspec.single_model_loop import train_and_compute_metrics
import wandb
from tqdm import tqdm

#warnings.filterwarnings('ignore')

if __name__ == "__main__": 

    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset_config = {'batch_size' : 256, 
                      'use_cuda' : use_cuda, 
                      'fix_asym' : False, 
                      'permute_dataset' : False, 
                      'seed' : None, 
                      'data_type' : 'multi'
    }
    
    all_loaders = get_datasets('data/',
                                dataset_config['batch_size'],
                                dataset_config['use_cuda'],
                                dataset_config['fix_asym'],
                                dataset_config['permute_dataset'], 
                                dataset_config['seed']
                        )

    loaders = all_loaders[['multi', 'double', 'single'].index(dataset_config['data_type'])]
    
    agents_params_dict = {'n_agents' : 2,
                         'n_in' : 784,
                         'n_ins' : None,
                         'n_hid' : 50,
                         'n_layer' : 1,
                         'n_out' : 10,
                         'train_in_out': (True, False),
                         'use_readout': True,
                         'cell_type': str(nn.RNN),
                         'use_bottleneck': False,
                         'dropout': 0}

    p_masks = [0.1]
    
    lr, gamma = 1e-3, 0.9
    params_dict = {'lr' : lr, 'gamma' : gamma}

    l1, gdnoise, lr, gamma, cooling = 1e-4, 1e-4, 0.1, 0.95, 0.95
    deepR_params_dict = {'l1' : l1, 'gdnoise' : gdnoise, 'lr' : lr, 'gamma' : gamma, 'cooling' : cooling}

    p_cons_params = (1/agents_params_dict['n_hid']**2, 0.999, 5)

    config = {
        'model_params' : {
            'agents_params' : agents_params_dict, 
            'use_deepR' : False, 
            'global_rewire' : False
            }, 
        'datasets' : dataset_config,
        'optimization' : {
            'agents' : params_dict,
            'connections' : deepR_params_dict,
        }, 
        'training' : {
            'decision_params' : ('last', 'max'),
            'n_epochs' : 30, 
            'n_tests' : 1, 
            'inverse_task' : False, 
            'early_stop' : True
        },       
        'task' : 'parity_digits',
        'p_cons' : p_cons_params,
        'do_training' : True
    }

    p_cons = np.geomspace(p_cons_params[0], p_cons_params[1], p_cons_params[2]).round(4)

    # WAndB tracking : 
    wandb.init(project='funcspec', entity='gbena', config=config)
    run_dir = wandb.run.dir 

    community_save_path = run_dir + '/single/state_dicts/'
    metrics_path = run_dir + '/single/metrics/'
    community_save_name = f'Community_State_Dicts_{agents_params_dict["n_out"]}' + '_Bottleneck'*agents_params_dict['use_bottleneck']

    config['saves'] = {'models_save_path' : community_save_path, 
                      'metrics_save_path' : metrics_path, 
                      'models_save_name' : community_save_name
    }
    
    #config['resume_run_id'] = '195cgoaq' #Use trained states from previous run
    wandb.config.update(config)

    for p_con in tqdm(p_cons, desc='Community Sparsity : ', position=0, leave=None) : 
        train_and_compute_metrics(p_con, config, loaders, device)




