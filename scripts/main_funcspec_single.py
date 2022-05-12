from os import mkdir
from community.common.wandb_utils import mkdir_or_save_torch
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
                      'fix_asym' : True, 
                      'permute_dataset' : False, 
                      'seed' : None, 
                      'data_type' : 'letters'
    }
    
    all_loaders = get_datasets('data/',
                                dataset_config['batch_size'],
                                dataset_config['use_cuda'],
                                dataset_config['fix_asym'],
                                dataset_config['permute_dataset'], 
                                dataset_config['seed']
                        )

    loaders = all_loaders[['multi', 'digits', 'letters', 'single'].index(dataset_config['data_type'])]
    
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
            'n_epochs' : 25, 
            'n_tests' : 1, 
            'inverse_task' : False, 
            'min_acc' : 0.9
        },       
        'task' : 'parity_digits',
        'p_cons' : p_cons_params,
        'do_training' : True
    }

    p_cons = np.geomspace(p_cons_params[0], p_cons_params[1], p_cons_params[2]).round(4)
    #p_cons = [0.1]

    # WAndB tracking : 
    wandb.init(project='funcspec', entity='gbena', config=config)
    run_dir = wandb.run.dir + '/'

    config['save_paths'] = {'training' : run_dir + 'training_results', 
                      'metrics' : run_dir + 'metric_results', 
    }
    
    wandb.config.update(config)
        
    metric_names = ['Correlation', 'Masks', 'Bottleneck']
    metric_results = {metric : {} for metric in metric_names}
    training_results, all_results = {}, {}

    for p_con in tqdm(p_cons, desc='Community Sparsity : ', position=0, leave=None) : 
        metrics, train_out, results = train_and_compute_metrics(p_con, config, loaders, device)
        training_results[p_con] = train_out
        all_results[p_con] = all_results
        for metric in metric_names : 
            metric_results[metric][p_con] = metrics[metric]

    for name, file in zip(['training_results', 'metric_results', 'all_results'], [training_results, metric_results, all_results]) : 
        mkdir_or_save_torch(file, name, run_dir)
        artifact = wandb.Artifact(name=name, type='dict')
        artifact.add_file(run_dir + name)
        wandb.log_artifact(artifact)

    #wandb.log_artifact(run_dir + 'training_results', name='training_results')
    #wandb.log_artifact(run_dir + 'metric_results', name='metric_results')




