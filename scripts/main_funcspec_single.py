from multiprocessing import connection
from os import mkdir
from community.common.wandb_utils import mkdir_or_save_torch
import torch
import numpy as np
import torch.nn as nn
from torchvision import *

from community.data.datasets import get_datasets_alphabet, get_datasets_symbols
from community.funcspec.single_model_loop import train_and_compute_metrics
import wandb
from tqdm import tqdm

#warnings.filterwarnings('ignore')

if __name__ == "__main__": 

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    n_classes = np.random.randint(2, 6)
    print(f'Training for {n_classes} classes')

    symbol_config = {'data_size' : (30000, 5000),
                                'nb_steps' : 50,
                                'n_symbols' : n_classes - 1,
                                'symbol_size' : 5,
                                'input_size' : 30,
                                'static' : False
                    }
    if symbol_config['static'] :
        symbol_config['nb_steps'] = 2

    dataset_config = {'batch_size' : 256, 
                      'use_cuda' : use_cuda, 
                      'fix_asym' : False, 
                      'permute_dataset' : False, 
                      'seed' : None, 
                      'data_type' : 'symbols',
                      'n_classes' : n_classes,
                      'symbol_config' : symbol_config
    }
    
    
    if dataset_config['data_type'] == 'symbols' : 
        loaders, datasets = get_datasets_symbols(symbol_config,
                                       dataset_config['batch_size'],
                                       dataset_config['use_cuda'])

        dataset_config['input_size'] = symbol_config['input_size'] ** 2
    else : 
        all_loaders = get_datasets_alphabet('data/',
                                dataset_config['batch_size'],
                                dataset_config['use_cuda'],
                                dataset_config['fix_asym'],
                                dataset_config['permute_dataset'], 
                                dataset_config['seed']
                        )
        loaders = all_loaders[['multi', 'double_d',  'double_l', 'single_d' 'single_l'].index(dataset_config['data_type'])]
        dataset_config['input_size'] = 784
    
    agents_params_dict = {'n_agents' : 2,
                         'n_in' : dataset_config['input_size'],
                         'n_ins' : None,
                         'n_hid' : 50,
                         'n_layer' : 1,
                         'n_out' : dataset_config['n_classes'],
                         'train_in_out': (True, True),
                         'use_readout': True,
                         'cell_type': str(nn.RNN),
                         'use_bottleneck': False,
                         'ag_dropout': 0.0}
    
    p_masks = [0.1]
    
    lr, gamma = 1e-3, 0.95
    params_dict = {'lr' : lr, 'gamma' : gamma}

    l1, gdnoise, lr, gamma, cooling = 1e-5, 1e-3, 1e-3, 0.95, 0.95
    deepR_params_dict = {'l1' : l1, 'gdnoise' : gdnoise, 'lr' : lr, 'gamma' : gamma, 'cooling' : cooling}

    p_cons_params = (1/agents_params_dict['n_hid']**2, 0.999, 10)
    #p_cons_params = (0, 1., 10)

    connections_params_dict = {'use_deepR' : False, 
                               'global_rewire' : False,
                               'com_dropout' : 0., 
                               'sparsity' : p_cons_params[0], 
                               'binarize' : True
    }

    config = {
        'model_params' : {
            'agents_params' : agents_params_dict, 
            'connections_params' : connections_params_dict            
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
            'stopping_acc' : 0.95,
            'early_stop' : False
        },       
        'task' : 'count',
        'p_cons' : p_cons_params,
        'do_training' : True
    }

    try : 
        p_cons = (np.geomspace(p_cons_params[0], p_cons_params[1], p_cons_params[2]) * agents_params_dict['n_hid']**2).round()
    except ValueError : 
        p_cons = (np.linspace(p_cons_params[0], p_cons_params[1], p_cons_params[2]) * agents_params_dict['n_hid']**2).round()

    p_cons = np.unique(p_cons / agents_params_dict['n_hid']**2)
    com_dropouts = p_cons*config['model_params']['connections_params']['com_dropout']
    com_dropouts[0] = 0.

    #p_cons = [0.1]

    # WAndB tracking : 
    wandb.init(project='funcspec', entity='gbena', config=config)
    run_dir = wandb.run.dir + '/'

    config['save_paths'] = {'training' : run_dir + 'training_results', 
                      'metrics' : run_dir + 'metric_results', 
    }
    
    wandb.config.update(config)
        
    #metric_names = ['Correlation', 'Masks', 'Bottleneck']

    metric_names = ['Correlation', 'Bottleneck']
    metric_results = {metric : {} for metric in metric_names}
    training_results, all_results = {}, {}

    pbar = tqdm(p_cons, position=0, leave=None)
    for p, p_con in enumerate(pbar) :

        pbar.set_description(f'Community Sparsity ({p_con}) ') 
        config['model_params']['connections_params']['com_dropout'] = com_dropouts[p]

        metrics, train_out, results = train_and_compute_metrics(p_con, config, loaders, device)
        training_results[p_con] = train_out
        all_results[p_con] = results
        for metric in metric_names : 
            metric_results[metric][p_con] = metrics[metric]

    for name, file in zip(['training_results', 'metric_results', 'all_results'], [training_results, metric_results, all_results]) : 
        mkdir_or_save_torch(file, name, run_dir)
        artifact = wandb.Artifact(name=name, type='dict')
        artifact.add_file(run_dir + name)
        wandb.log_artifact(artifact)

    #wandb.log_artifact(run_dir + 'training_results', name='training_results')
    #wandb.log_artifact(run_dir + 'metric_results', name='metric_results')




