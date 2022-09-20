from multiprocessing import connection
from os import mkdir
from community.utils.wandb_utils import mkdir_or_save_torch
import torch
import numpy as np
import torch.nn as nn
from torchvision import *

from community.data.datasets import get_datasets_alphabet, get_datasets_symbols
from community.funcspec.single_model_loop import init_and_train, train_and_compute_metrics
import wandb
from tqdm import tqdm

#warnings.filterwarnings('ignore')

if __name__ == "__main__": 

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    n_classes = 50

    print(f'Training for {n_classes} classes')

    symbol_config = {'data_size' : [30000, 5000],
                                'nb_steps' : 50,
                                'n_symbols' : n_classes - 1,
                                'input_size' : 50,
                                'static' : True, 
                                'symbol_type' : '0', 
                                'double_data' : False
                    }
                    
    if symbol_config['static'] :
        symbol_config['nb_steps'] = 2
        symbol_config['data_size'] = [d*2 for d in symbol_config['data_size']]

    if not symbol_config['double_data'] : 
            n_classes //= 2

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
                         'n_hid' : 25,
                         'n_layer' : 1,
                         'n_out' : dataset_config['n_classes'],
                         'train_in_out': (True, True),
                         'use_readout': True,
                         'cell_type': str(nn.RNN),
                         'use_bottleneck': False,
                         'ag_dropout': 0.0, 
                         'dual_readout' : True}
        
    lr, gamma = 1e-3, 0.95
    params_dict = {'lr' : lr, 'gamma' : gamma}

    l1, gdnoise, lr, gamma, cooling = 1e-5, 1e-3, 1e-3, 0.95, 0.95
    deepR_params_dict = {'l1' : l1, 'gdnoise' : gdnoise, 'lr' : lr, 'gamma' : gamma, 'cooling' : cooling}

    connections_params_dict = {'use_deepR' : False, 
                               'global_rewire' : False,
                               'com_dropout' : 0., 
                               'sparsity' : 0., 
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
            'decision_params' : ('last', '0'),
            'n_epochs' : 25, 
            'n_tests' : 1, 
            'inverse_task' : False, 
            'stopping_acc' : 0.95,
            'early_stop' : True

        },       
        'task' : 'none',
        'do_training' : True
    }

    n_hids = np.arange(1, 50)

    #If training for 1 agent infers 2 digits, make sure to enable dual readouts
    assert config['model_params']['agents_params']['dual_readout'] or (not (config['training']['decision_params'] == '0' and config['task'] == 'none') )
 
    # WAndB tracking : 
    wandb.init(project='funcspec', entity='gbena', config=config)
    run_dir = wandb.run.dir + '/'

    config['save_paths'] = {'training' : run_dir + 'training_results', 
                      'metrics' : run_dir + 'metric_results', 
    }
    
    wandb.config.update(config)

    #wandb.define_metric('p_connection')
    wandb.define_metric('n_hidden')

    #metric_names = ['Correlation', 'Masks', 'Bottleneck']

    metric_names = ['Correlation', 'Bottleneck']
    training_results, all_results = {}, {}

    pbar = tqdm(n_hids, position=0, leave=None)
    for p, n_hid in enumerate(pbar) :

        pbar.set_description(f'Agent Hidden neurons ({n_hid}) ') 
        config['model_params']['agents_params']['n_hid'] = n_hid

        trained_coms, train_out = init_and_train(config, loaders, device)
        training_results[n_hid] = train_out
        
    name = 'training_results'
    mkdir_or_save_torch(training_results, 'training_results', run_dir)
    artifact = wandb.Artifact(name=name, type='dict')
    artifact.add_file(run_dir + name)
    wandb.log_artifact(artifact)
