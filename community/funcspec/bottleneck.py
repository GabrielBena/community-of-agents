from turtle import pos
from attr import attrib
import torch
import numpy as np
import torch.nn as nn
import copy
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_n
import wandb
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from community.common.training import train_community
from community.common.init import init_community, init_optimizers
from community.utils.others import is_notebook
from community.utils.wandb_utils import get_wandb_artifact, mkdir_or_save_torch

# ------ Bottleneck Metric ------ : 

def readout_retrain(community, loaders, n_classes=10, deepR_params_dict={},
                    n_epochs=3, n_tests=3, train_all_param=False, force_connections=False,
                    use_tqdm=False, device=torch.device('cuda'), symbols=False) : 
    """
    Retrains the bottleneck-readout connections of each sub-network for each sub-task and stores performance.
    Args : 
        community : trained model on global task
        n_classes : number of classes of the new readout layer
        loaders : training and testing data-loaders
        n_tests : number of init and train to conduct
        n_epochs : number of epochs of training
        lrs : learning rate of training : [subnets, connections]
        train_all_params : train all sub-networks parameters as well as interconnections. If False, train only one bottleneck-readout at a time
    """

    notebook = is_notebook()
    if type(use_tqdm) is int : 
        position = use_tqdm
        use_tqdm = True
    elif use_tqdm : 
        position = 0 

    tqdm_f = tqdm_n if notebook else tqdm
    pbar = range(n_tests)
    if use_tqdm : 
        pbar = tqdm_f(pbar, position=position, desc='Bottleneck Metric Trials : ', leave=None)

    single_accs_total = []
    for test in pbar : 

        single_losses  = [[] for target in range(2)]
        single_accs = [[] for target in range(2)]
        for target in range(2) :

            f_community = copy.deepcopy(community)
            for f_agent in f_community.agents : 
                if f_agent.use_bottleneck : 
                    f_agent.readout = nn.Linear(f_agent.bottleneck.out_features, n_classes)
                else : 
                    f_agent.readout = nn.Linear(f_agent.dims[-2], n_classes)
                
                f_agent.dual_readout = False
                f_agent.to(device)
                f_community.use_common_readout = False

            for name, p in f_community.named_parameters() : 
                if 'readout' in name :#and str(agent) in name:
                    p.requires_grad = True
                else : 
                    p.requires_grad = train_all_param
        
            lr_ag, gamma = 1e-2, 0.9
            params_dict = {'lr' : lr_ag, 'gamma' : gamma}
            
            optimizers, schedulers = init_optimizers(f_community, params_dict, deepR_params_dict)

            try : 
                min_acc = community.best_acc * 0.95
            except AttributeError : 
                min_acc = None

            training_dicts = [{
                'n_epochs' : n_epochs, 
                'task' : str(target),
                'global_rewire' : False, 
                'check_gradients' : False, 
                'reg_factor' : 0.,
                'train_connections' : False,
                'decision_params' : (ts, 'both') ,
                'stopping_acc' : min_acc ,
                'early_stop' :  True,
                'deepR_params_dict' : deepR_params_dict,
                'data_type' : 'symbols' if symbols else None,
                'force_connections' : force_connections
            } for ts in ['mid', 'last'] ]

            train_outs = [train_community(f_community, *loaders, optimizers,
                        schedulers=schedulers, config=training_dict,
                        trials = (True, True), joint_training=True,
                        use_tqdm=position+1 if use_tqdm else False,
                        device=device) for training_dict in training_dicts]

            test_losses = np.array([train_out['test_losses'] for train_out in train_outs]) #before and after comms x n_agents
            test_accs = np.array([train_out['test_accs'] for train_out in train_outs]) #before and after comms x n_agents

            single_accs[target] = test_accs.max(1)
        
        single_accs_total.append(np.array(single_accs))
            
    return {'accs' : np.array(single_accs_total)}

def compute_bottleneck_metrics(p_cons, loaders, save_name, device=torch.device('cuda'), config=None) : 
    """
    Compute complete bottleneck metric, for every trained community and sparsity of interconnections
    """
    metrics = {}
    l = 0

    notebook = is_notebook()
    tqdm_f = tqdm_n if notebook else tqdm
    
    use_wandb = wandb.run is not None
    if use_wandb :
        config = wandb.config
    else : 
        assert config is not None, 'Provide configuration dict or run using WandB'

    save_path = config['saves']['metrics_save_path']
    community_state_path = config['saves']['models_save_path'] + config['saves']['models_save_name']
    agent_params_dict = config['model_params']['agents_params']

    try :  #Load states from file
        community_states = torch.load(community_state_path)
        print('Loading models from file')
        print(community_state_path)

    except FileNotFoundError: #Load from WandB artifacts
        try : 
            community_states, *_ = get_wandb_artifact(None, project='funcspec', name='state_dicts', run_id=config['resume_run_id'])
        except KeyError : 
            community_states, *_ = get_wandb_artifact(config, project='funcspec', name='state_dicts', process_config=True)
        print('Loading models from artifact')

    community = init_community(agent_params_dict, 0.1, device=device, use_deepR=config['model_params']['use_deepR'])

    for i, p_con in enumerate(tqdm_f(p_cons[l:], position=0, desc='Model Sparsity : ', leave=None)) : 

        metrics[p_con] = {}
        metrics[p_con]['losses'], metrics[p_con]['accs'] = [], []
        states = community_states[p_con]
        
        for i, state in enumerate(tqdm_f(states, position=1, desc = 'Model Trials', leave=None)) : 

            community.load_state_dict(state)
            metric = readout_retrain(community, loaders,
                                     deepR_params_dict=config['optimization']['connections'],
                                     use_tqdm=False, device=device)     
            for metric_name in ['losses', 'accs'] :
                metrics[p_con][metric_name].append(metric[metric_name])
                
        #metrics[p_con]['losses'] = np.array(metrics[p_con]['losses'])
        metrics[p_con]['accs'] = np.array(metrics[p_con]['accs'])
        
        mkdir_or_save_torch(metrics, save_name, save_path)

    figures = fig1, fig2 = plot_bottleneck_results(metrics)

    if use_wandb : 
        wandb.log_artifact(save_path + save_name, name='bottleneck', type='metric')
        wandb.log({'Bottleneck Metric' : wandb.Image(fig1), 'Bottleneck Difference Metric' : wandb.Image(fig2)})

    return metrics, figures
        

def plot_bottleneck_results(bottleneck_metric) : 

    p_cons = np.array(list(bottleneck_metric.keys()))
    l = len(p_cons)

    fig1, axs = plt.subplots(1, 2,  figsize=(20, 5))
    linestyles = ['--', '-', '-.']
    for i, (ax, metric_name) in enumerate(zip(axs, ['losses', 'accs'])) : 
        for t in range(2) : 
            for n in range(2) : 
                linestyle = linestyles[n]
                mean = np.array([bottleneck_metric[p_con][metric_name][..., t, n, -1].mean() for p_con in p_cons[:l]])
                std = np.array([bottleneck_metric[p_con][metric_name][..., t, n, -1].std() for p_con in p_cons[:l]])
                plot = ax.plot(p_cons[:l], mean, 
                        label=f'{metric_name}, Subnetwork {n}, Subtask {t}', linestyle=linestyle)
                col = plot[-1].get_color()
                ax.fill_between(p_cons[:l], mean-std, mean+std, color=col, alpha=0.2)
                for p_con in p_cons[:l] : 
                    data_points = bottleneck_metric[p_con][metric_name][..., t, n, -1].flatten()

                    ax.plot([p_con]*len(data_points), data_points, '.', color=col, alpha=0.4)
                    
            ax.legend()
            ax.set_xscale('log')
            
            ax.set_xlabel('Proportion of active connections', fontsize=15)
            ax.set_ylabel('Functional Specialization :\n Performance of subnetworks after re-training', fontsize=13)
            ax.set_title(f'Bottleneck Metric', fontsize=15)

    fig1.suptitle('Bottleneck Re-training Performance', fontsize=15)

    metrics = lambda p_con : (bottleneck_metric[p_con][metric_name][..., n, n, -1], bottleneck_metric[p_con][metric_name][..., 1-n, n, -1])
    norm_diff = lambda p_con : ((metrics(p_con)[0]-metrics(p_con)[1])/(metrics(p_con)[0]+metrics(p_con)[1]))
        
    fig2, axs = plt.subplots(1, 2,  figsize=(15, 5), sharex=False)
    linestyles = ['--', '-', '-.']
    x_axis = [p_cons, (1-p_cons)/(2*(1+p_cons))]
    x_labels = ['Proportion of active connections', 'Q Modularity Measure']
    metric_name = 'losses'
    for i, p_cons_Q in enumerate(x_axis) : 
        ax = axs[i]
        for n in range(2) : 
            linestyle = linestyles[n] 
            means = np.array([norm_diff(p_con).mean() for p_con in p_cons[:l]])
            #maxs = only_maxs[k][t]
            stds = np.array([norm_diff(p_con).std() for p_con in p_cons[:l]])
            plot = ax.plot(p_cons_Q[:l], -means, 
                    label=f'Subnetwork {n}', linestyle=linestyle)
            col = plot[-1].get_color()
            ax.fill_between(p_cons_Q[:l], -means-stds, -means+stds, color=col, alpha=0.2)

        ax.set_xlabel(x_labels[i], fontsize=15)
        ax.legend()
        ax.set_xscale('log'*(p_cons_Q is p_cons) + 'linear'*(p_cons_Q is not p_cons))
            
        ax.set_xlabel('Proportion of active connections', fontsize=15)
        if i==0 : ax.set_ylabel('Functional Specialization :\n Performance Difference', fontsize=13)
        if i==0 : ax.set_title(f'Bottleneck Metric', fontsize=15)

    fig2.supylabel('Functional Specialization', fontsize=15)
    fig2.tight_layout()

    return fig1, fig2
