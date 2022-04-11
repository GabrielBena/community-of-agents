from scipy import rand
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm.notebook import tqdm as tqdm_n
from tqdm import tqdm
import os 
import matplotlib.pyplot as plt
import wandb

from data.wandb_utils import get_wandb_artifact, mkdir_or_save_torch
from data.datasets_and_tasks import DoubleMNIST, get_digits, temporal_data, rotation_conflict_task

from models.community import init_community
from scipy.stats import pearsonr

def fixed_digit(data, i=0) : 
    # Return a modified version of data sample, where one digit is fixed
    fixed = data[:, i, 0, :]
    fixed = torch.repeat_interleave(fixed.unsqueeze(1), data.shape[-2], dim=1)
    datas = [fixed, data[:, i-1, ...]]
    if i == 1 : 
        datas = datas[::-1]
    new_data = torch.stack(datas, axis=1)
    
    return new_data

def fixed_information_data(data, target, fixed_mode='label', n_categories=10, i=0) : 
    # Return a modified version of data sample, where one quality is fixed (digit label, or parity, etc)
    digits = get_digits(target, n_classes=n_categories)
    if fixed_mode is not None : 
        if fixed_mode == 'label':
            d_idxs = [torch.where(digits[i] == d)[0] for d in range(10)]
        elif fixed_mode == 'parity' : 
            d_idxs = [torch.where(digits[i]%2 == p)[0] for p in range(2)]
            
        datas = [[data[:, j, idx, :] for idx in d_idxs] for j in range(2)]
        new_data = [torch.stack([d1, d2], axis=1) for d1, d2 in zip(*datas)]
        return new_data
    else : 
        return fixed_digit(data, i=i)

def fixed_rotation_data(data, digits, n_angles=4, reshape=None, i=0) : 
    if reshape is None : 
        reshape = data.shape[-1] == 784
    if reshape : 
        data = data[0].reshape(*data[0].shape[:2][::-1], 28, 28)
    data, target, angle_values = rotation_conflict_task(data, digits, n_angles)
    data = temporal_data(data)
    possible_angles = np.unique(angle_values.cpu())
    d_idxs =  [torch.where(angle_values[i] == a)[0] for a in possible_angles]
    datas = [[data[:, j, idx, :] for idx in d_idxs] for j in range(2)]
    new_data = [torch.stack([d1, d2], axis=1) for d1, d2 in zip(*datas)]
    return new_data


def pearsonr_torch(x, y):
    """
    Mimics `scipy.stats.pearsonr`
    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor
    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

def pearsonr_numpy(x, y) : 
    """
    Mimics `scipy.stats.pearsonr`
    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor
    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y
    """
    print(x.shape, y.shape)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    xm = x - mean_x
    ym = y - mean_y
    r_num = np.dot(xm, ym)
    print(xm, ym.shape)
    r_den = np.linalg.norm(xm, 2) * np.linalg.norm(ym, 2)
    r_val = r_num / r_den
    return r_val
    
#v_pearsonr = vmap(pearsonr)
v_pearsonr = np.vectorize(pearsonr, signature='(n),(n)->(),()')
#v_pearsonr = np.vectorize(pearsonr_numpy, )

def randperm_no_fixed(n) : 
    perm = torch.randperm(n)

    if (torch.arange(n) == perm ).any()  and n>4:
        return randperm_no_fixed(n)
    else : 
        return perm

def get_pearson_metrics(community, loaders, n_tests=32, fixed_mode='label', use_tqdm=False, device=torch.device('cuda')) : 
    
    n_samples = 128
    correlations = [[[] for _ in range(2)] for _ in range(2)]
    double_train_loader, double_test_loader = loaders
    
    if type(use_tqdm) is int : 
        position = use_tqdm
        use_tqdm = True

    elif use_tqdm : 
        position = 0    
    pbar = range(n_tests)
    if use_tqdm : 
        pbar = tqdm(pbar, position=position, desc='Metric Trials', leave=None)
    for test in pbar :
        data, target = next(iter(double_test_loader))
        if type(data) is list : 
            data = torch.stack(data)
        data, target = temporal_data(data, 5).to(device), target.to(device)
        batch_size = target.shape[0]
        for n in range(2) : 
            for k in range(2) : 
                if not 'rotation' in fixed_mode : 
                    datas = fixed_information_data(data, target, fixed_mode, 10, k)
                else : 
                    try : 
                        n_angles = int(fixed_mode.split('_')[-1])
                    except ValueError : 
                        n_angles = 4
                    datas = fixed_rotation_data(data, target, n_angles, reshape=True, i=k)

                if type(datas) is not list : 
                    datas = [datas]
                
                corrs = []
                for d in datas : 
                    _, states = community(d)
                    agent_states = states[-1][n][0]
                    perm = randperm_no_fixed(agent_states.shape[0])
                    agent_states = agent_states.detach().cpu().numpy()
                    cor = v_pearsonr(agent_states, agent_states[perm])[0]
                    #corrs.append(cor.cpu().data.numpy())
                    corrs.append(cor)
                correlations[n][k].append(np.concatenate(corrs))
                
                #perm = torch.randperm(batch_size)
                #idxs1, idxs2 = perm[:n_samples], perm[n_samples:2*n_samples]
                
                #corrs = torch.tensor([[pearsonr(s1, s2) for s1 in agent_states[idxs1]]
                                      #for s2 in agent_states[idxs2]]).cpu().data.numpy()
    return np.array(correlations)   

def get_cka_metrics(community, loaders, n_tests=128, use_tqdm=False, device=torch.device('cuda')) : 
    
    n_samples = 10
    correlations = [[[] for _ in range(2)] for _ in range(2)]
    double_train_loader, double_test_loader = loaders
    
    if type(use_tqdm) is int : 
        position = use_tqdm
        use_tqdm = True

    elif use_tqdm : 
        position = 0
    
    pbar = range(n_tests)
    if use_tqdm : 
        pbar = tqdm(pbar, position=position, desc='Metric Trials', leave=None)
    for test in pbar :
        data, target = next(iter(double_test_loader))
        data = temporal_data(data, 5).to(device)
        batch_size = target.shape[0]
        for n in range(2) : 
            for k in range(2) : 
                outputs, states = community(fixed_digit(data, k))
                agent_states = states[-1][n][0]
                
                perm = torch.randperm(batch_size)
                idxs1, idxs2 = perm[:n_samples], perm[n_samples:2*n_samples]
                
                corrs = linear_cka_distance(agent_states[idxs1], agent_states[idxs2], reduce_bias=False).cpu().data.numpy()
                correlations[n][k].append(1-corrs)

    return np.array(correlations)  

def compute_correlation_metric(p_cons, loaders, save_name, use_maxs=False, device=torch.device('cuda'), notebook=False, config=None) : 
    
    cka_distances = {}
    pearson_corrs_parity, pearson_corrs_label, pearson_corrs_rotation = {}, {}, {}
    l=0
    
    tqdm_f = tqdm_n if notebook else tqdm
    if wandb.run is not None :
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
        community_states, *_ = get_wandb_artifact(config, name='state_dicts', process_config=True)
        print('Loading models from artifact')

    community = init_community(agent_params_dict, 0.1, device, use_deepR=config['model_params']['use_deepR'])

    task = wandb.config['task']
    print(task)
    wandb.config.update({'correlation_use_maxs' : use_maxs})

    try :
        community_metrics = torch.load(community_state_path + '_metrics')
    except FileNotFoundError : 
        use_maxs = False
    
    if use_maxs : 
        community_metrics = torch.load(community_state_path + '_metrics')
        sorted_idx = lambda metric : [np.argsort(key(metric[p])) for p in p_cons]
        
        #key = lambda metric : metric
        #sorted_indices = sorted_idx(community_metrics['Acc'])

        key = lambda metric : -(metric- 0.5)**2
        sorted_indices = sorted_idx(community_metrics['Deciding_ags'])

    for i, p_con in enumerate(tqdm_f(p_cons[l:], position=0, desc='Model Sparsity', leave=None)) : 
        
        pearson_corrs_parity[p_con] = []
        pearson_corrs_label[p_con] = []
        pearson_corrs_rotation[p_con] = []

        if use_maxs : 
            idxs = sorted_indices[i][5:]
            states = np.array(community_states[p_con])[idxs]
        else : 
            states = community_states[p_con]
            
        for i, state in enumerate(tqdm_f(states, position=1, desc='Model Trials', leave=None)) : 
            community.load_state_dict(state)
            
            pearsons_parity = get_pearson_metrics(community, loaders, fixed_mode='parity', use_tqdm=False, device=device)
            pearsons_label = get_pearson_metrics(community, loaders, fixed_mode='label', use_tqdm=False, device=device)
            
            if 'rotation' in task : 
                pearsons_rotation = get_pearson_metrics(community, loaders, fixed_mode=task, use_tqdm=False, device=device)
            else : 
                pearsons_rotation = 0

            pearson_corrs_parity[p_con].append(pearsons_parity)
            pearson_corrs_label[p_con].append(pearsons_label)
            pearson_corrs_rotation[p_con].append(pearsons_rotation)

        pearson_corrs_parity[p_con] = np.array(pearson_corrs_parity[p_con])
        pearson_corrs_label[p_con] = np.array(pearson_corrs_label[p_con])
        pearson_corrs_rotation[p_con] = np.array(pearson_corrs_rotation[p_con])
        
        #community_states = torch.load(community_state_path)
        final_correlations = {'Pearson_Parity' : pearson_corrs_parity, 'Pearson_Label' : pearson_corrs_label, 'Pearson_Rotation' : pearson_corrs_rotation}
        mkdir_or_save_torch(final_correlations, save_name, save_path)

    wandb.log_artifact(save_path + save_name, name='correlations', type='metric')
    fig1, fig2 = plot_and_log_correlations(final_correlations)
    wandb.log({'Correlation Metric' : wandb.Image(fig1), 'Correlation Difference Metric' : wandb.Image(fig2)})

def plot_and_log_correlations(correlations) : 
    
    pearsons_parity = correlations['Pearson_Parity']
    pearsons_labels = correlations['Pearson_Label']
    pearsons_rotation = correlations['Pearson_Rotation']

    p_cons = np.array(list(pearsons_labels.keys()))
    l = len(p_cons)
    linestyles = ['dashed', 'solid']

    if not list(pearsons_rotation.values())[0].mean() == 0 : 
        metrics, metric_names = [pearsons_labels, pearsons_parity, pearsons_rotation], ['Label Fixed', 'Parity Fixed', 'Rotation Fixed']
    else : 
        metrics, metric_names = [pearsons_labels, pearsons_parity], ['Label Fixed', 'Parity Fixed']

    mean_metric = lambda metric : torch.tensor(metric).flatten(start_dim=3).mean(-1).data.numpy()

    fig1, axs = plt.subplots(1, len(metrics), figsize=(20, 5))
    for m, (metric, metric_name) in enumerate(zip(metrics, metric_names)) : 
        ax = axs[m]
        means = [mean_metric(p) for p in metric.values()]
        for n in range(2) : 
            for k in range(2) : 
                linestyle = linestyles[n]
                mean = np.array([p[:, n, k].mean() for p in means])
                std = np.array([p[:, n, k].std() for p in means])
                plot = ax.plot(p_cons[:l], mean, label=f'Subnetwork {n}, Digit {k} Fixed', linestyle=linestyle)
                col = plot[-1].get_color()
                ax.fill_between(p_cons[:l], mean-std, mean+std, color=col, alpha=0.2)
                for p, p_con in zip(means, p_cons[:l]) : 
                    data_points = p[:, n, k].flatten()
                    ax.plot([p_con]*int((len(data_points))), data_points, '.', color=col, alpha=0.4)

        ax.set_title(f'Correlation Metric : {metric_name}', fontsize=15)
        ax.set_xlabel('Proportion of active connections', fontsize=15)
        ax.set_ylabel('Functional Specialization :\n Mean Correlation', fontsize=13)
        ax.legend()
        ax.set_xscale('log')

    fig1.suptitle('Correlation Metric', fontsize=15)

    normal_metric = lambda p : (p[:, n, n], p[:, n, 1-n])
    diff_metric = lambda p : ((normal_metric(p)[0]-normal_metric(p)[1])/(normal_metric(p)[0]+normal_metric(p)[1]))
    diff_metric_nan = lambda p : diff_metric(p)[~np.isnan(diff_metric(p))]

    fig2, axs = plt.subplots(1, 2, figsize=(15, 5))
    metric = [mean_metric(p) for p in pearsons_labels.values()]

    x_axis = [p_cons, (1-p_cons)/(2*(1+p_cons))]
    x_labels = ['Proportion of active connections', 'Q Modularity Measure']
    for i, p_cons_Q in enumerate(x_axis) : 
        ax = axs[i]
        for n in range(2) : 
            linestyle = linestyles[n]
            mean = np.array([diff_metric_nan(p).mean() for p in metric])
                #maxs = only_maxs[k][t]
            std = np.array([diff_metric_nan(p).std() for p in metric])
            plot = ax.plot(p_cons_Q[:l], mean, 
                    label=f'Subnetwork {n}', linestyle=linestyle)
            col = plot[-1].get_color()
            ax.fill_between(p_cons_Q[:l], mean-std, mean+std, color=col, alpha=0.2)
            for p, p_con in zip(metric, p_cons[:l]) : 
                data_points = diff_metric_nan(p).flatten()

        #ax.hlines(xmin=p_cons[0], xmax=p_cons[-1], y=0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_xlabel(x_labels[i], fontsize=15)
        if i == 0 : ax.set_ylabel('Functional Specialization :\n Pearson Coefficient Difference', fontsize=13)
        ax.legend(loc='upper left'*(i==1)+'upper right'*(i==0))
        ax.set_xscale('log'*(p_cons_Q is p_cons) + 'linear'*(p_cons_Q is not p_cons))

    #fig2.tight_layout()
    fig2.suptitle('Correlation Metric Diff', fontsize=15)

    return fig1, fig2

    



        
