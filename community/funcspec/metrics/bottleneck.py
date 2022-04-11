import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from models.community import train_community, init_community, init_optimizers
from data.datasets_and_tasks import DoubleMNIST



### ------ Bottleneck Metric ------ : 

def readout_retrain(community, loaders, n_classes=10, lrs=[1e-3, 1e-3], n_epochs=5, n_tests=3,
                    use_tqdm=False, train_all_param=False, device=torch.device('cuda')) : 
    """
    Retrains the bottleneck-readout connections of each sub-network for each sub-task and stores performance.
    Args : 
        community : trained model on global task
        n_classes : number of classes of the new readout layer
        loader : training and testing data-loaders
        n_tests : number of init and train to conduct
        n_epochs : number of epochs of training
        lrs : learning rate of training : [subnets, connections]
        train_all_params : train all sub-networks parameters as well as interconnections. If False, train only one bottleneck-readout at a time
    """
    
    double_train_loader, double_test_loader = loaders
    
    pbar = range(n_tests)
    if use_tqdm : 
        pbar = tqdm(pbar, position=2, desc='Metric Trials : ', leave=None)
    #single_losses_total, single_accs_total = [], []
    single_losses_total, single_accs_total = [], []
    for test in pbar : 
        single_losses_dig, single_accs_dig = [[] for _ in range(2)], [[] for _ in range(2)]
        
        for target_digit in range(2) :
            single_losses_ag, single_accs_ag = [[] for _ in range(2)], [[] for _ in range(2)]
            for n in range(2) : 
                f_community = copy.deepcopy(community)
                for f_agent in community.agents : 
                    if f_agent.use_bottleneck : 
                        f_agent.readout = nn.Linear(f_agent.bottleneck.out_features, n_classes)
                    else : 
                        f_agent.readout = nn.Linear(f_agent.dims[-2], n_classes)
                    f_agent.to(device)

                for name, p in f_community.named_parameters() : 
                    if 'readout' in name and str(n) in name:
                        p.requires_grad = True
                    else : 
                        p.requires_grad = train_all_param
            
                params = lr_ag, gamma = lrs[0], 0.9
                params_dict = {'lr' : lr_ag, 'gamma' : gamma}

                deepR_params = l1, gdnoise, lr_con, gamma, cooling = 1e-5, 1e-3, lrs[1], 0.95, 0.95
                deepR_params_dict = {'l1' : l1, 'gdnoise' : gdnoise, 'lr' : lr_con, 'gamma' : gamma, 'cooling' : cooling}
                
                decision_params = ('last', str(n))    
                optimizers, schedulers = init_optimizers(f_community, params_dict, deepR_params_dict)
                optimizer_agents = optimizers[0]
                
                train_out = train_community(f_community, device, double_train_loader, double_test_loader, optimizers, 
                            n_epochs=n_epochs, global_rewire=False, schedulers=schedulers, deepR_params_dict=deepR_params_dict,
                            decision_params=decision_params, task=str(target_digit), train_connections=False,
                            trials = (True, True), use_tqdm=False)

                (train_losses, train_accs), (test_losses, test_accs), *_ = train_out

                single_losses_ag[n].extend(test_losses)
                single_accs_ag[n].extend(test_accs)

            single_losses_dig[target_digit] = np.array(single_losses_ag)
            single_accs_dig[target_digit] = np.array(single_accs_ag)
        

        single_losses_total.append(np.array(single_losses_dig))
        single_accs_total.append(np.array(single_accs_dig))
            
    return {'losses' : np.array(single_losses_total), 'accs' : np.array(single_accs_total)}


def compute_bottleneck_metrics(p_cons, community_state_path, model_params, loaders, save_name, use_maxs=True, device=torch.device('cuda')) : 
    """
    Compute complete bottleneck metric, for every trained community and sparsity of interconnections
    """
    metrics = {}
    l = 0
    
    double_train_loader, double_test_loader = loaders
    agents_params, community_params = model_params
    community_states = torch.load(community_state_path)
    community = init_community(agents_params, community_params, device)
    if use_maxs : 
        community_metrics = torch.load(community_state_path + '_metrics')
        sorted_idx = lambda metric : [np.argsort(metric[p]) for p in p_cons]
        sorted_indices = sorted_idx(community_metrics['Acc'])

    
    n_tests = len(community_states)
    for i, p_con in enumerate(tqdm(p_cons[l:], position=0, desc='Model Sparsity : ', leave=None)) : 
        metrics[p_con] = {}
        metrics[p_con]['losses'], metrics[p_con]['accs'] = [], []
        if use_maxs : 
            idxs = sorted_indices[i][5:]
            states = np.array(community_states[p_con])[idxs]
        else : 
            states = community_states[p_con]
        
        for i, state in enumerate(tqdm(states, position=1, desc = 'Model Trials', leave=None)) : 
            community = init_community(agents_params, community_params, device)
            community.load_state_dict(state)
            metric = readout_retrain(community, loaders, use_tqdm=True, device=device)     
            for metric_name in ['losses', 'accs'] :
                metrics[p_con][metric_name].append(metric[metric_name])
                
        metrics[p_con]['losses'] = np.array(metrics[p_con]['losses'])
        metrics[p_con]['accs'] = np.array(metrics[p_con]['accs'])
        
        torch.save({'metrics' : metrics}, save_name)
        
        
        
