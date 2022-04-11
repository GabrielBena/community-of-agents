import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from models.community import train_community, init_community
from models.weight_masks import GetSubnet_global, Mask_Community, Mask, get_proportions, get_repartitions
from data.datasets_and_tasks import DoubleMNIST


### ------Weight mask Metrics -------
def train_and_compute_mask_metric(community, sparsity, loaders, n_tests=10, n_epochs=1, lr=0.1, device=torch.device('cuda')) : 
    """
    Initializes and trains masks on community model.
    Args : 
        community : model to apply the mask onto
        sparsity : % of weights to be selected
        loader : training and testing data-loaders
        n_tests : number of init and train to conduct
        n_epochs : number of epochs of training
        lr : learning rate of training
        
    Returns : 
        proportion_per_agent : weight repartition among sub-networks
        accuracies : trained masks accuracies
        losses : trained masks losses
        best_states : state dicts of trained masks
    """
    prop_per_agent_total = []
    test_accuracies_total = []
    test_losses_total = []
    best_states_total = []
    double_train_loader, double_test_loader = loaders
    
    for test in tqdm(range(n_tests), position=3, desc='Metric Trials : ', leave=None) : 
        prop_per_agent = []
        test_accuracies = []
        test_losses = []
        best_states = []
        for target_digit in range(2) : 
            masked_community = Mask_Community(community, sparsity).to(device)
            momentum, wd = 0.9, 0.0005
            optimizer_agents = optim.SGD(
                [p for p in masked_community.parameters() if p.requires_grad],
                lr=lr,
                momentum=momentum,
                weight_decay=wd,
            )
            optimizer_connections = optim.Adam(community.connections.parameters())
            optimizers = [optimizer_agents, optimizer_connections]
            
            decision_params = ('last', 'max')
            train_out = train_community(masked_community, device, double_train_loader, double_test_loader, optimizers, 
                                    n_epochs=n_epochs, task=str(target_digit), decision_params=decision_params,
                                    train_connections=False, trials = (True, True), use_tqdm=False)
                                    

            (train_losses, train_accs), (test_loss, test_accs), deciding_agents, best_state = train_out

            prop = get_proportions_per_agent(masked_community)[0]
            
            prop_per_agent.append(prop)
            test_accuracies.append(np.array(test_accs))
            test_losses.append(np.array(test_loss))
            best_states.append(best_state)
        
        best_states_total.append(np.array(best_states))
        prop_per_agent_total.append(np.array(prop_per_agent))
        test_accuracies_total.append(np.array(test_accuracies))
        test_losses_total.append(np.array(test_losses))
        
    return np.array(prop_per_agent_total), np.array(test_accuracies_total), np.array(test_losses_total), best_states_total
    
def get_proportions_per_agent(masked_community) : 
    """
    Returns repartition of selected weights among agents : 
    Args : 
        masked_community : weight mask applied to community
    """

    prop_ag_0 = 0
    prop_ag_1 = 0
    sparsity = masked_community.sparsity
    numel = 0 
    for p in masked_community.scores.values() : 
        numel += p.numel()
    sparsity = masked_community.sparsity
    for n, p in get_proportions(masked_community).items() : 
        if 'agents' in n : 
            if n[7] == '0' : 
                prop_ag_0 += p[1]/sparsity
            elif n[7] == '1' : 
                prop_ag_1 += p[1]/sparsity
            
        if n[0] == '0' and 'thetas' not in n : 
            prop_ag_0 += p[1]/sparsity
        elif n[0] == '1' and 'thetas' not in n : 
            prop_ag_1 += p[1]/sparsity

    prop = np.array((prop_ag_0, prop_ag_1))
    return prop, prop*numel*sparsity    

def compute_mask_metric(p_cons, community_state_path, model_params, loaders, p_masks, lr, save_name, use_maxs=True, device=torch.device('cuda')) : 
    """
    Compute complete mask metric, for every trained community and sparsity of interconnections
    """
    proportions_per_agent = {}
    test_accuracies_supermasks = {}
    test_losses_supermasks = {}
    trained_masks = {}
    l = 0
    
    double_train_loader, double_test_loader = loaders
    agents_params, community_params = model_params
    community_states = torch.load(community_state_path)
    community = init_community(agents_params, community_params, device)
    n_tests = len(community_states)

    if use_maxs : 
        community_metrics = torch.load(community_state_path + '_metrics')
        sorted_idx = lambda metric : [np.argsort(metric[p]) for p in p_cons]
        sorted_indices = sorted_idx(community_metrics['Acc'])

    for i, p_con in enumerate(tqdm(p_cons[l:], position=0, desc='Community Sparsity : ', leave=None)) : 

        prop_per_agent = {}    
        test_accs = {}
        test_losses = {}
        best_states = {}
        for p_mask in tqdm(p_masks, position=1, desc='Mask Sparsity', leave=None) :
            prop_per_agent[p_mask], test_accs[p_mask], test_losses[p_mask], best_states[p_mask] = [], [], [], []
            
            if use_maxs : 
                idxs = sorted_indices[i][5:]
                states = np.array(community_states[p_con])[idxs]
            else : 
                states = community_states[p_con]

            for i, state in enumerate(tqdm(states, position=2, desc='Model Trials : ', leave=None)) :                     
                community.load_state_dict(state)
                prop, acc, loss, states = train_and_compute_mask_metric(community, p_mask, loaders, lr=lr, device=device)
                prop_per_agent[p_mask].append(prop)
                test_accs[p_mask].append(acc)
                test_losses[p_mask].append(loss)
                best_states[p_mask].append(states)
                
            prop_per_agent[p_mask], test_accs[p_mask] = np.array(prop_per_agent[p_mask]), np.array(test_accs[p_mask])
            test_losses[p_mask], best_states[p_mask] = np.array(test_losses[p_mask]), np.array(best_states[p_mask])
            
        proportions_per_agent[p_con] = prop_per_agent
        test_accuracies_supermasks[p_con] = test_accs
        test_losses_supermasks[p_con] = test_losses
        trained_masks[p_con] = best_states
        
        community_states = torch.load(community_state_path)
        torch.save({'Proportions' : proportions_per_agent, 'Accs' : test_accuracies_supermasks,
                    'Losses' : test_losses_supermasks, 'Trained_masks' : trained_masks}, save_name)
        
        
        
def get_metrics_from_saved_masks(mask_file, masked_community, sparsities) : 
    """
    Computes the weight mask metrics for different k% of selected weights from saved masks states
    """
    p_cons = list(mask_file['Trained_masks'].keys())
    l = len(p_cons)
    sparsity = list(mask_file['Trained_masks'][p_cons[0]].keys())[0]
    states = {p : np.array(mask_file['Trained_masks'][p][sparsity]) for p in p_cons[:l]}
    proportions = {}
    thresholds = {}
    for p_con in tqdm(p_cons[:]) : 
        proportions[p_con] = {s : [[] for _ in range(2)] for s in sparsities}
        thresholds[p_con] = [[] for _ in range(2)]
        for t in range(2) :
            for state in states[p_con][..., t].flatten() : 
                masked_community.load_state_dict(state)
                #thresholds[p_con][t].append(get_appartenance_treshold(masked_community, t)[1])
                for sparsity in sparsities : 
                    masked_community.sparsity = sparsity
                    prop = get_proportions_per_agent(masked_community)[0]
                    proportions[p_con][sparsity][t].append(prop)

        thresholds[p_con] = np.array(thresholds[p_con])
        for sparsity in sparsities :
            proportions[p_con][sparsity] = np.array(proportions[p_con][sparsity])

    return proportions, thresholds

        

