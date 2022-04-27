import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy
import math
import torch.autograd as autograd
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_n

import warnings
warnings.filterwarnings('ignore')

from community.common.training import train_community
from community.common.init import init_community

# ------ Weight Masks Models ------

"""
Weight Masks training in PyTorch
Based on  : 
"What's Hidden in a Randomly Weighted Neural Network?"
Vivek Ramanujan, Mitchell Wortsman, Aniruddha Kembhavi, Ali Farhadi, Mohammad Rastegari
(https://arxiv.org/abs/1911.13299)
https://github.com/RAIVNLab/hidden-networks
"""

class GetSubnet_global(autograd.Function):
    """
    Variation of the original sorting function, select top k% scoring parameter globally, instead of layer-per-layer
    Args : 
        score : single parameter score to be selected on (needed for backward pass to have only one corresponding gradient)
        scores : total list of scores, corresponding to all parameters of model
        k : % of weights to be selected
    
    """
    @staticmethod
    def forward(ctx, score, scores, k):
        score_idx = [s is score for s in scores].index(True)
        identities = torch.cat([torch.ones_like(s.flatten())*n for n, s in enumerate(scores)])
        _, idx = torch.cat([s.flatten() for s in scores]).sort()
        j = int((1 - k) * idx.numel())

        idxs_0 = [idx[:j][identities[idx[:j]]==n] for n, _ in enumerate(scores)]
        idxs_1 = [idx[j:][identities[idx[j:]]==n] for n, _ in enumerate(scores)]
        outs = []
        substract = [0]
        substract.extend([s.numel() for s in scores[:-1]])
        for i, (s, idx_0, idx_1) in enumerate(zip(scores, idxs_0, idxs_1)):
            sub = np.sum(substract[:i+1])
            out = s.clone()
            flat_out = out.flatten()
            flat_out[(idx_0-sub).long()] = 0
            flat_out[(idx_1-sub).long()] = 1
            
            #print(out.requires_grad)
            outs.append(out)
            
        return outs[score_idx]

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None, None
    
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None
    
def get_new_name(name) : 
    new_name = ''
    for n in name.split('.') : 
        new_name += n + '_'
    return new_name[:-1]

class Mask(nn.Module):
    """
    weight mask class, to be applied to any torch Module
    Args : 
        model : model to apply mask onto
        sparsity : % of the weights to select in mask
    """
    def __init__(self, model, sparsity=0.5) : 
        super().__init__()
        self.model = copy.deepcopy(model)
        self.sparsity = sparsity
        self.scores = nn.ParameterDict()
        for name, p in self.model.named_parameters():
            n = get_new_name(name)
            if 'bias' not in n : 
                self.scores[n] = nn.Parameter(torch.Tensor(p.size()), requires_grad=True)
                nn.init.kaiming_uniform_(self.scores[n], a=math.sqrt(5))
            else : 
                self.scores[n] = nn.Parameter(torch.Tensor(p.size()), requires_grad=True)
                nn.init.normal_(self.scores[n], 0, 1e-2)
            p.requires_grad = False
        
    def forward(self, x) : 
        subnets = [GetSubnet_global.apply(s, list(self.scores.values()), self.sparsity) for s in self.scores.values()]
        f_model = copy.deepcopy(self.model)
        for i, (name, p) in enumerate(f_model.named_parameters()) : 
            p *= subnets[i]
            
        return f_model(x)
    
class Mask_Community(nn.Module):
    """
    weight mask class, to be applied to Community class Module
    Args : 
        model : Community to apply mask onto
        sparsity : % of the weights to select in mask
    """
    def __init__(self, model, sparsity=0.05, scaling=False) : 
        super().__init__()
        self.model = copy.deepcopy(model)
        self.sparsity = sparsity
        self.scores = nn.ParameterDict()
        self.is_community = False
        for name, p in self.model.agents.named_parameters():
            if not 'ih' in name : #Mask not applied to input weights
                n = get_new_name(name)
                if 'bias' not in n : 
                    self.scores[n] = nn.Parameter(torch.Tensor(p.size()), requires_grad=True)
                    nn.init.kaiming_uniform_(self.scores[n], a=math.sqrt(5))
                else : 
                    self.scores[n] = nn.Parameter(torch.Tensor(p.size()), requires_grad=True)
                    nn.init.normal_(self.scores[n], 0, 1e-2)
            
            p.requires_grad = False
                
    def forward(self, x) : 
        subnets = [GetSubnet_global.apply(s, list(self.scores.values()), self.sparsity) for s in self.scores.values()]
        f_model = copy.deepcopy(self.model)
        i = 0
        for (name, p) in f_model.agents.named_parameters() : 
            if not 'ih' in name :
                p *= subnets[i]
                i += 1
        return f_model(x)
    
def get_proportions(masked_model) : 
    scores = list(masked_model.scores.values())
    d_scores = masked_model.scores
    k = masked_model.sparsity
    identities = torch.cat([torch.ones_like(s.flatten())*n for n, s in enumerate(scores)])
    _, idx = torch.cat([s.flatten() for s in scores]).sort()
    j = int((1 - k) * idx.numel())

    idxs_0 = [idx[:j][identities[idx[:j]]==n] for n, _ in enumerate(scores)]
    idxs_1 = [idx[j:][identities[idx[j:]]==n] for n, _ in enumerate(scores)]
    
    proportion_0 = [idxs.numel()/idx.numel() for idxs in idxs_0]
    proportion_1 = [idxs.numel()/idx.numel() for idxs in idxs_1]
    
    return {name : (p_0, p_1) for name, p_0, p_1 in zip(d_scores.keys(), proportion_0, proportion_1)}
    
    
def get_repartitions(masked_model) : 
    scores = list(masked_model.scores.values())
    k = masked_model.sparsity
    identities = torch.cat([torch.ones_like(s.flatten())*n for n, s in enumerate(scores)])
    _, idx = torch.cat([s.flatten() for s in scores]).sort()
    j = int((1 - k) * idx.numel())
    
    top_idxs = identities[idx[j:]]
    top_idxs = (top_idxs/identities.max()).round()
    
    return idx[j:].cpu().data.numpy(), top_idxs.cpu().data.numpy()


### ------Weight mask Metrics -------

def train_and_get_mask_metric(community, sparsity, loaders, n_tests=10, n_epochs=1, lr=0.1, device=torch.device('cuda')) : 
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

    notebook = is_notebook()
    tqdm_f = tqdm_n if notebook else tqdm
    
    for test in tqdm_f(range(n_tests), position=2, desc='Metric Trials : ', leave=None) : 
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
            try : 
                optimizer_connections = optim.Adam(community.connections.parameters())
            except ValueError : #no connections
                optimizer_connections = optim.Adam([torch.tensor(0)])

            optimizers = [optimizer_agents, optimizer_connections]

            training_dict = {
                    'n_epochs' : n_epochs, 
                    'task' : str(target_digit),
                    'global_rewire' : False, 
                    'check_gradients' : False, 
                    'reg_factor' : 0.,
                    'train_connections' : False,
                    'decision_params' : ('last', 'max'),
                    'early_stop' : False ,
                    'deepR_params_dict' : {},
                }

            train_out = train_community(masked_community, *loaders, optimizers, 
                                        config=training_dict, device=device,
                                        trials = (True, True), use_tqdm=True)
                                    
            test_loss, test_accs, best_state = train_out['test_losses'], train_out['test_accs'], train_out['best_state']

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
                prop, acc, loss, states = train_and_get_mask_metric(community, p_mask, loaders, lr=lr, device=device)
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



        

