import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
import copy

from .models.agents import Agent
from .models.ensembles import Community

cell_types_dict = {str(t) : t for t in [nn.RNN, nn.LSTM, nn.GRU]}


def init_community(agents_params_dict, p_connect, use_deepR=True, device=torch.device('cuda')) : 
    """
    Global model initialization
    Args : 
        agents_params : parameters of sub-networks
        p_connect : sparsity of interconnections
    """
    if type(agents_params_dict) is dict : 
        n_agents, n_in, n_ins, n_hidden, n_layers, n_out, train_in_out, use_readout, cell_type, use_bottleneck, dropout = list(agents_params_dict.values())
    else : 
        n_agents, n_in, n_ins, n_hidden, n_layers, n_out, train_in_out, use_readout, cell_type, use_bottleneck, dropout = agents_params_dict

    if type(cell_type) is str : 
        cell_type = cell_types_dict[cell_type]

    if n_ins is None : 
        agents = [Agent(n_in, n_hidden, n_layers, n_out, str(n),
                use_readout, train_in_out, cell_type, use_bottleneck, dropout) for n in range(n_agents)]

    else: 
        agents = [Agent(n_in, n_hidden, n_layers, n_out, str(n),
                    use_readout, train_in_out, cell_type, use_bottleneck, dropout) for n, n_in in enumerate(n_ins)]

    sparse_connections = (np.ones((n_agents, n_agents)) - np.eye(n_agents))*p_connect    
    community = Community(agents, sparse_connections, use_deepR).to(device)
    
    return community

def init_optimizers(community, params_dict, deepR_params_dict) : 
    """
    Optimizers initialization
    Args : 
        community : global model
        params_dict : sub-networks learning parameters
        deepR_params_dict : Sparse connections learning parameters
    """
    connect_params = community.connections.parameters()
    agents_params = community.agents.parameters()
    
    optimizer_agents = optim.Adam(agents_params, lr=params_dict['lr'])
    try : 
        optimizer_connections = optim.Adam(connect_params, lr=deepR_params_dict['lr'])
    except ValueError : #no connections
        optimizer_connections = optim.Adam([torch.tensor(0)])
    optimizers = [optimizer_agents, optimizer_connections]

    scheduler_agents = StepLR(optimizer_agents, step_size=1, gamma=params_dict['gamma'])
    scheduler_connections = StepLR(optimizer_connections, step_size=1, gamma=deepR_params_dict['gamma'])
    schedulers = [scheduler_agents, scheduler_connections]
    
    return optimizers, schedulers