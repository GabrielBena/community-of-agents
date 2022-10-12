from matplotlib.rcsetup import non_interactive_bk
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
from copy import copy

from .models.agents import Agent
from .models.ensembles import Community


def init_community(model_dict, device=torch.device("cuda")):
    """
    Global model initialization
    Args :
        agents_params : parameters of sub-networks
        p_connect : sparsity of interconnections
    """
    agents_params_dict = model_dict["agents_params"]
    connections_params_dict = model_dict["connections_params"]

    common_readout = model_dict["common_readout"]
    dual_readout = model_dict["common_dual_readout"]
    n_ins = model_dict["n_ins"]
    n_agents = model_dict["n_agents"]

    def modified_agent_dict(tag, n_in=None):
        new_dict = copy(agents_params_dict)
        if n_in is not None:
            new_dict["n_in"] = n_in
        new_dict["tag"] = tag
        return new_dict

    if n_ins is None:
        agents = [Agent(**modified_agent_dict(n)) for n in range(n_agents)]
    else:
        agents = [Agent(**modified_agent_dict(n, n_in)) for n, n_in in enumerate(n_ins)]

    community = Community(
        agents,
        common_readout=common_readout,
        dual_readout=dual_readout,
        **connections_params_dict
    ).to(device)

    return community


def init_optimizers(community, params_dict, deepR_params_dict):
    """
    Optimizers initialization
    Args :
        community : global model
        params_dict : sub-networks learning parameters
        deepR_params_dict : Sparse connections learning parameters
    """

    connect_params = community.connections.parameters()
    agents_params = community.agents.parameters()

    optimizer_agents = optim.Adam(agents_params, lr=params_dict["lr"])
    optimizer_agents = torch.optim.Adam(community.parameters(), lr=1e-3)
    try:
        optimizer_connections = optim.Adam(connect_params, lr=deepR_params_dict["lr"])
        scheduler_connections = StepLR(
            optimizer_connections, step_size=1, gamma=deepR_params_dict["gamma"]
        )
    except (ValueError, KeyError):  # no connections
        optimizer_connections = optim.Adam([torch.tensor(0)])
        scheduler_connections = StepLR(optimizer_connections, step_size=1)

    optimizers = [optimizer_agents, None]
    scheduler_agents = StepLR(optimizer_agents, step_size=1, gamma=params_dict["gamma"])
    schedulers = [scheduler_agents, scheduler_connections]

    return optimizers, schedulers
