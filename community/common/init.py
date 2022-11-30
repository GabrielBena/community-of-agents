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
    agents_params_dict = model_dict["agents"]
    connections_params_dict = model_dict["connections"]

    n_readouts = model_dict["n_readouts"]
    readout_from = model_dict["readout_from"]
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
        n_readouts=n_readouts,
        readout_from=readout_from,
        **connections_params_dict,
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
    reg_readout = params_dict["reg_readout"]
    if not reg_readout:
        optimizer = torch.optim.Adam(community.parameters(), lr=1e-3)
    else:
        optimizer = torch.optim.AdamW(
            [p for n, p in community.named_parameters() if "readout" not in n],
            lr=1e-3,
        )
        optimizer.add_param_group(
            {
                "params": [
                    p for n, p in community.named_parameters() if "readout" in n
                ],
                "lr": 1e-3,
                "weight_decay": reg_readout,
            }
        )

    scheduler_agents = StepLR(optimizer, step_size=1, gamma=params_dict["gamma"])
    schedulers = [scheduler_agents, None]

    return [optimizer, None], schedulers
