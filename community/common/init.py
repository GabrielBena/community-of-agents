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

    agents_config = model_dict["agents"]
    connections_config = model_dict["connections"]
    sparsity = connections_config["sparsity"]
    readout_config = model_dict["readout"]

    n_agents = model_dict["n_agents"]

    try:
        n_ins = model_dict["n_ins"]
        n_ins[0]
    except (KeyError, TypeError) as e:
        n_ins = [None for _ in range(n_agents)]

    try:
        n_hiddens = model_dict["agents"]["n_hiddens"]
        n_hiddens[0]
    except (KeyError, TypeError) as e:
        n_hiddens = [None for _ in range(n_agents)]

    def modified_agent_dict(tag, n_in=None, n_hid=None):
        new_dict = copy(agents_config)
        if n_in is not None:
            new_dict["n_in"] = n_in
        if n_hid is not None:
            new_dict["n_hidden"] = n_hid
        try:
            new_dict.pop("n_hiddens")
        except KeyError:
            pass

        new_dict["tag"] = tag
        return new_dict

    agents = [
        Agent(**modified_agent_dict(n, n_in, n_hid))
        for n, (n_in, n_hid) in enumerate(zip(n_ins, n_hiddens))
    ]

    community = Community(
        agents,
        sparsity,
        readout_config,
        connections_config,
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
