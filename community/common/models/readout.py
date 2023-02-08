import torch
import torch.nn as nn
import numpy as np


def gather(l, rf):
    return [l[i] for i in rf]


def init_readout_weights(readout):
    if not isinstance(readout, nn.ReLU):

        try:
            nn.init.kaiming_uniform_(readout.weight, nonlinearity="relu")
        except AttributeError:
            [init_readout_weights(r) for r in readout]


def gather(l, gather_from):
    return [l[i] for i in gather_from]


def get_readout_dimensions(
    agents, common_readout, readout_from, n_readouts, n_hid, n_out
):

    nested_params = readout_from, n_readouts, n_hid, n_out
    n_agents = len(agents)

    if isinstance(n_readouts, list):
        # every parameters must be follow the same nested structure
        assert len(torch.tensor([len(p) for p in nested_params]).unique()) == 1
        # return nested recursive
        return [
            get_readout_dimensions(agents, common_readout, *n_params)
            for n_params in zip(*nested_params)
        ]

    elif isinstance(n_out, list):
        assert len(n_out) == n_readouts
        return [
            get_readout_dimensions(agents, common_readout, readout_from, 1, n_hid, n_o)
            for n_o in n_out
        ]
    else:
        if readout_from is None:
            if common_readout:
                readout_from = np.arange(n_agents)
            else:
                readout_from = [0]

        if n_hid is None:

            readout_dims = [
                np.sum([ag.dims[-2] for ag in gather(agents, readout_from)]),
                n_out,
            ]
        elif isinstance(n_hid, list):
            readout_dims = [
                np.sum([ag.dims[-2] for ag in gather(agents, readout_from)]),
                *n_hid,
                n_out,
            ]
        else:
            readout_dims = [
                np.sum([ag.dims[-2] for ag in gather(agents, readout_from)]),
                n_hid,
                n_out,
            ]

        return readout_dims


def create_readout_from_dims(readout_dims):

    try:
        readout = nn.ModuleList(
            [create_readout_from_dims(r_dim) for r_dim in readout_dims]
        )
    except (TypeError, IndexError) as e:
        readout = [
            nn.Linear(d1, d2) for d1, d2 in zip(readout_dims[:-1], readout_dims[1:])
        ]

        if len(readout) == 1:
            readout = readout[0]
        else:
            for i in range(1, len(readout), 2):
                readout.insert(i, nn.ReLU())
            readout = nn.Sequential(*readout)

    return readout


def readout_process(readout, readout_from, input):

    if isinstance(readout_from, list):
        out = [readout_process(r, rf, input) for r, rf in zip(readout, readout_from)]
    else:
        if readout_from is None:
            gathered_input = torch.cat([*input], -1)
        elif isinstance(readout_from, tuple):
            gathered_input = torch.cat([s for s in gather(input, readout_from)], -1)
        else:
            gathered_input = torch.cat([s for s in gather(input, [readout_from])], -1)
        try:
            out = readout(gathered_input).squeeze()
        except NotImplementedError:
            out = torch.stack([r(gathered_input) for r in readout]).squeeze()
    try:
        return torch.stack([*out]).squeeze()
    except RuntimeError:
        return out
