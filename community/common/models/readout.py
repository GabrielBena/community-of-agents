import torch
import torch.nn as nn
import numpy as np
from community.data.tasks import get_factors_list
from warnings import warn


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
    elif n_readouts > 1:
        return [
            get_readout_dimensions(
                agents, common_readout, readout_from, 1, n_hid, n_out
            )
            for _ in range(n_readouts)
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


def configure_readouts(config, task=None):

    task = config["task"] if task is None else task
    readout_from = config["model"]["readout"]["readout_from"]

    if isinstance(task, list):

        if isinstance(readout_from, list):
            assert len(task) == len(readout_from)
            readout_config = [
                configure_single_readout(config, t, rf)
                for (t, rf) in zip(task, readout_from)
            ]
        else:
            readout_config = [configure_readouts(config, t) for t in task]

        readout_config = {
            k: [r[k] for r in readout_config] for k in readout_config[0].keys()
        }

    else:
        readout_config = configure_single_readout(config, task)

    return readout_config


def configure_single_readout(config, task, readout_from=None, n_hid=None):

    n_classes = config["datasets"]["n_classes"]
    n_classes_per_ag = config["datasets"]["n_classes_per_digit"]
    n_symbols = config["datasets"]["n_digits"]

    readout_config = {}

    readout_config["readout_from"] = (
        readout_from
        if readout_from is not None
        else config["model"]["readout"]["readout_from"]
    )

    readout_config["n_hid"] = (
        n_hid if n_hid is not None else config["model"]["readout"]["n_hid"]
    )

    # dummy_target = torch.zeros(2, 10)
    # n_out = get_task_target(dummy_target, task, n_classes_per_ag)[1]
    # readout_config["n_out"] = n_out

    try:
        task = int(task)
        readout_config["n_readouts"] = 1
        readout_config["n_out"] = n_classes_per_ag
        return readout_config

    except ValueError:
        pass

    if task == "family":

        factors = get_factors_list(n_symbols)
        readout_config["n_readouts"] = len(factors)
        readout_config["n_out"] = [n_classes for _ in range(len(factors))]

    elif task in ["both", "all", "none", "parity-digits-both"]:

        readout_config["n_readouts"] = n_symbols
        readout_config["n_out"] = [n_classes_per_ag for _ in range(n_symbols)]

    elif (
        task
        in [
            "sum",
            "max",
            "min",
            "count-max",
            "count-min",
        ]
        or "parity" in task
        or "count" in task
        or "bit" in task
    ):

        readout_config["n_readouts"] = 1
        readout_config["n_out"] = n_classes_per_ag

        if task == "sum":
            readout_config["n_out"] = n_classes

        elif "parity" in task:
            if "digits" in task:
                if "equal" in task:
                    readout_config["n_out"] = n_classes_per_ag + 1
            elif "both" in task:
                readout_config["n_readouts"] = n_symbols
                readout_config["n_out"] = [2 for _ in range(n_symbols)]
            else:
                readout_config["n_out"] = 2

        elif "bit" in task:
            if "last" in task:
                n_last = int(task.split("-")[-1])
                readout_config["n_out"] = int(2**n_last)
            else:
                n_bit = np.floor(np.log2(n_classes_per_ag - 1)) + 1
                readout_config["n_out"] = int(2**n_bit)

        else:  # task in ["parity-digits", "max", "min", "inv_parity-digits"]:
            readout_config["n_out"] = n_classes_per_ag

    else:
        warn(f"can't auto configure readout for task {task}")
        print(f" Warning ! Can't auto configure readout for task {task}")

    return readout_config
