from typing import Type
from warnings import warn
from community.data.tasks import get_factors_list
import numpy as np


def get_training_dict(config):

    training_dict = {
        "n_epochs": config["training"]["n_epochs"],
        "task": config["task"],
        "global_rewire": config["optimization"]["connections"]["global_rewire"],
        "check_gradients": False,
        "reg_factor": 0.0,
        "train_connections": True,
        "decision": config["training"]["decision"],
        "stopping_acc": config["training"]["stopping_acc"],
        "early_stop": config["training"]["early_stop"],
        "deepR_params_dict": config["optimization"]["connections"],
        "data_type": config["datasets"]["data_type"],
        "force_connections": config["training"]["force_connections"],
        "sparsity": config["model"]["connections"]["sparsity"],
        "n_classes": config["datasets"]["n_classes"],
        "n_classes_per_digit": config["datasets"]["n_classes_per_digit"],
        "common_input": config["datasets"]["common_input"],
        "nb_steps": config["datasets"]["nb_steps"],
    }

    return training_dict


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

    elif task in ["both", "all", "none", "parity-both"]:

        readout_config["n_readouts"] = n_symbols
        readout_config["n_out"] = [n_classes_per_ag for _ in range(n_symbols)]

    elif (
        task
        in [
            "sum",
            "parity-digits",
            "inv_parity-digits",
            "parity",
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

        if task == "sum":
            readout_config["n_out"] = n_classes
        elif task == "parity":
            readout_config["n_out"] = 2

        elif "bit" in task:
            if "last" in task:
                n_last = int(task.split("-")[-1])
                readout_config["n_out"] = int(2**n_last)
            else:
                n_bit = np.floor(np.log2(n_classes_per_ag - 1)) + 1
                readout_config["n_out"] = int(2**n_bit)

        elif task in ["parity-equal"]:
            readout_config = n_classes_per_ag + 1

        else:  # task in ["parity-digits", "max", "min", "inv_parity-digits"]:
            readout_config["n_out"] = n_classes_per_ag

    else:
        warn(f"can't auto configure readout for task {task}")
        print(f" Warning ! Can't auto configure readout for task {task}")

    return readout_config


def find_and_change(config, param_name, param_value):

    for key, value in config.items():
        if type(value) is dict:
            find_and_change(value, param_name, param_value)
        else:
            if key == param_name:
                config[key] = param_value


sentinel = object()


def _finditem(obj, key, sentinel=sentinel):
    if key in obj:
        return obj[key]
    for k, v in obj.items():
        if isinstance(v, dict):
            item = _finditem(v, key, sentinel)
            if item is not sentinel:
                return item

    return sentinel


def get_new_config(config, key_prefix="config"):
    new_config = {}
    for k1, v1 in config.items():
        if type(v1) is dict:
            sub_config = get_new_config(v1, k1)
            new_config.update({key_prefix + "." + k: v for k, v in sub_config.items()})
        else:
            new_config[key_prefix + "." + k1] = v1
    return new_config


def ensure_config_coherence(config, v_params):

    n_agents = config["model"]["n_agents"]

    if config["task"] == "shared_goals":
        task = config["task"] = [
            [str(i), str((i + 1) % n_agents)] for i in range(n_agents)
        ]
    if config["task"] == "count-max":
        try:
            config["datasets"]["adjust_probas"] = True
            config["datasets"]["symbol_config"]["adjust_probas"] = True
        except KeyError:
            pass

    if "parity" in config["task"]:
        config["datasets"]["fix_asym"] = True

    if "n_classes_per_digit" in v_params:
        config["datasets"]["n_classes"] = (
            config["datasets"]["n_classes_per_digit"] * config["model"]["n_agents"]
        )
        config["datasets"]["symbol_config"]["n_symbols"] = (
            config["datasets"]["n_classes"] - 1
        )
    elif "n_classes" in v_params:
        config["datasets"]["n_classes_per_digit"] = (
            config["datasets"]["n_classes"] // config["model"]["n_agents"]
        )
        config["datasets"]["symbol_config"]["n_symbols"] = (
            config["datasets"]["n_classes"] - 1
        )
    elif "n_symbols" in v_params:
        config["datasets"]["n_classes"] = (
            config["datasets"]["symbol_config"]["n_symbols"] - 1
        )
        config["datasets"]["n_classes_per_digit"] = (
            config["datasets"]["n_classes"] // config["model"]["n_agents"]
        )

    elif "common_input" in v_params:
        if config["datasets"]["data_type"] != "symbols":

            config["datasets"]["input_size"] = 784 * (
                1 + config["datasets"]["common_input"]
            )
            config["model"]["agents"]["n_in"] = config["datasets"]["input_size"]
