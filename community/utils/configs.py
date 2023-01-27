from typing import Type
from warnings import warn
from community.data.tasks import get_factors_list


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

        """
        config["model"]["agents"]["n_out"] = n_classes_per_ag

        def get_nested_readout(task_list, n_readouts):
            try:
                return [[int(t) for t in task_list] for _ in range(n_readouts)]

            except TypeError:
                return [
                    get_nested_readout(t, n_r) for t, n_r in zip(task_list, n_readouts)
                ]
            except ValueError:
                return [[i for i in range(n_agents)] for t in task_list]

        def get_nested_len(task_list):
            if type(task_list[0]) is list:
                return [get_nested_len(t) for t in task_list]
            else:
                return len(task_list)

        if common_readout:
            n_readouts = config["model"]["n_readouts"] = get_nested_len(task)
            config["model"]["readout_from"] = get_nested_readout(task, n_readouts)
            config["model"]["agents"]["n_readouts"] = None

        else:
            config["model"]["n_readouts"] = None
            config["model"]["agents"]["n_readouts"] = len(task[0])
        """

    else:
        readout_config = configure_single_readout(config, task)

    return readout_config


def configure_single_readout(config, task, readout_from=None, n_hid=None):

    n_classes = config["datasets"]["n_classes"]
    n_classes_per_ag = config["datasets"]["n_classes_per_digit"]
    symbol_config = config["datasets"]["symbol_config"]
    n_symbols = symbol_config["n_diff_symbols"]

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

        factors = get_factors_list(symbol_config["n_diff_symbols"])
        readout_config["n_readouts"] = len(factors)
        readout_config["n_out"] = [n_classes for _ in range(len(factors))]

    elif task in ["both", "all", "none", "parity-both"]:

        readout_config["n_readouts"] = n_symbols
        readout_config["n_out"] = [n_classes_per_ag for _ in range(n_symbols)]

    elif task in [
        "sum",
        "parity-digits",
        "inv_parity-digits",
        "parity",
        "max",
        "min",
        "count-max",
        "count-min",
    ]:

        readout_config["n_readouts"] = 1
        if task == "sum":
            readout_config["n_out"] = n_classes
        elif task == "parity":
            readout_config["n_out"] = 2
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
