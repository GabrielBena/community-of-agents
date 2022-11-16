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
        "decision_params": config["training"]["decision_params"],
        "stopping_acc": config["training"]["stopping_acc"],
        "early_stop": config["training"]["early_stop"],
        "deepR_params_dict": config["optimization"]["connections"],
        "data_type": config["datasets"]["data_type"],
        "force_connections": config["training"]["force_connections"],
        "sparsity": config["model_params"]["connections_params"]["sparsity"],
        "n_classes": config["datasets"]["n_classes"],
        "n_classes_per_digit": config["datasets"]["n_classes_per_digit"],
    }

    return training_dict


def configure_readouts(config):

    common_readout = config["model_params"]["common_readout"]
    n_agents = config["model_params"]["n_agents"]
    task = config["task"]
    n_classes = config["datasets"]["n_classes"]
    n_classes_per_ag = config["datasets"]["n_classes_per_digit"]
    symbol_config = config["datasets"]["symbol_config"]

    if task == "family":

        config["model_params"]["agents_params"]["n_out"] = n_classes
        factors = get_factors_list(symbol_config["n_diff_symbols"])
        print(factors)

        if common_readout:

            config["model_params"]["n_readouts"] = len(factors)
            config["model_params"]["agents_params"]["n_readouts"] = None
            config["model_params"]["readout_from"] = None

        else:
            config["model_params"]["n_readouts"] = None
            config["model_params"]["agents_params"]["n_readouts"] = len(factors)

    elif type(task) is list:

        config["model_params"]["agents_params"]["n_out"] = n_classes_per_ag

        def get_nested_readout(task_list, n_readouts):
            try:
                return [[int(t) for t in task_list] for _ in range(n_readouts)]
            except TypeError:
                return [
                    get_nested_readout(t, n_r) for t, n_r in zip(task_list, n_readouts)
                ]

        def get_nested_len(task_list):
            if type(task_list[0]) is list:
                return [get_nested_len(t) for t in task_list]
            else:
                return len(task_list)

        if common_readout:
            n_readouts = config["model_params"]["n_readouts"] = get_nested_len(task)
            config["model_params"]["readout_from"] = get_nested_readout(
                task, n_readouts
            )
            config["model_params"]["agents_params"]["n_readouts"] = None

        else:
            config["model_params"]["n_readouts"] = None
            config["model_params"]["agents_params"]["n_readouts"] = len(task[0])

    elif task in ["both", "all", "none"]:

        config["model_params"]["agents_params"]["n_out"] = n_classes_per_ag

        if common_readout:
            config["model_params"]["n_readouts"] = n_agents
            config["model_params"]["agents_params"]["n_readouts"] = None
            config["model_params"]["readout_from"] = None
        else:
            config["model_params"]["n_readouts"] = None
            config["model_params"]["agents_params"]["n_readouts"] = n_agents
            config["model_params"]["readout_from"] = None

    else:
        warn(f"can't auto configure readout for task {task}")
        print(f" Warning ! Can't auto configure readout for task {task}")


def find_and_change(config, param_name, param_value):

    for key, value in config.items():
        if type(value) is dict:
            find_and_change(value, param_name, param_value)
        else:
            if key == param_name:
                config[key] = param_value


def find_varying_param(config, param_to_change=None):

    if param_to_change is None:
        param_to_change = config["varying_param"]

    for k, v in config.items():
        if type(v) is dict:
            return find_varying_param(v, param_to_change)
        else:
            if k == param_to_change and v is not None:
                return v


def get_new_config(config, key_prefix="config"):
    new_config = {}
    for k1, v1 in config.items():
        if type(v1) is dict:
            sub_config = get_new_config(v1, k1)
            new_config.update({key_prefix + "." + k: v for k, v in sub_config.items()})
        else:
            new_config[key_prefix + "." + k1] = v1
    return new_config
