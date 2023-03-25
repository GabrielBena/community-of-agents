from copy import deepcopy
from itertools import product


def get_all_v_params(varying_params, excluded_params={}):
    return [
        {
            k: p
            for k, p in zip(varying_params.keys(), params)
            if k not in excluded_params
        }
        for params in product(*varying_params.values())
    ]


def get_training_dict(config):

    training_dict = {
        "n_epochs": config["training"]["n_epochs"],
        "task": config["task"],
        "global_rewire": config["optimization"]["connections"]["global_rewire"],
        "check_gradients": config["training"]["check_gradients"],
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


def find_and_change(config, param_name, param_value):

    for key, value in config.items():
        if type(value) is dict:
            find_and_change(value, param_name, param_value)
        else:
            if key == param_name:
                config[key] = param_value

    return config


def copy_and_change_config(config, varying_params):

    config = deepcopy(config)
    for n, v in varying_params.items():
        find_and_change(config, n, v)

    return config


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

    if "parity" in config["task"] and False:
        if config["datasets"]["cov_ratio"] == 1:
            config["datasets"]["fix_asym"] = True
        else:
            config["datasets"]["fix_asym"] = False

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
    if "common_input" in v_params:
        if config["datasets"]["data_type"] != "symbols":

            config["datasets"]["input_size"] = 784 * (
                1 + config["datasets"]["common_input"]
            )
            config["model"]["agents"]["n_in"] = config["datasets"]["input_size"]

    if config["model"]["readout"]["common_readout"]:

        config["training"]["decision"][1] = "all"

        """
        if config["datasets"]["common_input"]:

            if config["training"]["decision"][1] == "both":
                config["training"]["decision"][1] = "all"
        else:
            if config["training"]["decision"][1] == "all":
                config["training"]["decision"][1] = "both"
        """
    else:
        config["training"]["decision"][1] = "max"

    if "cov_ratio" in v_params:
        config["datasets"]["symbol_config"]["cov_ratio"] = config["datasets"][
            "cov_ratio"
        ]
        config["data_regen"][0] = True

    if config["datasets"]["data_type"] == "symbols":
        config["datasets"]["input_size"] = (
            config["datasets"]["symbol_config"]["input_size"] ** 2
        )
    else:
        config["datasets"]["input_size"] = 784 * (
            1 + config["datasets"]["common_input"]
        )
        if "digits" in config["datasets"]["data_type"]:
            config["datasets"]["n_classes_per_digit"] = min(
                10, config["datasets"]["n_classes_per_digit"]
            )
            config["datasets"]["n_classes"] = (
                config["datasets"]["n_classes_per_digit"] * config["model"]["n_agents"]
            )

    config["model"]["n_in"] = config["datasets"]["input_size"]
