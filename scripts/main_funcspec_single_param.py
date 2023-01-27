from os import mkdir
from community.utils.configs import find_and_change
from community.utils.wandb_utils import mkdir_or_save_torch
import torch
import numpy as np
import torch.nn as nn
from torchvision import *
import pyaml

from community.data.datasets.datasets import get_datasets_alphabet, get_datasets_symbols
from community.funcspec.single_model_loop import train_and_compute_metrics
import wandb
from tqdm import tqdm

# warnings.filterwarnings('ignore')

if __name__ == "__main__":

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    # n_classes = np.random.choice([2, 5, 10])
    n_classes = 50
    print(f"Training for {n_classes} classes")

    symbol_config = {
        "data_size": [30000, 5000],
        "nb_steps": 50,
        "n_symbols": n_classes - 1,
        "input_size": 50,
        "static": True,
        "symbol_type": "0",
        "double_data": False,
    }

    if symbol_config["static"]:
        symbol_config["nb_steps"] = 6
        symbol_config["data_size"] = [d * 2 for d in symbol_config["data_size"]]

    if not symbol_config["double_data"]:
        n_classes //= 2

    dataset_config = {
        "batch_size": 256,
        "use_cuda": use_cuda,
        "fix_asym": False,
        "permute_dataset": False,
        "seed": None,
        "data_type": "symbols",
        "n_classes": n_classes,
        "symbol_config": symbol_config,
    }

    if dataset_config["data_type"] == "symbols":
        loaders, datasets = get_datasets_symbols(
            symbol_config, dataset_config["batch_size"], dataset_config["use_cuda"]
        )

        dataset_config["input_size"] = symbol_config["input_size"] ** 2
    else:
        all_loaders = get_datasets_alphabet(
            "data/",
            dataset_config["batch_size"],
            dataset_config["use_cuda"],
            dataset_config["fix_asym"],
            dataset_config["permute_dataset"],
            dataset_config["seed"],
        )
        loaders = all_loaders[
            ["multi", "double_d", "double_l", "single_d" "single_l"].index(
                dataset_config["data_type"]
            )
        ]
        dataset_config["input_size"] = 784

    agents_params_dict = {
        "n_agents": 2,
        "n_in": dataset_config["input_size"],
        "n_ins": None,
        "n_hidden": 12,
        "n_layer": 1,
        "n_out": dataset_config["n_classes"],
        "train_in_out": (True, True),
        "use_readout": True,
        "cell_type": str(nn.RNN),
        "use_bottleneck": False,
        "ag_dropout": 0.0,
        "dual_readout": False,
    }

    p_masks = [0.1]

    lr, gamma = 1e-3, 0.95
    params_dict = {"lr": lr, "gamma": gamma}

    l1, gdnoise, lr, gamma, cooling = 1e-5, 1e-3, 1e-3, 0.95, 0.95
    deepR_params_dict = {
        "l1": l1,
        "gdnoise": gdnoise,
        "lr": lr,
        "gamma": gamma,
        "cooling": cooling,
    }

    p_cons_params = (1 / agents_params_dict["n_hidden"] ** 2, 0.999, 10)
    n_hids_params = (5, 50, 1)

    connections_params_dict = {
        "use_deepR": False,
        "global_rewire": False,
        "comms_dropout": 0.0,
        "sparsity": 0,
        "binarize": True,
        "comms_start": "mid",
    }

    config = {
        "model_params": {
            "agents_params": agents_params_dict,
            "connections_params": connections_params_dict,
            "common_readout": True,
            "dual_readout": True,
        },
        "datasets": dataset_config,
        "optimization": {
            "agents": params_dict,
            "connections": deepR_params_dict,
        },
        "training": {
            "decision_params": ("last", "both"),
            "n_epochs": 1,
            "inverse_task": False,
            "stopping_acc": 0.95,
            "early_stop": False,
            "force_connections": False,
        },
        "task": "both",
        "p_cons": p_cons_params,
        "n_hiddens": n_hids_params,
        "varying_param": "n_hidden",
        "metrics_only": False,
        "n_tests": 5,
    }

    if config["task"] == "both":
        if config["model_params"]["common_readout"]:
            assert config["model_params"]["dual_readout"]
            config["model_params"]["agents_params"]["dual_readout"] = False
        else:
            assert config["model_params"]["agents_params"]["dual_readout"]

    if config["varying_param"] == "sparsity":
        try:
            varying_params = (
                np.geomspace(p_cons_params[0], p_cons_params[1], p_cons_params[2])
                * agents_params_dict["n_hid"] ** 2
            ).round()
        except ValueError:
            varying_params = (
                np.linspace(p_cons_params[0], p_cons_params[1], p_cons_params[2])
                * agents_params_dict["n_hid"] ** 2
            ).round()

        varying_params = np.unique(varying_params / agents_params_dict["n_hid"] ** 2)

    elif config["varying_param"] == "n_hidden":
        varying_params = np.linspace(
            n_hids_params[0], n_hids_params[1], n_hids_params[2], dtype=int
        )

    else:
        raise ValueError(' config["varying_param"] should be "sparsity" or "n_hidden" ')

    # p_cons = [0.1]
    with open("latest_config.yml", "w") as config_file:
        pyaml.dump(config, config_file)

    # WAndB tracking :
    wandb.init(project="funcspec", entity="gbena", config=config)
    run_dir = wandb.run.dir + "/"

    config["save_paths"] = {
        "training": run_dir + "training_results",
        "metrics": run_dir + "metric_results",
    }

    wandb.config.update(config)
    wandb.define_metric(config["varying_param"])

    # metric_names = ['Correlation', 'Masks', 'Bottleneck']
    # metric_names = ['Correlation', 'Bottleneck']
    metric_names = ["Mean_Corr", "Base_corr", "Bottleneck"]

    metric_results = {metric: {} for metric in metric_names}
    training_results, all_results = {}, {}

    pbar = tqdm(varying_params, position=0, leave=None)
    for p, v_param in enumerate(pbar):

        pbar.set_description(f'Community {config["varying_param"]} : {v_param} ')

        find_and_change(config, v_param)
        # wandb.config.update(config, allow_val_change=True)

        metrics, train_out, results = train_and_compute_metrics(config, loaders, device)
        training_results[v_param] = train_out
        all_results[v_param] = results
        for metric in metric_names:
            metric_results[metric][v_param] = metrics[metric]

    for name, file in zip(
        ["training_results", "metric_results", "all_results"],
        [training_results, metric_results, all_results],
    ):
        mkdir_or_save_torch(file, name, run_dir)
        artifact = wandb.Artifact(name=name, type="dict")
        artifact.add_file(run_dir + name)
        wandb.log_artifact(artifact)

    # wandb.log_artifact(run_dir + 'training_results', name='training_results')
    # wandb.log_artifact(run_dir + 'metric_results', name='metric_results')
