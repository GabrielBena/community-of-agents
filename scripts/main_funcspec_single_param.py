from warnings import warn
import torch
import torch.nn as nn
from torchvision import *
import pyaml
from yaml.loader import SafeLoader

import pandas as pd
import numpy as np

from community.data.tasks import get_factors_list, get_task_family_dict
from community.utils.configs import (
    configure_readouts,
    find_and_change,
    ensure_config_coherence,
)
from community.utils.wandb_utils import mkdir_or_save_torch, update_dict
from community.data.datasets import get_datasets_alphabet, get_datasets_symbols
from community.data.tasks import get_task_target
from community.funcspec.single_model_loop import train_and_compute_metrics
import wandb

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_n


# warnings.filterwarnings('ignore')

if __name__ == "__main__":

    try:
        seed = int(os.environ["PBS_ARRAY_INDEX"])
    except KeyError:
        seed = np.random.randint(100)

    # Use for debugging
    debug_run = False
    if debug_run:
        print("Debugging Mode is activated ! Only doing mock training")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # print(f"Training on {device}")

    n_agents = 2
    n_digits = n_agents

    data_sizes = np.array([60000, 10000])

    n_classes_per_digit = 16
    n_classes = n_classes_per_digit * n_digits

    dataset_config = {
        "batch_size": 512 if use_cuda else 256,
        "data_sizes": None if (not debug_run) else data_sizes // 10,
        "common_input": True,
        "use_cuda": use_cuda,
        "fix_asym": False,
        "permute_dataset": True,
        "seed": seed,
        "data_type": "symbols",
        "n_classes": n_classes,
        "n_classes_per_digit": n_classes_per_digit,
        "nb_steps": 2,
        "split_classes": False,
    }

    if dataset_config["data_type"] == "symbols":

        print(f"Training for {n_classes} classes")

        symbol_config = {
            "data_size": data_sizes if not debug_run else data_sizes // 5,
            "nb_steps": dataset_config["nb_steps"],
            "n_symbols": n_classes - 1,
            "input_size": 60,
            "static": True,
            "symbol_type": "mod_5",
            "n_diff_symbols": n_digits,
            "parallel": False,
            "adjust_probas": False,
        }

        if not symbol_config["static"]:
            symbol_config["nb_steps"] = 50

        dataset_config["input_size"] = symbol_config["input_size"] ** 2
        dataset_config["symbol_config"] = symbol_config

    else:
        dataset_config["input_size"] = 784 * (1 + dataset_config["common_input"])
        dataset_config["symbol_config"] = {}

    p_masks = [0.1]

    lr, gamma = 1e-3, 0.95
    optim_config = {
        "lr": lr,
        "gamma": gamma,
        "reg_readout": None,
    }

    l1, gdnoise, lr, gamma, cooling = 1e-5, 1e-3, 1e-3, 0.95, 0.95
    deepR_config = {
        "l1": l1,
        "gdnoise": gdnoise,
        "lr": lr,
        "gamma": gamma,
        "cooling": cooling,
        "global_rewire": False,
    }

    connections_config = {
        "use_deepR": False,
        "comms_dropout": 0.0,
        "sparsity": 0.01,
        "binarize": False,
        "comms_start": "start",
        "comms_out_scale": 0.1,
    }

    agents_config = {
        "n_in": dataset_config["input_size"],
        "n_hidden": 20,
        "n_layers": 1,
        "n_out": n_classes_per_digit,
        "n_readouts": 1,
        "train_in_out": (True, True),
        "cell_type": str(nn.RNN),
        "use_bottleneck": False,
        "ag_dropout": 0.0,
    }

    model_config = {
        "agents": agents_config,
        "connection": connections_config,
        "common_readout": True,
        "dual_readout": True,
    }

    config = {
        "model": model_config,
        "datasets": dataset_config,
        "optimization": {
            "agents": optim_config,
            "connections": deepR_config,
        },
        "training": {
            "decision": ["last", "all"],
            "n_epochs": 30 if not debug_run else 1,
            "inverse_task": False,
            "stopping_acc": 0.95,
            "early_stop": False,
            "force_connections": False,
        },
        "metrics": {"chosen_timesteps": ["mid-", "last"]},
        "varying_params_sweep": {
            # "common_input": False,
            # "common_readout": True,
            # "use_bottleneck": True,
        },
        "varying_params_local": {},
        ###------ Task ------
        "task": "bitxor",
        ### ------ Task ------
        "metrics_only": False,
        "n_tests": 10 if not debug_run else 1,
        "debug_run": debug_run,
        "use_tqdm": 2,
        "data_regen": dataset_config["data_type"] != "symbols",
    }

    try:
        os.environ["PBS_ARRAY_INDEX"]
        config["use_tqdm"] = False
    except KeyError:
        pass

    with open("latest_config.yml", "w") as config_file:
        pyaml.dump(config, config_file)

    # WAndB tracking :
    wandb.init(project="funcspec", entity="gbena", config=config)
    run_dir = wandb.run.dir + "/"

    if debug_run:
        os.environ["WANDB_MODE"] = "offline"
        pass

    # WAndB tracking :

    wandb.init(project="funcspec_V2", entity="m2snn", config=config)
    # wandb.init(project="Funcspec", entity="gbena", config=config)

    run_dir = wandb.run.dir + "/"

    config["save_paths"] = {
        "training": run_dir + "training_results",
        "metrics": run_dir + "metric_results",
    }


    varying_params_sweep = wandb.config["varying_params_sweep"]

    # Adjust Parameters based on wandb sweep
    for param_name, param in varying_params_sweep.items():
        wandb.define_metric(param_name)
        if param is not None:
            find_and_change(config, param_name, param)

    n = config["model"]["agents"]["n_hidden"]

    varying_params_local = [
        {"sparsity": s}
        for s in np.unique(
            (np.geomspace(1, n**2, 2) / n**2)
            if debug_run
            else (np.geomspace(1, n**2, 25) / n**2)
        )
    ]

    # varying_params_local = [{"use_bottleneck": t} for t in [True, False]]

    ensure_config_coherence(config, varying_params_sweep)

    def get_data(config):

        if config["datasets"]["data_type"] == "symbols":
            loaders, datasets = get_datasets_symbols(
                config["datasets"],
                config["datasets"]["batch_size"],
                config["datasets"]["use_cuda"],
            )
        else:
            all_loaders = get_datasets_alphabet("data/", config["datasets"])
            loaders = all_loaders[
                ["multi", "double_d", "double_l", "single_d" "single_l"].index(
                    config["datasets"]["data_type"]
                )
            ]
            datasets = [l.dataset for l in loaders]

        return loaders, datasets

    loaders, datasets = get_data(config)

    pbar_0 = varying_params_local
    if config["use_tqdm"]:
        pbar_0 = tqdm(pbar_0, position=0, desc="Varying Params", leave=None)

    metric_results, metric_datas, training_results = {}, [], []

    # Go through local sweep
    for v, v_params_local in enumerate(pbar_0):

        v_params_all = v_params_local.copy()
        v_params_all.update(wandb.config["varying_params_sweep"])

        for param_name, param in v_params_local.items():
            wandb.define_metric(param_name)
            if param is not None:
                find_and_change(config, param_name, param)

        if config["use_tqdm"]:
            pbar_0.set_description(f"Varying Params : {v_params_all}")

        wandb.config.update(
            {"varying_params_local": v_params_local}, allow_val_change=True
        )

        ensure_config_coherence(config, v_params_all)
        configure_readouts(config)

        pbar_1 = range(config["n_tests"])

        if config["use_tqdm"]:
            pbar_1 = tqdm(pbar_1, desc="Trials : ", position=1, leave=None)

        # Repetitions per varying parameters
        for test in pbar_1:

            if config["data_regen"] and test * v != 0:
                seed += 1
                config["datasets"]["seed"] = seed
                loaders, datasets = get_data(config)

            # config = update_dict(config, wandb.config)

            (metric_data, train_outs, metric_result) = train_and_compute_metrics(
                config, loaders, device
            )

            metric_data["seed"] = np.full_like(list(metric_data.values())[0], seed)

            metric_datas.append(metric_data)
            training_results.append(train_outs)

            for m_name, metric in metric_result.items():
                metric_results.setdefault(m_name, [])
                metric_results[m_name].append(metric)

    final_data = pd.concat([pd.DataFrame.from_dict(d) for d in metric_datas])
    data_table = wandb.Table(dataframe=final_data)
    wandb.log({"Metric Results": data_table})

     try:
        metric_results = {k: np.stack(v, -1) for k, v in metric_results.items()}
    except ValueError:
        pass

    final_log = {}

    try:
        bottleneck_global_diff = final_data["bottleneck_global_diff"].mean()
        final_log["bottleneck_global_diff"] = bottleneck_global_diff
    except KeyError:
        pass

    best_test_acc = final_data["best_acc"].mean()
    final_log["best_test_acc"] = best_test_acc

    bottleneck_det = final_data["bottleneck_det"].mean()
    final_log["bottleneck_det"] = bottleneck_det

    wandb.log(final_log)

    for name, file in zip(
        ["training_results", "metric_results"],
        [training_results, metric_results],
    ):
        mkdir_or_save_torch(file, name, run_dir)
        artifact = wandb.Artifact(name=name, type="dict")
        artifact.add_file(run_dir + name)
        wandb.log_artifact(artifact)

    wandb.finish()