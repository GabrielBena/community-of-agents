from warnings import warn
from community.data.tasks import get_factors_list, get_task_family_dict
from community.utils.configs import configure_readouts, find_and_change
from community.utils.wandb_utils import mkdir_or_save_torch, update_dict
import torch
import torch.nn as nn
from torchvision import *
import pyaml
from yaml.loader import SafeLoader

import pandas as pd
import numpy as np

from community.data.datasets import get_datasets_alphabet, get_datasets_symbols
from community.data.tasks import get_task_target
from community.funcspec.single_model_loop import train_and_compute_metrics
import wandb

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_n


# warnings.filterwarnings('ignore')

if __name__ == "__main__":

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

    n_classes_per_digit = 10
    n_classes = n_classes_per_digit * n_digits

    dataset_config = {
        "batch_size": 512 if use_cuda else 256,
        "data_sizes": None if (not debug_run) else data_sizes // 10,
        "common_input": False,
        "use_cuda": use_cuda,
        "fix_asym": False,
        "permute_dataset": False,
        "seed": None,
        "data_type": "double_d",
        "n_classes": n_classes,
        "n_classes_per_digit": n_classes_per_digit,
    }

    if dataset_config["data_type"] == "symbols":

        print(f"Training for {n_classes} classes")

        symbol_config = {
            "data_size": data_sizes if not debug_run else data_sizes // 5,
            "nb_steps": 50,
            "n_symbols": n_classes - 1,
            "input_size": 60,
            "static": True,
            "symbol_type": "mod_5",
            "common_input": dataset_config["common_input"],
            "n_diff_symbols": n_digits,
            "parallel": False,
            "adjust_probas": False,
        }

        if symbol_config["static"]:
            symbol_config["nb_steps"] = 10

        dataset_config["input_size"] = symbol_config["input_size"] ** 2
        dataset_config["symbol_config"] = symbol_config

    else:
        if dataset_config["data_type"] in ["digits", "double_d", "single_d"]:

            dataset_config["n_classes_per_digit"] = 10
            dataset_config["n_classes"] = n_classes_per_digit * n_digits

        dataset_config["input_size"] = 784
        dataset_config["symbol_config"] = {}

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

    model_config = {
        "agents": agents_config,
        "connections": connections_config,
        "n_agents": n_agents,
        "n_ins": None,
        "common_readout": False,
        "n_readouts": 1,
        "readout_from": None,
        "readout_n_hid": None,
    }

    """
    varying_params = [
        [
            {"sparsity": s, "n_hidden": n}
            for s in np.array([1, 2, 10, n**2 // 2, n**2]) / n**2
        ]
        for n in np.linspace(10, 100, 5, dtype=int)
    ]
    try:
        varying_params = [v_p for v_params in varying_params for v_p in v_params]
    except TypeError:
        pass
    """

    config = {
        "model": model_config,
        "datasets": dataset_config,
        "optimization": {
            "agents": optim_config,
            "connections": deepR_config,
        },
        "training": {
            "decision": ["last", "all"],
            "n_epochs": 25 if not debug_run else 1,
            "inverse_task": False,
            "stopping_acc": 0.95,
            "early_stop": False,
            "force_connections": False,
        },
        "metrics": {"chosen_timesteps": ["mid-", "last"]},
        "sweep_params": {},
        "varying_params": {},
        ###------ Task ------
        "task": "parity-digits",
        ### ------ Task ------
        "metrics_only": False,
        "n_tests": 5 if not debug_run else 2,
        "debug_run": debug_run,
        "use_tqdm": 2,
    }

    try:
        os.environ["PBS_ARRAY_INDEX"]
        config["use_tqdm"] = False
    except KeyError:
        pass

    with open("latest_config.yml", "w") as config_file:
        pyaml.dump(config, config_file)

    if debug_run:
        os.environ["WANDB_MODE"] = "offline"
        pass

    wandb.init(project="funcspec_V2", entity="m2snn", config=config)
    # wandb.init(project="Funcspec", entity="gbena", config=config)

    run_dir = wandb.run.dir + "/"

    config["save_paths"] = {
        "training": run_dir + "training_results",
        "metrics": run_dir + "metric_results",
    }
    # WAndB tracking :

    sweep_params = wandb.config["sweep_params"]

    for param_name, param in sweep_params.items():
        wandb.define_metric(param_name)
        if param is not None:
            find_and_change(config, param_name, param)

    n = config["model"]["agents"]["n_hidden"]

    varying_params = [
        {"sparsity": s}
        for s in np.unique(
            np.array([1, 2, 10, n**2 // 100, n**2 // 10, n**2 // 2, n**2])
            / n**2
        )
    ]

    pbar0 = varying_params
    if config["use_tqdm"]:
        pbar0 = tqdm(pbar0, position=0, desc="Varying Params")

    for v_params in pbar0:
        
        if config['use_tqdm'] : 
            pbar0.set_description(f"Varying Params : {v_params}{sweep_params}")

        wandb.config.update({"varying_params": v_params}, allow_val_change=True)

        for param_name, param in v_params.items():
            wandb.define_metric(param_name)
            if param is not None:
                find_and_change(config, param_name, param)

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
            dataset_config["fix_asym"] = True

        if "n_classes_per_digit" in v_params:
            config["datasets"]["n_classes"] = (
                config["datasets"]["n_classes_per_digit"] * config["model"]["n_agents"]
            )
            config["datasets"]["symbol_config"]["n_symbols"] = (
                config["datasets"]["n_classes"] - 1
            )
        elif "n_classes" in varying_params:
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

        configure_readouts(config)

        metric_results, metric_datas, training_results = {}, [], []

        pbar = range(config["n_tests"])

        # print(
        #    f'Training {n_agents} agents of size {n_hidden} on task {task} using {"common"*common_readout + "separate"*(1-common_readout)} readout and decision {decision}, with {sparsity * n_hidden**2} connections'
        # )

        if config["use_tqdm"]:
            pbar = tqdm(pbar, desc="Trials : ", position=1)

        for test in pbar:

            if dataset_config["data_type"] == "symbols":
                loaders, datasets = get_datasets_symbols(
                    symbol_config,
                    dataset_config["batch_size"],
                    dataset_config["use_cuda"],
                )
            else:
                all_loaders = get_datasets_alphabet(
                    "data/",
                    dataset_config["batch_size"],
                    dataset_config["data_sizes"],
                    dataset_config["use_cuda"],
                    dataset_config["fix_asym"],
                    dataset_config["n_classes_per_digit"],
                )
                loaders = all_loaders[
                    ["multi", "double_d", "double_l", "single_d" "single_l"].index(
                        dataset_config["data_type"]
                    )
                ]

            # config = update_dict(config, wandb.config)

            (metric_data, train_outs, metric_result) = train_and_compute_metrics(
                config, loaders, device
            )

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
