from warnings import warn
import torch
import torch.nn as nn
from torchvision import *
import pyaml
from yaml.loader import SafeLoader

import pandas as pd
import numpy as np

from community.utils.configs import (
    copy_and_change_config,
    ensure_config_coherence,
)
from community.utils.wandb_utils import mkdir_or_save, update_dict
from community.data.datasets.generate import get_datasets_alphabet, get_datasets_symbols
from community.data.tasks import get_task_target
from community.funcspec.single_model_loop import train_and_compute_metrics
from community.common.models.readout import configure_readouts
import wandb

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_n


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

    wandb_log = False

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # print(f"Training on {device}")

    n_agents = 2
    n_digits = n_agents

    data_sizes = np.array([60000, 10000])

    n_classes_per_digit = 10
    n_classes = n_classes_per_digit * n_digits

    use_symbols = False

    dataset_config = {
        "batch_size": 512 if use_cuda else 256,
        "data_size": None if (not debug_run) else data_sizes // 10,
        "common_input": True,
        "use_cuda": use_cuda,
        "fix_asym": False,
        "permute_dataset": True,
        "seed": seed,
        "data_type": "symbols" if use_symbols else "double_d",
        "n_digits": n_digits,
        "n_classes": n_classes,
        "n_classes_per_digit": n_classes_per_digit,
        "nb_steps": 2,
        "split_classes": False,
        "cov_ratio": 1.0,
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
            "random_transform": True,
            "cov_ratio": dataset_config["cov_ratio"],
        }

        if not symbol_config["static"]:
            symbol_config["nb_steps"] = 50

        dataset_config["input_size"] = symbol_config["input_size"] ** 2
        dataset_config["symbol_config"] = symbol_config

    else:
        dataset_config["input_size"] = 784 * (1 + dataset_config["common_input"])
        dataset_config["symbol_config"] = {}

    p_masks = [0.1]

    optim_config = {
        "lr": 1e-3,
        "gamma": 0.92,
        "reg_readout": None,
    }

    deepR_config = {
        "l1": 1e-5,
        "gdnoise": 1e-3,
        "lr": 1e-3,
        "gamma": 0.95,
        "cooling": 0.95,
        "global_rewire": False,
    }

    connections_config = {
        "use_deepR": False,
        "global_rewire": False,
        "comms_dropout": 0.0,
        "sparsity": 0.1,
        "binarize": False,
        "comms_start": "start",
        "comms_out_scale": 1,
    }

    agents_config = {
        "n_in": dataset_config["input_size"],
        "n_hidden": 20,
        "n_bot": 5,
        "n_layers": 1,
        "train_in_out": (True, True),
        "cell_type": str(nn.RNN),
        "ag_dropout": 0.0,
    }

    readout_config = {
        "common_readout": True,
        "n_hid": None,
        "readout_from": None,
    }

    model_config = {
        "agents": agents_config,
        "connections": connections_config,
        "readout": readout_config,
        "n_agents": n_agents,
    }

    default_config = {
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
            "check_gradients": False,
        },
        "metrics": {"chosen_timesteps": ["mid-", "last"]},
        "varying_params_sweep": {},
        "varying_params_local": {},
        ###------ Task ------
        "task": "parity-digits",
        ### ------ Task ------
        "metrics_only": False,
        "n_tests": 5 if not debug_run else 1,
        "debug_run": debug_run,
        "use_tqdm": 2,
        "data_regen": [False, dataset_config["data_type"] != "symbols"],
        "wandb_log": wandb_log,
    }

    try:
        os.environ["PBS_ARRAY_INDEX"]
        default_config["use_tqdm"] = False
    except KeyError:
        pass

    with open("latest_config.yml", "w") as config_file:
        pyaml.dump(default_config, config_file)

    if debug_run:
        os.environ["WANDB_MODE"] = "offline"
        pass

    # WAndB tracking :
    wandb.init(project="funcspec_V2", entity="m2snn", config=default_config)
    # wandb.init(project="Funcspec", entity="gbena", config=config)

    run_dir = wandb.run.dir + "/"

    if wandb_log:
        save_path = run_dir
    else:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        save_path = f"{dir_path}/../wandb_results/"
        if wandb.run.sweep_id:
            save_path += f"sweeps/{wandb.run.sweep_id}/"
        else:
            save_path += "runs/"

        save_path += f"{wandb.run.id}/"

    default_config["save_path"] = save_path

    varying_params_sweep = wandb.config["varying_params_sweep"]
    # Adjust Parameters based on wandb sweep
    default_config = copy_and_change_config(default_config, varying_params_sweep)
    ensure_config_coherence(default_config, varying_params_sweep)

    n = default_config["model"]["agents"]["n_hidden"]

    sparsities = np.concatenate(
        [
            np.array([0]),
            np.unique(np.geomspace(1, n**2, 10, endpoint=True, dtype=int)) / n**2,
        ]
    )

    varying_params_local = [
        {"sparsity": s}
        for s in np.unique(
            (np.geomspace(1, n**2, 2) / n**2) if debug_run else sparsities
        )
    ]

    # varying_params_local = [{"cov_ratio": c} for c in [0, 0.5, 1]]

    pbar_0 = varying_params_local
    if default_config["use_tqdm"]:
        pbar_0 = tqdm(pbar_0, position=0, desc="Varying Params", leave=None)

    metric_results, metric_datas, training_results = {}, [], []

    # Go through local sweep
    for v, v_params_local in enumerate(pbar_0):

        v_params_all = v_params_local.copy()
        # Sweep param always overrides
        v_params_all.update(wandb.config["varying_params_sweep"])
        config = copy_and_change_config(default_config, v_params_all)
        ensure_config_coherence(config, v_params_all)

        config["varying_params_local"] = v_params_all

        if config["use_tqdm"]:
            pbar_0.set_description(f"Varying Params : {v_params_all}")

        if wandb_log:

            wandb.config.update(
                {"varying_params_local": v_params_local}, allow_val_change=True
            )

        if v == 0:
            loaders, datasets = get_data(config)
        elif config["data_regen"][0]:
            seed += 1
            config["datasets"]["seed"] = seed
            loaders, datasets = get_data(config)

        readout_config = configure_readouts(config)
        config["model"]["readout"].update(readout_config)

        pbar_1 = range(config["n_tests"])

        if config["use_tqdm"]:
            pbar_1 = tqdm(pbar_1, desc="Trials : ", position=1, leave=None)

        # Repetitions per varying parameters
        for test in pbar_1:

            if config["data_regen"][1] and test != 0:
                seed += 1
                config["datasets"]["seed"] = seed
                loaders, datasets = get_data(config)

            # config = update_dict(config, wandb.config)

            (
                metric_data,
                train_outs,
                metric_result,
                all_results,
            ) = train_and_compute_metrics(config, loaders, device)

            metric_data["seed"] = np.full_like(list(metric_data.values())[0], seed)

            metric_datas.append(metric_data)
            training_results.append(train_outs)

            for m_name, metric in metric_result.items():
                metric_results.setdefault(m_name, [])
                metric_results[m_name].append(metric)

    final_table = pd.concat([pd.DataFrame.from_dict(d) for d in metric_datas])
    if wandb_log:
        data_table = wandb.Table(dataframe=final_table)
        wandb.log({"Metric Results": data_table})

    try:
        metric_results = {k: np.stack(v, -1) for k, v in metric_results.items()}
    except ValueError:
        pass

    if wandb_log:

        final_log = {}

        try:
            retraining_global_diff = final_table["retraining_global_diff"].mean()
            final_log["retraining_global_diff"] = retraining_global_diff

            correlations_global_diff = final_table["correlations_global_diff"].mean()
            final_log["correlations_global_diff"] = correlations_global_diff

            ablations_global_diff = final_table["ablations_global_diff"].mean()
            final_log["ablations_global_diff"] = ablations_global_diff

        except KeyError:
            pass

        best_test_acc = final_table["best_acc"].mean()
        final_log["best_test_acc"] = best_test_acc

        retraining_det = final_table["retraining_det"].mean()
        final_log["retraining_det"] = retraining_det

        wandb.log(final_log)

    for name, file, save_mode in zip(
        ["metric_results", "metric_table", "training_results"],
        [all_results, final_table],
        ["torch", "pickle", "torch"],
    ):
        mkdir_or_save(file, name, default_config["save_path"], save_mode)

        if wandb_log and save_mode == "torch":
            artifact = wandb.Artifact(name=name, type="dict")
            artifact.add_file(default_config["save_path"] + name)
            wandb.log_artifact(artifact)

    wandb.finish()
