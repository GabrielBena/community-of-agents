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
from community.common.manual_sweeps import get_config_manual_lock
import wandb

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_n

import joblib

import secrets
import string
import sys
import json
import argparse


def get_data(config):

    if config["datasets"]["data_type"] == "symbols":
        loaders, datasets = get_datasets_symbols(
            config["datasets"],
            config["datasets"]["batch_size"],
            config["datasets"]["use_cuda"],
        )
    else:
        all_loaders = get_datasets_alphabet("data/", config["datasets"])
        datasets, loaders = all_loaders[config["datasets"]["data_type"]]

    return loaders, datasets


def generate_id(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits."""
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


# warnings.filterwarnings('ignore')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Main Training",
        description="Train and compute metrics, run sweeps",
    )

    parser.add_argument(
        "-l",
        "--wandb_log",
        help="Log the run in wandb",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "-m",
        "--manual_sweep",
        help="Do manual sweep with locked files",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Do a debug run with limited data and iterations",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-v",
        "--varying_params_sweep",
        default={},
        help="Varying params passed by wandb agent for sweep",
        type=dict,
    )

    parser.add_argument(
        "-s",
        "--use_symbols",
        help="Use Symbol Generation",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    print(args)

    f_path = os.path.realpath(__file__)
    dir_path = os.path.split(f_path)[0]
    sweep_id = None

    wandb_log = args.wandb_log
    manual_sweep = args.manual_sweep

    try:
        seed = int(os.environ["PBS_ARRAY_INDEX"])
        hpc = True
    except KeyError:
        seed = np.random.randint(100)
        hpc = False

    # Use for debugging
    debug_run = args.debug
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

    use_symbols = args.use_symbols

    dataset_config = {
        "batch_size": 512 if use_cuda else 256,
        "data_size": None if (not debug_run) else data_sizes // 10,
        "common_input": False,
        "use_cuda": use_cuda,
        "fix_asym": False,
        "permute_dataset": True,
        "seed": seed,
        "data_type": "symbols" if use_symbols else "double_digits",
        "n_digits": n_digits,
        "n_classes": n_classes,
        "n_classes_per_digit": n_classes_per_digit,
        "nb_steps": 3,
        "split_classes": True,
        "cov_ratio": 1.0,
    }

    symbol_config = {
        "data_size": data_sizes if not debug_run else data_sizes // 5,
        "input_size": 60,
        "static": True,
        "symbol_type": "mod_5",
        "parallel": False,
        "adjust_probas": False,
        "random_transform": True,
    }

    if not symbol_config["static"]:
        symbol_config["nb_steps"] = 50

    dataset_config["symbol_config"] = symbol_config

    if dataset_config["data_type"] == "symbols":
        dataset_config["input_size"] = symbol_config["input_size"] ** 2
    else:
        dataset_config["input_size"] = 784 * (1 + dataset_config["common_input"])
        if dataset_config["data_type"] == "double_d":
            dataset_config["n_classes"] = min(10, dataset_config["n_classes"])
            pass

    p_masks = [0.1]

    optim_config = {
        "lr": 1e-3,
        "gamma": 0.95,
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
        "comms_start": "last",
        "comms_out_scale": 1,
    }

    agents_config = {
        "n_in": dataset_config["input_size"],
        "n_hidden": 20,
        "n_bot": None,
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

    training_config = {
        "decision": ["last", "all"],
        "n_epochs": 50 if not debug_run else 1,
        "inverse_task": False,
        "stopping_acc": 0.95,
        "early_stop": False,
        "force_connections": False,
        "check_gradients": False,
    }

    default_config = {
        "model": model_config,
        "datasets": dataset_config,
        "optimization": {
            "agents": optim_config,
            "connections": deepR_config,
        },
        "training": training_config,
        "metrics": {"chosen_timesteps": ["0", "mid-", "last"]},
        "varying_params_sweep": {},
        "varying_params_local": {},
        ###------ Task ------
        "task": "family",
        ### ------ Task ------
        "metrics_only": False,
        "n_tests": 10 if not debug_run else 1,
        "debug_run": debug_run,
        "use_tqdm": 2,
        "data_regen": [False, False],  # dataset_config["data_type"] != "symbols"],
        "wandb_log": wandb_log,
        "sweep_id": None,
        "hpc": hpc,
    }

    with open("latest_config.yml", "w") as config_file:
        pyaml.dump(default_config, config_file)

    if debug_run:
        os.environ["WANDB_MODE"] = "offline"
        pass

    if not manual_sweep:

        # WAndB tracking :
        wandb.init(project="funcspec_V2", entity="m2snn", config=default_config)
        varying_params_sweep = wandb.config["varying_params_sweep"]
        # Adjust Parameters based on wandb sweep

        default_config = copy_and_change_config(default_config, varying_params_sweep)
        ensure_config_coherence(default_config, varying_params_sweep)

    else:

        os.environ["WANDB_MODE"] = "offline"

        if sweep_id is None:
            with open(f"{dir_path}/manual_sweeps/latest") as fp:
                sweep_id = json.load(fp)

        sweep_path = f"{dir_path}/manual_sweeps/sweeps/{sweep_id}"
        run_id = generate_id()
        # Manually retreive parameter of sweep

        load = True
        varying_params_sweep, load = get_config_manual_lock(sweep_path, run_id)

        if varying_params_sweep is None:
            print("sweep done")
            quit()

        sweep_id = varying_params_sweep["sweep_id"]
        default_config = copy_and_change_config(default_config, varying_params_sweep)
        ensure_config_coherence(default_config, varying_params_sweep)
        default_config["varying_params_sweep"] = varying_params_sweep

        wandb.init(
            project="funcspec_V2",
            entity="m2snn",
            config=default_config,
            id=run_id,
            group=sweep_id,
        )

    # ------ Save Path ------
    if wandb_log and not manual_sweep:
        save_path = wandb.run.dir + "/"
    else:

        dir_path = os.path.dirname(os.path.abspath(__file__))
        save_path = f"{dir_path}/../wandb_results/"

        if not manual_sweep and wandb.run.sweep_id:
            save_path += f"sweeps/{wandb.run.sweep_id}/"
        elif default_config["sweep_id"]:
            save_path += f"sweeps/{default_config['sweep_id']}/"
        else:
            save_path += "runs/"

        save_path += f"{wandb.run.id}/"

    default_config["save_path"] = save_path

    # ------ Local Varying Params ------
    n = default_config["model"]["agents"]["n_hidden"]
    sparsities = np.concatenate(
        [
            np.array([0]),
            np.unique(
                (np.geomspace(1, n**2, 30, endpoint=True, dtype=int) / n**2).round(
                    4
                )
            ),
        ]
    )

    varying_params_local = [
        {"sparsity": s}
        for s in np.unique(
            (np.geomspace(1, n**2, 1) / n**2) if debug_run else sparsities
        )
    ]

    # ------- Do Local Sweep ------
    pbar_0 = varying_params_local
    if default_config["use_tqdm"]:
        pbar_0 = tqdm(pbar_0, position=0, desc="Varying Params", leave=None)

    metric_results, metric_datas, training_results = {}, [], []

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
        for name, file, save_mode in zip(
            ["metric_table"],
            [final_table],
            ["pickle"],
        ):
            mkdir_or_save(file, name, default_config["save_path"], save_mode)

            

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

    if manual_sweep:
        varying_params_sweep, load = get_config_manual_lock(
            sweep_path, run_id, mark_as_done=True
        )
        #Rerun 
        os.execv(sys.executable, ["python"] + sys.argv)
