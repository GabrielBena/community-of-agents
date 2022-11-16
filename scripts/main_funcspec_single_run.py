from warnings import warn
from community.data.tasks import get_factors_list, get_task_family_dict
from community.utils.configs import configure_readouts, find_and_change
from community.utils.wandb_utils import mkdir_or_save_torch, update_dict
import torch
import torch.nn as nn
from torchvision import *
import pyaml
import pandas as pd
import numpy as np

from community.data.datasets import get_datasets_alphabet, get_datasets_symbols
from community.funcspec.single_model_loop import train_and_compute_metrics
import wandb

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_n


# warnings.filterwarnings('ignore')

if __name__ == "__main__":

    # Use for debugging
    test_run = False

    if test_run:
        print("Debugging Mode is activated ! Only doing mock training")

    use_cuda = True
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    n_agents = 2
    n_digits = n_agents

    n_classes_per_digit = 30
    n_classes = n_classes_per_digit * n_digits

    print(f"Training for {n_classes} classes")

    symbol_config = {
        "data_size": [30000, 5000],
        "nb_steps": 50,
        "n_symbols": n_classes - 1,
        "input_size": 60,
        "static": True,
        "symbol_type": "mod_5",
        "common_input": True,
        "n_diff_symbols": n_digits,
        "parallel": False,
    }

    if symbol_config["static"]:
        symbol_config["nb_steps"] = 6
        symbol_config["data_size"] = [
            d if not test_run else d // 5 for d in symbol_config["data_size"]
        ]

    dataset_config = {
        "batch_size": 256,
        "use_cuda": use_cuda,
        "fix_asym": False,
        "permute_dataset": False,
        "seed": None,
        "data_type": "symbols",
        "n_classes": n_classes,
        "n_classes_per_digit": n_classes_per_digit,
        "symbol_config": symbol_config,
    }

    if dataset_config["data_type"] == "symbols":

        dataset_config["input_size"] = symbol_config["input_size"] ** 2
    else:
        dataset_config["input_size"] = 784

    agents_params_dict = {
        "n_in": dataset_config["input_size"],
        "n_hidden": 15,
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
    params_dict = {
        "lr": lr,
        "gamma": gamma,
        "reg_readout": None,
    }

    l1, gdnoise, lr, gamma, cooling = 1e-5, 1e-3, 1e-3, 0.95, 0.95
    deepR_params_dict = {
        "l1": l1,
        "gdnoise": gdnoise,
        "lr": lr,
        "gamma": gamma,
        "cooling": cooling,
        "global_rewire": False,
    }

    connections_params_dict = {
        "use_deepR": False,
        "comms_dropout": 0.0,
        "sparsity": 0,
        "binarize": True,
        "comms_start": "mid",
    }

    model_params_dict = {
        "agents_params": agents_params_dict,
        "connections_params": connections_params_dict,
        "n_agents": n_agents,
        "n_ins": None,
        "common_readout": True,
        "n_readouts": 1,
        "readout_from": None,
    }

    config = {
        "model_params": model_params_dict,
        "datasets": dataset_config,
        "optimization": {
            "agents": params_dict,
            "connections": deepR_params_dict,
        },
        "training": {
            "decision_params": ("last", "all"),
            "n_epochs": 30 if not test_run else 1,
            "inverse_task": False,
            "stopping_acc": 0.95,
            "early_stop": False,
            "force_connections": False,
        },
        "metrics": {"chosen_timesteps": ["0", "mid-", "last"]},
        "varying_params": {},
        "task": "all",  # "family",
        "metrics_only": False,
        "n_tests": 20 if not test_run else 2,
        "test_run": test_run,
        "use_tqdm": False,
    }

    with open("latest_config.yml", "w") as config_file:
        pyaml.dump(config, config_file)

    if test_run:
        os.environ["WANDB_MODE"] = "offline"

    wandb.init(project="funcspec_V2", entity="m2snn", config=config)

    run_dir = wandb.run.dir + "/"

    config["save_paths"] = {
        "training": run_dir + "training_results",
        "metrics": run_dir + "metric_results",
    }

    # WAndB tracking :
    varying_params = wandb.config["varying_params"]

    for param_name, param in varying_params.items():
        wandb.define_metric(param_name)
        if param is not None:
            find_and_change(config, param_name, param)

    if config["task"] == "shared_goals":
        task = config["task"] = [[str(i), str((i + 1) % 3)] for i in range(n_agents)]

    configure_readouts(config)

    metric_results, metric_datas, training_results = {}, [], []

    pbar = range(config["n_tests"])
    if config["use_tqdm"]:
        pbar = tqdm(pbar, desc="Trials : ")

    for test in pbar:

        if dataset_config["data_type"] == "symbols":
            loaders, datasets = get_datasets_symbols(
                symbol_config, dataset_config["batch_size"], dataset_config["use_cuda"]
            )
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
