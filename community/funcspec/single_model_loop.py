from re import S
import wandb
import torch
import pandas as pd
import numpy as np
import numpy.linalg as LA

from community.funcspec.masks import train_and_get_mask_metric
from community.funcspec.bottleneck import readout_retrain
from community.funcspec.correlation import get_pearson_metrics
from community.common.training import train_community
from community.common.init import init_community, init_optimizers
from community.utils.configs import get_training_dict, _finditem
import copy
from time import sleep
import warnings


def init_and_train(config, loaders, device):

    use_wandb = wandb.run is not None

    agents_params_dict = config["model"]["agents"]
    connections_params_dict = config["model"]["connections"]

    deepR_params_dict = config["optimization"]["connections"]
    params_dict = config["optimization"]["agents"]

    use_tqdm = config["use_tqdm"]

    # Check varying parameters :
    varying_params_all = wandb.config["varying_params_local"].copy()
    varying_params_all.update(wandb.config["varying_params_sweep"])

    sentinel = object()

    for v_param_name, v_param in varying_params_all.items():
        found = _finditem(config, v_param_name, sentinel)
        if found is sentinel:
            warnings.warn("Parameter {v_param_name} couldn't be found ! ")
        else:
            assert (
                found == v_param
            ), f"{v_param_name} is different ({found}) than expected ({v_param}) !"
            if use_wandb:
                wandb.log({v_param_name: v_param})
                if v_param_name == "sparsity":
                    wandb.log({"q_measure": (1 - v_param) / (2 * (1 + v_param))})
    # ------  Train ------

    train_outs = {}
    trained_coms = {}
    # for use_bottleneck in [True, False]:

    community = init_community(config["model"], device)

    optimizers, schedulers = init_optimizers(community, params_dict, deepR_params_dict)

    if not config["metrics_only"]:

        training_dict = get_training_dict(config)
        train_out = train_community(
            community,
            *loaders,
            optimizers,
            schedulers,
            config=training_dict,
            device=device,
            use_tqdm=use_tqdm,
        )

        test_accs = train_out["test_accs"]
        if len(test_accs.shape) == 1:
            best_test_acc = np.max(test_accs)
        else:
            best_test_acc = np.max(test_accs, 0).mean()

        mean_d_ags = train_out["deciding_agents"].mean()

        metric_names = [
            "Best Test Acc",  # + "_bottleneck" * use_bottleneck,
            "Mean Decision",  # + "_bottleneck" * use_bottleneck,
        ]
        for metric, metric_name in zip([best_test_acc, mean_d_ags], metric_names):

            try:
                for m, sub_metric in enumerate(metric):
                    if use_wandb:
                        wandb.define_metric(metric_name + f"_{m}")
                    if use_wandb:
                        wandb.define_metric(metric_name + f"_{m}")
                    if use_wandb:
                        wandb.log({metric_name + f"_{m}": sub_metric})
            except:
                if use_wandb:
                    wandb.define_metric(metric_name)
                if use_wandb:
                    wandb.define_metric(metric_name)
                if use_wandb:
                    wandb.log({metric_name: metric})

        community.best_acc = best_test_acc

        # train_outs[f'With{(1-use_bottleneck)*"out"} Bottleneck'] = train_out
        # trained_coms[f'With{(1-use_bottleneck)*"out"} Bottleneck'] = copy.deepcopy(
        #    community
        # )

        train_outs, trained_coms = train_out, copy.deepcopy(community)

    return trained_coms, train_outs


def compute_all_metrics(trained_coms, loaders, config, device):

    symbols = config["datasets"]["data_type"] == "symbols"
    deepR_params_dict = config["optimization"]["connections"]
    n_classes = config["datasets"]["n_classes_per_digit"]
    chosen_timesteps = config["metrics"]["chosen_timesteps"]

    n_agents = config["model"]["n_agents"]
    try:
        n_digits = config["datasets"]["symbol_config"]["n_diff_symbols"]
    except KeyError:
        n_digits = 2

    use_tqdm = config["use_tqdm"]

    # community = trained_coms["Without Bottleneck"]

    """
    # print('Correlations')
    correlations_results = get_pearson_metrics(
        community,
        loaders,
        device=device,
        use_tqdm=1 if use_tqdm else False,
        symbols=symbols,
        chosen_timesteps=chosen_timesteps,
    )
    mean_corrs, relative_corrs, base_corrs = list(
        correlations_results.values()
    )  # n_agents x n_targets x n_timesteps
    """
    # print('Weight Masks')
    # masks_metric = {}
    # masks_results, masked_coms = train_and_get_mask_metric(community, 0.1, loaders, device=device, n_tests=1, n_epochs=1, use_tqdm=1, use_optimal_sparsity=True, symbols=symbols)
    # masks_props, masks_accs, _, masks_states, masks_spars = list(masks_results.values())
    # masks_metric, masks_accs, masks_spars = masks_props.mean(0), masks_accs.mean(0).max(-1), masks_spars.mean(0)

    community = trained_coms  # ["Without Bottleneck"]
    # print('Bottlenecks Retrain')
    bottleneck_results, _ = readout_retrain(
        community,
        loaders,
        config,
        n_epochs=1,
        device=device,
        use_tqdm=use_tqdm,
        chosen_timesteps=chosen_timesteps,
        n_hid=30,
        common_input=config["datasets"]["common_input"],
    )
    bottleneck_metric = bottleneck_results["accs"]  # n_agents n_targets x n_timesepts

    # ------ Log ------
    # metrics = [correlations_metric, masks_metric, bottleneck_metric]
    # metric_names = ['Correlation', 'Masks', 'Bottleneck']
    # all_results = [correlations_results, masks_results, bottleneck_results]

    metric_names = ["bottleneck"]
    metrics = [bottleneck_metric]

    metric_results = {
        metric_name: metric for metric, metric_name in zip(metrics, metric_names)
    }

    metric_data = define_and_log(metric_results, wandb.config, community.best_acc)

    return metric_data, metric_results


def define_and_log(metrics, config, best_acc):

    varying_params_all = wandb.config["varying_params_local"].copy()
    varying_params_all.update(wandb.config["varying_params_sweep"])

    diff_metric = lambda metric: (metric[0] - metric[1]) / (metric[0] + metric[1])
    global_diff_metric = (
        lambda metric: np.abs(diff_metric(metric[0]) - diff_metric(metric[1])) / 2
    )

    metric_ts = config["metrics"]["chosen_timesteps"]
    metric_data = {}

    for step, ts in enumerate(metric_ts):

        metric_data.setdefault("Step", [])
        metric_data["Step"].append(ts)

        metric_data.setdefault("best_acc", [])
        metric_data["best_acc"].append(best_acc)

        for v_param_name, v_param in varying_params_all.items():
            metric_data.setdefault(v_param_name, [])
            metric_data[v_param_name].append(v_param)

        for metric_name, metric in metrics.items():

            try:
                step_single_metrics = metric[..., step]

                metric_data.setdefault(metric_name + "_det", [])
                metric_data.setdefault(metric_name + "_det_col_norm", [])

                metric_data[metric_name + "_det"].append(
                    np.abs(LA.det(step_single_metrics))
                )

                metric_data[metric_name + "_det_col_norm"].append(
                    np.abs(LA.det(step_single_metrics))
                    / step_single_metrics.sum(0).prod()
                )

                for norm in [1, 2, "fro", "nuc"]:

                    metric_data.setdefault(metric_name + f"_norm_{norm}", [])
                    metric_data[metric_name + f"_norm_{norm}"].append(
                        LA.norm(step_single_metrics, norm)
                    )

                if step_single_metrics.shape[0] == 2:
                    community_diff_metric = global_diff_metric(step_single_metrics)

                    metric_data.setdefault(metric_name + "_global_diff", [])
                    metric_data[metric_name + "_global_diff"].append(
                        community_diff_metric
                    )

            except TypeError:
                continue

    return metric_data


def train_and_compute_metrics(config, loaders, device):

    # ------ Train ------

    trained_coms, train_outs = init_and_train(config, loaders, device)

    # ------ Metrics ------

    metric_data, metric_results = compute_all_metrics(
        trained_coms, loaders, config, device
    )

    return metric_data, train_outs, metric_results
