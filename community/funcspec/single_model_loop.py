from re import S
import wandb
import torch
import pandas as pd

from community.funcspec.masks import train_and_get_mask_metric
from community.funcspec.bottleneck import readout_retrain
from community.funcspec.correlation import get_pearson_metrics
from community.common.training import train_community
from community.common.init import init_community, init_optimizers
from community.utils.configs import get_training_dict, find_varying_param
import numpy as np
import copy
from time import sleep


def init_and_train(config, loaders, device):

    use_wandb = wandb.run is not None

    agents_params_dict = config["model_params"]["agents_params"]
    connections_params_dict = config["model_params"]["connections_params"]

    deepR_params_dict = config["optimization"]["connections"]
    params_dict = config["optimization"]["agents"]

    for v_param_name, v_param in wandb.config["varying_params"].items():
        if use_wandb:
            wandb.log({v_param_name: v_param})

            if v_param_name == "sparsity":
                wandb.log({"q_measure": (1 - v_param) / (2 * (1 + v_param))})

    # ------  Train ------

    train_outs = {}
    trained_coms = {}
    for use_bottleneck in [False]:

        agents_params_dict["use_bottleneck"] = use_bottleneck
        community = init_community(config["model_params"], device)
        print(
            community.nb_connections,
            community.agents[0].dims,
            community.use_common_readout,
        )
        optimizers, schedulers = init_optimizers(
            community, params_dict, deepR_params_dict
        )
        optimizers[0] = torch.optim.Adam(community.parameters(), lr=1e-3)

        if not config["metrics_only"]:

            training_dict = get_training_dict(config)
            train_out = train_community(
                community,
                *loaders,
                optimizers,
                schedulers,
                config=training_dict,
                device=device,
                use_tqdm=1,
            )

            test_accs = train_out["test_accs"]
            if len(test_accs.shape) == 1:
                best_test_acc = np.max(test_accs)
            else:
                best_test_acc = np.max(test_accs, 0).mean()

            mean_d_ags = train_out["deciding_agents"].mean()

            metric_names = [
                "Best Test Acc" + "_bottleneck" * use_bottleneck,
                "Mean Decision" + "_bottleneck" * use_bottleneck,
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
            train_outs[f'With{(1-use_bottleneck)*"out"} Bottleneck'] = train_out
            trained_coms[f'With{(1-use_bottleneck)*"out"} Bottleneck'] = copy.deepcopy(
                community
            )

    return trained_coms, train_outs


def compute_all_metrics(trained_coms, loaders, config, device):

    symbols = config["datasets"]["data_type"] == "symbols"
    deepR_params_dict = config["optimization"]["connections"]
    n_classes = config["datasets"]["n_classes"]
    chosen_timesteps = config["metrics"]["chosen_timesteps"]

    metric_names = ["mean_corr", "bottleneck"]

    community = trained_coms["Without Bottleneck"]
    # print('Correlations')
    correlations_results = get_pearson_metrics(
        community,
        loaders,
        device=device,
        use_tqdm=1,
        symbols=symbols,
        chosen_timesteps=chosen_timesteps,
    )
    mean_corrs, relative_corrs, base_corrs = list(
        correlations_results.values()
    )  # n_agents x n_targets x n_timesteps

    # print('Weight Masks')
    # masks_metric = {}
    # masks_results = train_and_get_mask_metric(community, 0.1, loaders, device=device, n_tests=1, n_epochs=1, use_tqdm=1, use_optimal_sparsity=True, symbols=symbols)
    # masks_props, masks_accs, _, masks_states, masks_spars = list(masks_results.values())
    # masks_metric, masks_accs, masks_spars = masks_props.mean(0), masks_accs.mean(0).max(-1), masks_spars.mean(0)

    community = trained_coms["Without Bottleneck"]
    # print('Bottlenecks Retrain')
    bottleneck_results = readout_retrain(
        community,
        loaders,
        n_classes,
        deepR_params_dict=deepR_params_dict,
        n_tests=1,
        n_epochs=3,
        device=device,
        use_tqdm=1,
        symbols=symbols,
        chosen_timesteps=chosen_timesteps,
    )
    bottleneck_metric = bottleneck_results["accs"]  # n_agents n_targets x n_timesepts

    # ------ Log ------
    # metrics = [correlations_metric, masks_metric, bottleneck_metric]
    # metric_names = ['Correlation', 'Masks', 'Bottleneck']
    # all_results = [correlations_results, masks_results, bottleneck_results]

    metrics = [mean_corrs, bottleneck_metric]
    all_results = [correlations_results, bottleneck_results]

    metric_results = {
        metric_name: metric for metric, metric_name in zip(metrics, metric_names)
    }
    all_metric_results = {
        metric_name: metric for metric, metric_name in zip(all_results, metric_names)
    }

    metric_data, metric_log = define_and_log(
        metric_results, wandb.config, community.best_acc
    )

    return metric_data, metric_log, all_metric_results


def define_and_log(metrics, config, best_acc):

    diff_metric = lambda metric: (metric[0] - metric[1]) / (metric[0] + metric[1])
    global_diff_metric = (
        lambda metric: np.abs(diff_metric(metric[0]) - diff_metric(metric[1])) / 2
    )
    """
    for varying_param in config['varying_params'].keys() : 
        wandb.define_metric('metric_*', step_metric=varying_param)
    """

    metric_data = {}
    metric_log = {}

    for step in range(1, 3):

        metric_data.setdefault("Step", [])
        metric_data["Step"].append(step)

        metric_data.setdefault("best_acc", [])
        metric_data["best_acc"].append(best_acc)

        for v_param_name, v_param in config["varying_params"].items():
            metric_data.setdefault(v_param_name, [])
            metric_data[v_param_name].append(v_param)

        for metric_name, metric in metrics.items():

            step_single_metrics = metric[..., step]
            ag_diff_metrics = []

            for ag in range(2):
                ag_single_metrics = step_single_metrics[ag]
                ag_diff_metric = diff_metric(ag_single_metrics)
                metric_log[
                    f"metric_{metric_name}_diff_ag_{ag}_step_{step}"
                ] = ag_diff_metric
                for task in range(2):
                    metric_log[
                        f"metric_{metric_name}_ag_{ag}_dig_{task}_step_{step}"
                    ] = ag_single_metrics[task]
                ag_diff_metrics.append(ag_diff_metric)

            community_diff_metric = global_diff_metric(step_single_metrics)
            metric_log[
                f"metric_{metric_name}_global_diff_step_{step}"
            ] = community_diff_metric

            metric_data.setdefault(metric_name + "_global_diff", [])
            metric_data[metric_name + "_global_diff"].append(community_diff_metric)

    return metric_log, metric_data
    wandb.log(metric_log)
    table = wandb.Table(dataframe=pd.DataFrame.from_dict(metric_data))
    wandb.log({"Metric Results": table})

    """
    metric_data = {}

    for ag in range(2) : 
        metric_data.setdefault('Agent', [])
        for step in range(3) : 
            metric_data.setdefault('Step', [])
            for target in range(2) : 
                metric_data.setdefault('Target', [])

                metric_data['Agent'].append(ag)
                metric_data['Step'].append(step)
                metric_data['Target'].append(target)

                for v_param_name, v_param in config['varying_params'].items() :
                    metric_data.setdefault(v_param_name, [])
                    metric_data[v_param_name].append(v_param)

                for metric_name, metric in metrics.items() : 
                    metric_data.setdefault(metric_name, [])
                    try : 
                        metric_data[metric_name].append(metric[ag, target, step])
                    except IndexError : 
                            metric_data[metric_name].append(metric[ag, step])
   
    metric_data_diff = {}
    for ag in range(2) : 
        metric_data_diff.setdefault('Agent', [])
        for step in range(3) : 
            metric_data_diff.setdefault('Step', [])

            metric_data_diff['Agent'].append(ag)
            metric_data_diff['Step'].append(step)

            for v_param_name, v_param in config['varying_params'].items() :
                metric_data_diff.setdefault(v_param_name, [])
                metric_data_diff[v_param_name].append(v_param)

            for metric_name, metric in metrics.items() : 
                metric_data_diff.setdefault(metric_name + '_diff', [])
                try : 
                    metric_data_diff[metric_name+ '_diff'].append(diff_metric(metric[ag, :, step]))
                except IndexError : 
                    metric_data_diff.pop(metric_name+ '_diff', None)

    """

    # table_diff = wandb.Table(dataframe = pd.DataFrame.from_dict(metric_data_diff))

    # wandb.log({'Metric Results Diff' : table_diff})


def train_and_compute_metrics(config, loaders, device):

    # ------ Train ------

    trained_coms, train_outs = init_and_train(config, loaders, device)

    # ------ Metrics ------

    metric_data, metric_log, all_metric_results = compute_all_metrics(
        trained_coms, loaders, config, device
    )

    return metric_data, metric_log, train_outs, all_metric_results
