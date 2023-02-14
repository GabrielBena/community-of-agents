import wandb
import pandas as pd
from tqdm import tqdm, trange
import json
import argparse
import multiprocessing as mp
import torch
import os, shutil
from functools import partial


from community.funcspec.single_model_loop import compute_all_metrics
from community.common.init import init_community, init_optimizers
from wandb_results_load import get_correct_artifact, get_pandas_from_json
from main_funcspec_single_param import get_data, ensure_config_coherence
from community.utils.configs import find_and_change
from community.common.models.readout import configure_readouts


def single_retrain(train_out, config, loaders):

    train_out = train_out[0]

    community = init_community(config["model"], device="cpu")

    community.load_state_dict(train_out["best_state"])
    test_accs = train_out["test_accs"]
    if len(test_accs.shape) == 1:
        best_test_acc = test_accs.max()
    else:
        best_test_acc = test_accs.max(0).mean()
    community.best_acc = best_test_acc

    # pbar_1.set_description(f"Recomputing Metrics, {community.nb_connections}")

    metric_data, metric_results = compute_all_metrics(
        community, loaders, config, device="cpu"
    )

    return metric_data, metric_results


def retrain_metric(run, training_artifact, metric_artifact, loaders, parallel=True):

    try:
        shutil.rmtree("artifacts/")
    except FileNotFoundError:
        pass

    config = run.config
    varying_params_sweep = config["varying_params_sweep"]
    ensure_config_coherence(config, varying_params_sweep)
    for param_name, param in varying_params_sweep.items():
        find_and_change(config, param_name, param)

    readout_config = configure_readouts(config)
    config["model"]["readout"].update(readout_config)

    new_metric_results = []
    new_metric_log = []

    if training_artifact is not None:

        training_results = torch.load(training_artifact.file())
        metric_results = get_pandas_from_json(metric_artifact)
        config["use_tqdm"] = 2

        max_size = 10

        pbar_1 = tqdm(range(max_size), position=1, desc="Recomputing Metrics")

        if not parallel:
            for train_out in zip(training_results, pbar_1):

                metric_data = single_retrain(train_out, config, loaders)

            new_metric_log.append(metric_data)

        else:
            pool = mp.Pool(mp.cpu_count())
            single_retrain_partial = partial(
                single_retrain, config=config, loaders=loaders
            )
            new_metric_log = pool.map(
                single_retrain_partial, zip(training_results, pbar_1)
            )

        final_data = pd.concat([pd.DataFrame.from_dict(d) for d in new_metric_log])

        return final_data, new_metric_results, new_metric_log


if __name__ == "__main__":
    """"""

    parser = argparse.ArgumentParser(
        description="Get data from Wandb Sweep and concatenate."
    )
    parser.add_argument(
        "-p1",
        "--sweep_path",
        default="m2snn/funcspec_V2/b275wyza",
        help="path of sweep to use",
    )

    parser.add_argument(
        "-p2",
        "--save_path",
        help="path to save gathered data",
        default="results/recomputed_metrics/",
    )

    parser.add_argument(
        "-N", "--max_size", default=3, help="max size of table to process", type=int
    )

    args = parser.parse_args()
    print(vars(args))

    save_name = args.save_path + f'/{args.sweep_path.split("/")[-1]}'

    api = wandb.Api(timeout=None)
    sweep = api.sweep(args.sweep_path)
    runs = sweep.runs

    # pool = mp.Pool(mp.cpu_count())

    if args.max_size is None:
        max_size = len(runs)
    else:
        max_size = args.max_size

    all_artifacts = [
        (
            get_correct_artifact(run, "training_results")[0],
            get_correct_artifact(run, "MetricResults"),
        )
        for run, _ in zip(runs, tqdm(range(max_size), desc="Getting Artifacts"))
    ]

    new_metric_dataframe = []

    for run, artifacts, _ in zip(
        runs, all_artifacts, tqdm(range(max_size), desc="Runs", position=0)
    ):

        # try:
        #    shutil.rmtree("artifacts/")
        # except FileNotFoundError:
        #    pass

        if not None in artifacts:
            config = run.config
            loaders, datasets = get_data(config)
            new_metric_data, *_ = retrain_metric(run, *artifacts, loaders)

            # print(new_metric_data.head())

            new_metric_dataframe.append(new_metric_data)

    final_data = pd.concat(new_metric_dataframe)
    final_data.to_pickle(save_name)
