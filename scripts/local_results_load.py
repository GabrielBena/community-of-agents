import pandas as pd
import json
import argparse
import torch
import multiprocessing as mp
import os
from functools import partial
from tqdm import tqdm
from pathlib import Path


def get_pandas_from_json(run, sweep_path, name="metric_table"):

    save = f"{sweep_path}/{run}/{name}"

    with open(save, "r") as j:
        contents = json.loads(j.read())
        data = pd.DataFrame(contents["data"], columns=contents["columns"])

    names = [run] * len(contents["data"])
    data["name"] = names

    return data


def get_pandas_from_csv(run, sweep_path, name="metric_table"):

    save = f"{sweep_path}/{run}/{name}"
    data = pd.read_csv(save)
    names = [run] * len(data)
    data["name"] = names

    return data


def get_pandas_from_pickle(run, sweep_path, name="metric_table"):

    save = f"{sweep_path}/{run}/{name}"
    data = pd.read_pickle(save)
    names = [run] * len(data)
    data["name"] = names

    return data


def get_all_data_and_save(
    sweep_id,
    save_path,
    max_size=None,
    reload=False,
    name="metric_table",
    format="pickle",
):

    pool = mp.Pool(mp.cpu_count())

    sweep_path = f"/mnt/storage/gb21/wandb_results/sweeps/{sweep_id}/"
    runs = os.listdir(sweep_path)
    if max_size is None:
        max_size = len(runs)
    save_name = save_path + f"/{name}_{sweep_id}"

    if not reload:
        try:
            existing_results = pd.read_pickle(save_name)
        except FileNotFoundError:
            existing_results = None
    else:
        existing_results = None

    if name == "metric_table":
        if format == "json":
            partial_load = partial(get_pandas_from_json, sweep_path=sweep_path)
            dfs = pool.map(partial_load, tqdm(runs))
        elif format == "csv":
            partial_load = partial(get_pandas_from_csv, sweep_path=sweep_path)
            dfs = pool.map(partial_load, tqdm(runs))
        elif format == "pickle":
            partial_load = partial(get_pandas_from_pickle, sweep_path=sweep_path)
            dfs = pool.map(partial_load, tqdm(runs))
    else:
        dfs = None

    # dfs = pool.map(get_df, zip(tqdm(range(max_size)), [sweep_path] * max_size))

    if dfs is not None:
        total_data = pd.concat(dfs)

        if existing_results is not None:
            total_data = pd.concat([existing_results, total_data])

        return total_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get data from Wandb Sweep and concatenate."
    )
    parser.add_argument(
        "-p1",
        "--sweep_id",
        default="0s9fg2jm",
        help="path of sweep to use",
    )

    parser.add_argument(
        "-p2",
        "--save_path",
        help="path to save gathered data",
        default="/mnt/storage/gb21/wandb_results/compiled/",
    )
    parser.add_argument(
        "-N", "--max_size", default=None, help="max size of table to process", type=int
    )

    parser.add_argument(
        "-r",
        "--reload",
        default=False,
        help="Reload results or load existing ones",
        type=bool,
    )

    parser.add_argument(
        "-n", "--name", default="metric_table", help="Artifact name", type=str
    )

    parser.add_argument(
        "-f", "--format", default="csv", help="Format to load", type=str
    )

    args = parser.parse_args()
    print(vars(args))

    total_data = get_all_data_and_save(**vars(args))

    if args.name == "metric_table":
        print(total_data.head())
        print(total_data.shape)

    path = Path(args.save_path + f'/{args.sweep_id.split("/")[-1]}/')
    path.mkdir(exist_ok=True, parents=True)

    save_name = args.save_path + f'/{args.sweep_id.split("/")[-1]}/{args.name}'

    try:
        total_data.to_pickle(save_name)
    except AttributeError:
        torch.save(total_data, save_name)
