import wandb
from community.utils.wandb_utils import get_wandb_artifact, get_wandb_runs
import pandas as pd
from tqdm import tqdm, trange
import json
import argparse
import multiprocessing as mp


def get_correct_artifact(run, wanted_name="MetricResults"):

    try:
        artifact = [
            a
            for a in run.logged_artifacts()
            if wanted_name in a.name and run.state == "finished"
        ][0]
    except IndexError:
        artifact = None

    return artifact, run.name


def get_pandas_from_json(art):

    artifact, name = art
    if artifact is None:
        return None
    else:
        with open(artifact.file(), "r") as j:
            contents = json.loads(j.read())
            data = pd.DataFrame(contents["data"], columns=contents["columns"])

        names = [name] * len(contents["data"])
        data["name"] = names

        return data


def get_df(run):
    return get_pandas_from_json(get_correct_artifact(*run))


def get_all_data_and_save(sweep_path, save_path, max_size=None, reload=False):

    api = wandb.Api(timeout=None)
    sweep = api.sweep(sweep_path)
    runs = sweep.runs

    pool = mp.Pool(mp.cpu_count())

    if max_size is None:
        max_size = len(runs)

    if not reload:

        try:
            existing_results = pd.read_pickle(
                save_path + f'/{sweep_path.split("/")[-1]}'
            )
            print("Existing Results Loaded")
            arts = []
            for run, _ in zip(runs, tqdm(range(max_size))):
                if run.name not in existing_results["name"].values:
                    arts.append(get_correct_artifact(run))
                else:
                    print("All new results Loaded")
                    break

        except FileNotFoundError:
            existing_results = None

            arts = [
                get_correct_artifact(run[0]) for run in zip(runs, tqdm(range(max_size)))
            ]
    else:
        existing_results = None

        arts = [
            get_correct_artifact(run[0]) for run in zip(runs, tqdm(range(max_size)))
        ]

    dfs = pool.map(get_pandas_from_json, tqdm(arts))

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
        "--sweep_path",
        default="m2snn/funcspec_V2/1ldo81ej",
        help="path of sweep to use",
    )

    parser.add_argument(
        "-p2",
        "--save_path",
        help="path to save gathered data",
        default="results/sweep_tables/",
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

    args = parser.parse_args()
    print(vars(args))

    total_data = get_all_data_and_save(**vars(args))
    print(total_data.head())
    print(total_data.shape)

    total_data.to_pickle(args.save_path + f'/{args.sweep_path.split("/")[-1]}')
