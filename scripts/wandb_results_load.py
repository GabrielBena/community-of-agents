import wandb
from community.utils.wandb_utils import get_wandb_artifact, get_wandb_runs
import pandas as pd
from tqdm import tqdm
import json
import argparse


def get_correct_artifact(run, wanted_name="MetricResults"):
    try:
        artifact = [
            a
            for a in run.logged_artifacts()
            if wanted_name in a.name and run.state == "finished"
        ][0]
    except IndexError:
        artifact = None
    return artifact


def get_pandas_from_json(artifact, run):
    if artifact is None:
        return None
    else:
        with open(artifact.file(), "r") as j:
            contents = json.loads(j.read())
            data = pd.DataFrame(contents["data"], columns=contents["columns"])
        names = [run.name] * len(contents["data"])
        data["name"] = names

        return data


def get_all_data_and_save(sweep_path, save_path, max_size=None):

    ## Get Results
    api = wandb.Api(timeout=None)
    sweep = api.sweep(sweep_path)
    runs = sweep.runs

    if max_size is None:
        max_size = len(runs)

    pandas = [
        get_pandas_from_json(get_correct_artifact(run), run)
        for run, i in zip(runs, tqdm(range(max_size)))
    ]
    if pandas is not None:
        total_data = pd.concat(pandas)

    return total_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get data from Wandb Sweep and concatenate."
    )
    parser.add_argument(
        "-p1",
        "--sweep_path",
        default="gbena/funcspec/j3uqaib0",
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

    args = parser.parse_args()
    print(vars(args))

    total_data = get_all_data_and_save(**vars(args))
    print(total_data.head())

    total_data.to_pickle(args.save_path + f'/{args.sweep_path.split("/")[-1]}')
