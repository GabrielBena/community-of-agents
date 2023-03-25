import numpy as np
from community.utils.configs import get_all_v_params

import secrets
import string
import os
from pathlib import Path

import joblib
import yaml
import json
import pickle
from filelock import Timeout, FileLock
import time


def generate_id(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits."""
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_sweep(varying_params, d_path):
    # ------ Create Sweep Id (length 8) ------

    sweep_id = generate_id()
    sweep_path = f"{d_path}/sweeps/{sweep_id}"
    path = Path(sweep_path)
    path.mkdir(exist_ok=True, parents=True)
    varying_params["sweep_id"] = [sweep_id]
    all_params = get_all_v_params(varying_params)

    # joblib.dump(all_params, f"{sweep_path}/all_params")
    save_params(f"{sweep_path}/all_params", all_params)
    save_params(f"{sweep_path}/all_params_init", all_params)

    with open(f"{sweep_path}/varying_params", "w") as fp:
        yaml.dump(varying_params, fp)

    with open(f"{d_path}/latest", "w") as fp:
        json.dump(sweep_id, fp)

    print(sweep_id)


def load_params(path):
    with open(path, "r") as f:
        # Read each line and parse the JSON string back into a dictionary

        return [json.loads(line) for line in f]


def save_params(path, all_params):
    with open(path, "w") as f:
        # Iterate over list of dictionaries and write each one on a new line
        for d in all_params:
            f.write(json.dumps(d) + "\n")

        f.flush()
        os.fsync(f.fileno())


def get_config_manual_lock(sweep_path, run_id):
    lock = FileLock(f"{sweep_path}/all_params.lock")
    with lock:
        all_configs = load_params(f"{sweep_path}/all_params")
        for config in all_configs:
            try:
                config["run_id"]
            except KeyError:
                config["run_id"] = run_id
                save_params(f"{sweep_path}/all_params", all_configs)
                time.sleep(0.1)
                return config

    return


if __name__ == "__main__":

    # ----- Define Varyings Params -----
    varying_params = {
        "n_bot": [None, 5],
        "common_input": [False, True],
        "common_readout": [False, True],
        "n_hidden": np.unique(np.geomspace(10, 100, 25, dtype=int)).tolist(),
        "cov_ratio": np.unique(np.linspace(0.01, 1, 25).round(2)).tolist(),
    }

    # varying_params = {
    #    "n_hidden": [10, 100],
    #    "cov_ratio": [0, 1],
    # }

    f_path = os.path.realpath(__file__)
    d_path = os.path.split(f_path)[0]

    generate_sweep(varying_params, d_path)
