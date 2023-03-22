import numpy as np
from community.utils.configs import get_all_v_params

import secrets
import string
import os
from pathlib import Path
import joblib
import json


def generate_id(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits."""
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


if __name__ == "__main__":

    # ----- Define Varyings Params -----
    varying_params = {
        "n_hidden": np.unique(np.geomspace(10, 100, 2, dtype=int)),
        "cov_ratio": np.unique(np.geomspace(0.01, 2).round(2)),
    }

    varying_params = {
        "n_hidden": [10, 100],
        "cov_ratio": [0, 1],
    }

    # ------ Create Sweep Id (length 8) ------

    sweep_id = generate_id()
    f_path = os.path.realpath(__file__)
    sweep_path = f"{os.path.split(f_path)[0]}/sweeps/{sweep_id}"
    path = Path(sweep_path)
    path.mkdir(exist_ok=True, parents=True)
    varying_params["sweep_id"] = [sweep_id]
    all_params = get_all_v_params(varying_params)

    joblib.dump(all_params, f"{sweep_path}/all_params")
    with open(f"{sweep_path}/varying_params", "w") as fp:
        json.dump(varying_params, fp)

    os.unlink(f"{os.path.split(f_path)[0]}/sweeps/latest")
    os.symlink(sweep_path, f"{os.path.split(f_path)[0]}/sweeps/latest")
