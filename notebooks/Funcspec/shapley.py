import torch.multiprocessing as mp
from torch.multiprocessing import Pool, set_start_method
import torch
import numpy as np
from copy import deepcopy

from community.common.decision import get_decision
from community.common.training import get_loss, get_acc
from community.data.process import process_data
from community.data.tasks import get_task_target
from community.common.init import init_community
from community.common.models.readout import configure_readouts
from community.data.datasets.generate import get_datasets_symbols

from tqdm import tqdm
from msapy import msa
from itertools import repeat
from functools import partial
import time
import os

import yaml
from yaml.loader import SafeLoader


def masked_inference(excluded, community, data, t_target, decision):

    excluded = np.array(excluded, dtype=int)

    common_readout = community.readout is not None

    n_hid, n_agents = community.agent_dims[1], community.n_agents
    masked_units = [
        excluded[(i * n_hid <= excluded) * (excluded < (i + 1) * n_hid)] - i * n_hid
        for i in range(n_agents)
    ]

    """

    f_community = deepcopy(community).to("cpu")

    for ag in range(n_agents):
        for param in [
            f"agents.{ag}.cell.weight_ih_l0",
            f"agents.{ag}.cell.weight_hh_l0",
            f"agents.{ag}.cell.bias_ih_l0",
            f"agents.{ag}.cell.bias_hh_l0",
        ]:

            p = f_community.get_parameter(param).data
            p[masked_units[ag]] = 0

            if "weight_hh" in param:
                p[:, masked_units[ag]] = 0
        try:
            p = f_community.get_parameter(f"connections.{ag}{1-ag}.weight").data
            p[:, masked_units[ag]] = 0

            p = f_community.get_parameter(f"connections.{1-ag}{ag}.weight").data
            p[masked_units[ag]] = 0
        except AttributeError:
            pass
            
    """

    # data, target = next(iter(loaders[1]))
    # data, target = data.to('cpu'), target.to('cpu')

    state_masks = [
        np.array([i not in mu for i in np.arange(n_hid)]) for mu in masked_units
    ]
    output, *_ = community(data, state_masks=state_masks)
    output, deciding_ags = get_decision(output, *decision, target=t_target)
    loss = get_loss(output, t_target)

    pred = output.argmax(dim=-1)
    correct = pred.eq(t_target.view_as(pred))
    acc = (
        (correct.sum(-1) * np.prod(t_target.shape[:-1]) / t_target.numel())
        .cpu()
        .data.numpy()
    )

    # if not common_readout:
    # acc = acc[int(task)]

    return float(acc)


def get_data(task, loaders, n_classes_per_digit, symbols, common_input):

    data, target = [], []
    for (d, t), _ in zip(loaders[1], range(4)):
        data.append(d)
        target.append(t)

    data, target = torch.cat(data), torch.cat(target)
    # data, target = datasets[1].data[0][:512], datasets[1].data[1][:512]
    data, target = process_data(
        data, target, task, symbols=symbols, common_input=common_input
    )
    t_target = get_task_target(target, task, n_classes_per_digit)

    return data.float(), t_target


def compute_shapley_values(
    community,
    nodes,
    n_permutations,
    datasets,
    config,
    parallel=False,
):

    permutation_space = msa.make_permutation_space(
        elements=nodes, n_permutations=n_permutations
    )
    combination_space = msa.make_combination_space(permutation_space=permutation_space)
    complement_space = msa.make_complement_space(
        combination_space=combination_space, elements=nodes
    )
    input(f"Number of combinations to go through : {len(complement_space)}, Continue ?")

    n_classes_per_digit = config["datasets"]["n_classes_per_digit"]
    symbols = config["datasets"]["data_type"] == "symbols"
    common_readout = community.readout is not None
    common_input = config["datasets"]["common_input"]

    torch.set_num_threads(1)
    community.to("cpu")

    all_accs = []

    for task in ["0", "1"]:

        if common_readout:
            decision = ["last", task]
        else:
            decision = ["last", f"max_{task}"]

        data, t_target = get_data(
            task, datasets, n_classes_per_digit, symbols, common_input
        )

        n_processes = mp.cpu_count()
        # set_start_method('spawn', force=True)
        pool = mp.Pool(n_processes)

        masked_inf = partial(
            masked_inference,
            community=community,
            data=data,
            t_target=t_target,
            decision=decision,
        )

        print(f"Task {task} : Performance without ablations : {masked_inf([])}")

        if parallel:

            try:
                all_accs.append(
                    pool.map(
                        masked_inf,
                        tqdm(complement_space),
                    )
                )
            except KeyboardInterrupt:
                exit()
            finally:
                pool.terminate()
                pool.join()
        else:
            all_accs.append(
                list(
                    map(
                        masked_inf,
                        tqdm(complement_space),
                    )
                )
            )

    contributions = [dict(zip(combination_space, accs)) for accs in all_accs]
    lesion_effects = [dict(zip(complement_space, accs)) for accs in all_accs]

    shapley_tables = [
        msa.get_shapley_table(contributions=c, permutation_space=permutation_space)
        for c in contributions
    ]
    shapley_tables = [shap[nodes] for shap in shapley_tables]
    shapley_tables_avg = [shap.mean() for shap in shapley_tables]

    return shapley_tables, shapley_tables_avg, all_accs, contributions, lesion_effects


if __name__ == "__main__":

    with open("latest_config.yml", "r") as config_file:
        config = yaml.load(config_file, SafeLoader)

    n_agents = 2
    n_classes_per_digit = 10
    n_classes = n_agents * n_classes_per_digit
    batch_size = 512
    use_cuda = torch.cuda.is_available()

    data_config = {
        "data_size": np.array([batch_size, batch_size]),
        "nb_steps": 50,
        "n_symbols": n_classes - 1,
        "symbol_type": "mod_5",
        "input_size": 60,
        "static": True,
        "common_input": True,
        "n_diff_symbols": 2,
        "parallel": False,
        "adjust_probabilites": False,
    }

    task = config["task"] = "both"

    config["model"]["agents"]["n_in"] = data_config["input_size"] ** 2
    n_hidden = config["model"]["agents"]["n_hidden"] = 10
    config["model"]["n_agents"] = n_agents

    common_readout = config["model"]["readout"]["common_readout"] = True
    config["model"]["readout"]["n_hid"] = None
    config["model"]["readout"]["readout_from"] = None

    config["datasets"]["n_classes"] = n_classes
    config["datasets"]["n_classes_per_digit"] = n_classes_per_digit
    config["datasets"]["symbol_config"]["n_diff_symbols"] = n_agents

    readout_config = configure_readouts(config)
    config["model"]["readout"].update(readout_config)

    config["model"]["connections"]["sparsity"] = 1 / n_hidden**2  # .005
    config["model"]["connections"]["comms_out_scale"] = 1
    config["model"]["connections"]["comms_start"] = "start"
    config["model"]["connections"]["binarize"] = False

    decision = config["training"]["decision"] = [
        "last",
        "both" if common_readout else "0",
    ]
    n_epochs = config["training"]["n_epochs"] = 15

    community = init_community(config["model"])
    loaders, datasets = get_datasets_symbols(
        data_config, batch_size, use_cuda, plot=True
    )
    try:

        saved_results = torch.load("saves/results")
        train_results = saved_results[str(config)]
        community.load_state_dict(train_results["best_state"])
    except (KeyError, FileNotFoundError) as e:
        pass

    nodes = list(
        np.arange(n_agents * config["model"]["agents"]["n_hidden"]).astype(str)
    )
    n_permutations = 100

    (
        shapley_tables,
        shapley_tables_avg,
        all_accs,
        contributions,
        lesion_effects,
    ) = compute_shapley_values(
        community,
        nodes,
        n_permutations,
        datasets,
        config,
    )
