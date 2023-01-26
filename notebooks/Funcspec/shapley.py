import torch.multiprocessing as mp
from torch.multiprocessing import Pool, set_start_method
import torch
import numpy as np
from copy import deepcopy
from community.common.decision import get_decision
from community.common.training import get_loss
from community.data.process import process_data
from community.data.tasks import get_task_target
from tqdm import tqdm
from msapy import msa
from itertools import repeat
from functools import partial
import time


def masked_inference(
    excluded, community, data, t_target, decision, task, common_readout
):

    excluded = np.array(excluded, dtype=int)

    n_hid, n_agents = community.agent_dims[1], community.n_agents
    masked_units = [
        excluded[(i * n_hid < excluded) * (excluded < (i + 1) * n_hid)] - i * n_hid
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
    loss, tt_target, output = get_loss(output, t_target)

    pred = output.argmax(dim=-1)
    correct = pred.eq(tt_target.view_as(pred))
    acc = (
        (correct.sum(-1) * np.prod(tt_target.shape[:-1]) / tt_target.numel())
        .cpu()
        .data.numpy()
    )
    if not common_readout:
        acc = acc[int(task)]

    return acc


def get_data(task, datasets, n_classes_per_digit):
    data, target = datasets[1].data[0][:512], datasets[1].data[1][:512]
    data, target = process_data(data, target, task, symbols=True)
    t_target = get_task_target(target, task, n_classes_per_digit)

    return data, t_target


def compute_shapley_values(
    community,
    nodes,
    n_permutations,
    datasets,
    config,
):

    permutation_space = msa.make_permutation_space(
        elements=nodes, n_permutations=n_permutations
    )
    combination_space = msa.make_combination_space(permutation_space=permutation_space)
    complement_space = msa.make_complement_space(
        combination_space=combination_space, elements=nodes
    )
    input(f"Number of combinations to go through : {len(complement_space)}")

    n_classes_per_digit = config["datasets"]["n_classes_per_digit"]
    common_readout = config["model"]["common_readout"]

    torch.set_num_threads(1)
    community.to("cpu")

    all_accs = []

    for task in ["0", "1"]:

        if common_readout:
            decision = ["last", task]
        else:
            decision = ["last", "max"]

        data, t_target = get_data(task, datasets, n_classes_per_digit)

        n_processes = mp.cpu_count()
        # set_start_method('spawn', force=True)
        pool = mp.Pool(n_processes)

        masked_inf = partial(
            masked_inference,
            community=community,
            data=data,
            t_target=t_target,
            decision=decision,
            task=task,
            common_readout=common_readout,
        )

        print(f"Task {task} : Performance without ablations : {masked_inf([])}")

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

    contributions = [dict(zip(combination_space, accs)) for accs in all_accs]
    lesion_effects = [dict(zip(complement_space, accs)) for accs in all_accs]

    shapley_tables = [
        msa.get_shapley_table(contributions=c, permutation_space=permutation_space)
        for c in contributions
    ]
    shapley_tables = [shap[nodes] for shap in shapley_tables]
    shapley_tables_avg = [shap.mean() for shap in shapley_tables]

    return shapley_tables, shapley_tables_avg, all_accs, contributions, lesion_effects
