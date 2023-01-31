import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy
import math
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_n
import wandb
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

from community.common.training import train_community
from community.common.init import init_community
from community.utils.others import is_notebook
from community.utils.wandb_utils import get_wandb_artifact, mkdir_or_save_torch
from .subnets import GetSubnet_global

# ------ Weight Masks Models ------

"""
Weight Masks training in PyTorch
Based on  : 
"What's Hidden in a Randomly Weighted Neural Network?"
Vivek Ramanujan, Mitchell Wortsman, Aniruddha Kembhavi, Ali Farhadi, Mohammad Rastegari
(https://arxiv.org/abs/1911.13299)
https://github.com/RAIVNLab/hidden-networks
"""


def get_new_name(name):
    new_name = ""
    for n in name.split("."):
        new_name += n + "_"
    return new_name[:-1]


class Mask(nn.Module):
    """
    weight mask class, to be applied to any torch Module
    Args :
        model : model to apply mask onto
        sparsity : % of the weights to select in mask
    """

    def __init__(self, model, sparsity=0.5):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.sparsity = sparsity
        self.scores = nn.ParameterDict()

        for name, p in self.model.named_parameters():
            n = get_new_name(name)
            if "bias" not in n:
                self.scores[n] = nn.Parameter(
                    torch.Tensor(p.size()), requires_grad=True
                )
                nn.init.kaiming_uniform_(self.scores[n], a=math.sqrt(5))
            else:
                self.scores[n] = nn.Parameter(
                    torch.Tensor(p.size()), requires_grad=True
                )
                nn.init.normal_(self.scores[n], 0, 1e-2)
            p.requires_grad = False

    def get_subnets(self):
        return [
            GetSubnet_global.apply(s, list(self.scores.values()), self.sparsity)
            for s in self.scores.values()
        ]

    def forward(self, x):
        subnets = self.get_subnets()
        f_model = copy.deepcopy(self.model)
        for i, (name, p) in enumerate(f_model.named_parameters()):
            p *= subnets[i]

        return f_model(x)


class Mask_Community(nn.Module):
    """
    weight mask class, to be applied to Community class Module
    Args :
        model : Community to apply mask onto
        sparsity : % of the weights to select in mask
    """

    def __init__(self, model, sparsity=0.05, include_ih=False, include_readout=False):
        super().__init__()

        self.model = copy.deepcopy(model)

        self.__sparsity = sparsity

        self.scores = nn.ParameterDict()
        self.is_community = False

        self.include_ih = include_ih
        self.include_readout = include_readout

        for name, p in self.get_model_params(self.model).items():
            if (not "ih" in name or include_ih) and (
                not "readout" in name or include_readout
            ):  # Mask not applied to input weights
                n = get_new_name(name)
                if "bias" not in n:
                    self.scores[n] = nn.Parameter(
                        torch.Tensor(p.size()), requires_grad=True
                    )
                    nn.init.kaiming_uniform_(self.scores[n], a=math.sqrt(5))
                else:
                    self.scores[n] = nn.Parameter(
                        torch.Tensor(p.size()), requires_grad=True
                    )
                    nn.init.normal_(self.scores[n], 0, 1e-2)

        self.update_subnets()

    def get_model_params(self, model):

        model_params = {}
        for n, p in model.named_parameters():
            if "connections" not in n:
                model_params[n] = p

            p.requires_grad = False

        return model_params

    def update_subnets(self):
        self.subnets = {
            n: GetSubnet_global.apply(s, list(self.scores.values()), self.sparsity)
            for n, s in self.scores.items()
        }

    def set_sparsity(self, sparsity):
        self.__sparsity = sparsity
        self.update_subnets()

    @property
    def sparsity(self):
        return self.__sparsity

    def to(self, device):

        self.update_subnets()
        return super().to(device)

    @property
    def proportions(self):
        total = np.sum([s.data.cpu().sum() for n, s in self.subnets.items()])
        sums = np.array(
            [
                np.sum(
                    [
                        s.data.cpu().sum()
                        for n, s in self.subnets.items()
                        if n[7] == str(i)
                    ]
                )
                for i in range(len(self.model.agents))
            ]
        )
        return sums / total

    def forward(self, x, conns=0.0):
        f_model = copy.deepcopy(self.model)

        for (name, p) in self.get_model_params(f_model).items():
            try:
                p *= self.subnets[get_new_name(name)]
            except RuntimeError:  # subnets on cpu
                self.update_subnets()
                p *= self.subnets[get_new_name(name)]
            except KeyError:  # not using this parameter :
                pass

        return f_model(x)


def get_proportions(masked_model):

    scores = list(masked_model.scores.values())
    d_scores = masked_model.scores
    k = masked_model.sparsity
    identities = torch.cat(
        [torch.ones_like(s.flatten()) * n for n, s in enumerate(scores)]
    )
    _, idx = torch.cat([s.flatten() for s in scores]).sort()
    j = int((1 - k) * idx.numel())

    idxs_0 = [idx[:j][identities[idx[:j]] == n] for n, _ in enumerate(scores)]
    idxs_1 = [idx[j:][identities[idx[j:]] == n] for n, _ in enumerate(scores)]

    proportion_0 = [idxs.numel() / idx.numel() for idxs in idxs_0]
    proportion_1 = [idxs.numel() / idx.numel() for idxs in idxs_1]

    return {
        name: (p_0, p_1)
        for name, p_0, p_1 in zip(d_scores.keys(), proportion_0, proportion_1)
    }


def get_repartitions(masked_model):
    scores = list(masked_model.scores.values())
    k = masked_model.sparsity
    identities = torch.cat(
        [torch.ones_like(s.flatten()) * n for n, s in enumerate(scores)]
    )
    _, idx = torch.cat([s.flatten() for s in scores]).sort()
    j = int((1 - k) * idx.numel())

    top_idxs = identities[idx[j:]]
    top_idxs = (top_idxs / identities.max()).round()

    return idx[j:].cpu().data.numpy(), top_idxs.cpu().data.numpy()


### ------Weight mask Metrics -------


def train_mask(
    community,
    sparsity,
    loaders,
    target_digit,
    n_classes=10,
    lr=0.1,
    decision_params=["last", "max"],
    n_epochs=1,
    device=torch.device("cpu"),
    use_tqdm=False,
    symbols=False,
    force_connections=False,
    include_ih=False,
    include_r=False,
    multi_objectives=True,
):
    masked_community = Mask_Community(
        community,
        sparsity,
        include_ih=include_ih,
        include_readout=include_r,
    ).to(device)

    """
    momentum, wd = 0.9, 0.0005
    optimizer_agents = optim.SGD(
        [p for p in masked_community.parameters() if p.requires_grad],
        lr=lr,
        momentum=momentum,
        weight_decay=wd,
    )
    """
    optimizer_agents = torch.optim.AdamW(
        [p for p in masked_community.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-2,
    )

    optimizers = [optimizer_agents, None]

    d_params = decision_params[::]
    if multi_objectives:
        d_params[1] += f"_{target_digit}"

    training_dict = {
        "n_epochs": n_epochs,
        "task": target_digit,
        "global_rewire": False,
        "check_gradients": False,
        "reg_factor": 0.0,
        "train_connections": False,
        "decision": d_params,
        "stopping_acc": None,
        "early_stop": False,
        "deepR_params_dict": {},
        "data_type": "symbols" if symbols else None,
        "force_connections": force_connections,
        "n_classes": n_classes,
        "n_classes_per_digit": n_classes,
    }

    train_out = train_community(
        masked_community,
        *loaders,
        optimizers,
        config=training_dict,
        device=device,
        trials=(True, True),
        use_tqdm=use_tqdm,
    )

    return (
        masked_community,
        train_out["test_losses"],
        train_out["test_accs"],
        train_out["best_state"],
    )


def find_optimal_sparsity(
    masked_community,
    target_digit,
    loaders,
    min_acc=None,
    device=torch.device("cpu"),
    use_tqdm=False,
    symbols=False,
    n_classes=10,
    d_params=("last", "max"),
):

    optimizers = None, None

    if type(use_tqdm) is int:
        position = use_tqdm
        use_tqdm = True
    elif use_tqdm:
        position = 0

    def test_masked_com():

        training_dict = {
            "n_epochs": 1,
            "task": str(target_digit),
            "global_rewire": False,
            "check_gradients": False,
            "reg_factor": 0.0,
            "train_connections": False,
            "decision": d_params,
            "stopping_acc": None,
            "early_stop": False,
            "deepR_params_dict": {},
            "data_type": "symbols" if symbols else None,
            "force_connections": False,
            "n_classes": n_classes,
            "n_classes_per_digit": n_classes,
        }

        train_out = train_community(
            masked_community,
            *loaders,
            optimizers,
            config=training_dict,
            device=device,
            trials=(False, True),
            use_tqdm=False,
        )
        return train_out

    if min_acc is None:
        s = masked_community.sparsity
        masked_community.sparsity = 1.0
        train_out = test_masked_com()
        min_acc = train_out["test_accs"].max()
        masked_community.sparsity = s
        print(masked_community.sparsity)

    train_out = test_masked_com()
    test_acc = train_out["test_accs"].max()

    pbar = tqdm(total=100, position=position, leave=None)
    pbar.set_description(
        f"For Mask of Sparsity {masked_community.sparsity}, Acc is {test_acc}, min_acc = {min_acc}"
    )

    while test_acc >= min_acc:
        # print(test_acc, masked_community.sparsity)
        masked_community.sparsity *= 0.7
        train_out = test_masked_com()
        test_acc = train_out["test_accs"].max()
        pbar.set_description(
            f"For Mask of Sparsity {masked_community.sparsity}, Acc is {test_acc}"
        )
        pbar.update(1)

    return masked_community.sparsity, test_acc


def train_and_get_mask_metric(
    community,
    initial_sparsity,
    loaders,
    config,
    n_epochs=1,
    lr=0.1,
    use_optimal_sparsity=False,
    device=torch.device("cuda"),
    use_tqdm=False,
    chosen_timesteps=["last"],
):
    """
    Initializes and trains masks on community model.
    Args :
        community : model to apply the mask onto
        sparsity : % of weights to be selected
        loader : training and testing data-loaders
        n_tests : number of init and train to conduct
        n_epochs : number of epochs of training
        lr : learning rate of training

    Returns :
        proportion_per_agent : weight repartition among sub-networks
        accuracies : trained masks accuracies
        losses : trained masks losses
        best_states : state dicts of trained masks
    """
    prop_per_agent_total = []
    test_accuracies_total = []
    test_losses_total = []
    best_states_total = []
    sparsities_total = []
    masked_models_total = []

    if type(use_tqdm) is int:
        position = use_tqdm
        use_tqdm = True
    elif use_tqdm:
        position = 0

    decision_params = config["training"]["decision"]
    n_classes = config["datasets"]["n_classes_per_digit"]
    symbols = config["datasets"]["data_type"] == "symbols"
    if symbols:
        include_ih = config["datasets"]["symbol_config"]["common_input"]
    else:
        include_ih = False

    include_r = config["model"]["common_readout"]

    notebook = is_notebook()
    tqdm_f = tqdm_n if notebook else tqdm

    pbar = chosen_timesteps
    if use_tqdm:
        pbar = tqdm_f(pbar, position=position, desc="Mask Metric Trials : ", leave=None)

    if config["model"]["common_readout"]:
        multi_objectives = config["model"]["n_readouts"] > 1
    else:
        multi_objectives = config["model"]["agents"]["n_readouts"] > 1

    for ts in pbar:

        prop_per_agent = []
        test_accuracies = []
        test_losses = []
        best_states = []
        sparsities = []
        masked_models = []

        d_params = decision_params[::]
        d_params[0] = ts

        for target_digit in ["0", "1"]:

            masked_community, test_loss, test_accs, best_state = train_mask(
                community,
                initial_sparsity,
                loaders,
                target_digit,
                n_classes,
                lr,
                d_params,
                n_epochs,
                device,
                position + 1 if use_tqdm else False,
                symbols=symbols,
                include_ih=include_ih,
                include_r=include_r,
                multi_objectives=multi_objectives,
            )

            if use_optimal_sparsity:

                optimal_sparsity, test_accs = find_optimal_sparsity(
                    masked_community,
                    target_digit,
                    loaders,
                    min_acc=None,
                    device=device,
                    symbols=symbols,
                    use_tqdm=position + 1 if use_tqdm else False,
                )

            # prop = get_proportions_per_agent(masked_community)[0]
            prop = masked_community.proportions

            prop_per_agent.append(prop)
            test_accuracies.append(np.array(test_accs))
            test_losses.append(np.array(test_loss))
            best_states.append(best_state)
            sparsities.append(masked_community.sparsity)
            masked_models.append(masked_community)

        best_states_total.append(np.stack(best_states, -1))
        prop_per_agent_total.append(np.stack(prop_per_agent, -1))
        test_accuracies_total.append(np.stack(test_accuracies, -1).max(0)[0])
        # test_losses_total.append(np.stack(test_losses, -1))
        sparsities_total.append(np.stack(sparsities, -1))
        masked_models_total.append(masked_models)

    results_dict = {
        "proportions": np.stack(
            prop_per_agent_total, -1
        ),  # n_agents x n_targets x timesteps
        "test_accs": np.stack(test_accuracies_total, -1),
        # "test_losses": np.stack(test_losses_total, -1),
        "best_states": best_states_total,
        "sparsities": np.stack(sparsities_total, -1),
    }

    return results_dict, masked_models_total


def get_proportions_per_agent(masked_community, method=0):
    """
    Returns repartition of selected weights among agents :
    Args :
        masked_community : weight mask applied to community
    """

    if method == 0:
        # Average of averages

        proportions = get_proportions(masked_community)
        return (
            np.array(
                [
                    np.mean(
                        [
                            p[1] / np.sum(p)
                            for n, p in get_proportions(masked_community).items()
                            if n[7] == str(i)
                        ]
                    )
                    for i in range(len(masked_community.model.agents))
                ]
            ),
            0,
        )
    else:
        prop_ag_0 = 0
        prop_ag_1 = 0
        sparsity = masked_community.sparsity
        numel = 0
        for p in masked_community.scores.values():
            numel += p.numel()
        sparsity = masked_community.sparsity
        for n, p in get_proportions(masked_community).items():
            if "agents" in n:
                if n[7] == "0":
                    prop_ag_0 += p[1] / sparsity
                elif n[7] == "1":
                    prop_ag_1 += p[1] / sparsity

            if n[0] == "0" and "thetas" not in n:
                prop_ag_0 += p[1] / sparsity
            elif n[0] == "1" and "thetas" not in n:
                prop_ag_1 += p[1] / sparsity

        prop = np.array((prop_ag_0, prop_ag_1))
        return prop, prop * numel * sparsity


def compute_mask_metric(
    p_cons,
    loaders,
    save_name,
    p_masks=[0.1],
    lr=1e-1,
    device=torch.device("cuda"),
    config=None,
):
    """
    Compute complete mask metric, for every trained community and sparsity of interconnections
    """
    proportions_per_agent = {}
    test_accuracies_supermasks = {}
    test_losses_supermasks = {}
    trained_masks = {}
    l = 0

    notebook = is_notebook()

    tqdm_f = tqdm_n if notebook else tqdm

    use_wandb = wandb.run is not None
    if use_wandb:
        config = wandb.config
    else:
        assert config is not None, "Provide configuration dict or run using WandB"

    save_path = config["saves"]["metrics_save_path"]
    community_state_path = (
        config["saves"]["models_save_path"] + config["saves"]["models_save_name"]
    )
    agent_params_dict = config["model_params"]["agents_params"]

    try:  # Load states from file
        community_states = torch.load(community_state_path)
        print("Loading models from file")
        print(community_state_path)
    except FileNotFoundError:  # Load from WandB artifacts
        try:
            community_states, *_ = get_wandb_artifact(
                None,
                project="funcspec",
                name="state_dicts",
                run_id=config["resume_run_id"],
            )
        except KeyError:
            community_states, *_ = get_wandb_artifact(
                config, project="funcspec", name="state_dicts", process_config=True
            )
        print("Loading models from artifact")
    community = init_community(
        agent_params_dict,
        0.1,
        device=device,
        use_deepR=config["model_params"]["use_deepR"],
    )

    task = wandb.config["task"]

    for i, p_con in enumerate(
        tqdm_f(p_cons[l:], position=0, desc="Community Sparsity : ", leave=None)
    ):

        prop_per_agent = {}
        test_accs = {}
        best_states = {}

        states = community_states[p_con]

        for p_mask in tqdm_f(p_masks, position=1, desc="Mask Sparsity", leave=None):

            prop_per_agent[p_mask], test_accs[p_mask], best_states[p_mask] = [], [], []

            for i, state in enumerate(
                tqdm_f(states, position=2, desc="Model Trials : ", leave=None)
            ):
                community.load_state_dict(state)

                results = train_and_get_mask_metric(
                    community, p_mask, loaders, lr=lr, device=device
                )

                prop, acc, states = (
                    results["proporttions"],
                    results["test_accs"],
                    results["best_states"],
                )
                prop_per_agent[p_mask].append(prop)
                test_accs[p_mask].append(acc)
                best_states[p_mask].append(states)

            prop_per_agent[p_mask], test_accs[p_mask] = np.array(
                prop_per_agent[p_mask]
            ), np.array(test_accs[p_mask])
            best_states[p_mask] = np.array(best_states[p_mask])

        proportions_per_agent[p_con] = prop_per_agent
        test_accuracies_supermasks[p_con] = test_accs
        trained_masks[p_con] = best_states

        metric = {
            "Proportions": proportions_per_agent,
            "Accs": test_accuracies_supermasks,
            "Trained_masks": trained_masks,
        }

        mkdir_or_save_torch(metric, save_name, save_path)

    figures = fig1, fig2 = plot_mask_metric(metric)
    if use_wandb:
        wandb.log_artifact(save_path + save_name, name="masks", type="metric")
        wandb.log(
            {
                "Mask Metric": wandb.Image(fig1),
                "Mask Difference Metric": wandb.Image(fig2),
            }
        )

    return metric, figures


def get_metrics_from_saved_masks(
    mask_file, masked_community=None, sparsities=[0.1, 0.05, 0.01], config=None
):

    """
    Computes the weight mask metrics for different k% of selected weights from saved masks states
    """
    if masked_community is None:
        assert config is not None or wandb.run is not None
        try:
            config = wandb.run.config
            agent_params_dict = config["model_params"]["agents_params"]
            device = torch.device("cpu")
            community = init_community(
                agent_params_dict,
                0.1,
                device=device,
                use_deepR=config["model_params"]["use_deepR"],
            )
            masked_community = Mask_Community(community, sparsities[0]).to(device)

        except AttributeError:
            print("Provide masked community model or run using wandb")
            return

    p_cons = list(mask_file["Trained_masks"].keys())
    l = len(p_cons)
    sparsity = list(mask_file["Trained_masks"][p_cons[0]].keys())[0]
    states = {p: np.array(mask_file["Trained_masks"][p][sparsity]) for p in p_cons[:l]}
    proportions = {}
    for p_con in tqdm(p_cons[:]):
        proportions[p_con] = {s: [[] for _ in range(2)] for s in sparsities}
        for t in range(2):
            for state in states[p_con][..., t].flatten():
                masked_community.load_state_dict(state)
                for sparsity in sparsities:
                    masked_community.sparsity = sparsity
                    prop = get_proportions_per_agent(masked_community)[0]
                    proportions[p_con][sparsity][t].append(prop)

        for sparsity in sparsities:
            proportions[p_con][sparsity] = np.transpose(
                np.array(proportions[p_con][sparsity]), (1, 0, 2)
            )[None, ...]

    mask_file["Proportions"] = proportions

    return mask_file


def plot_mask_metric(mask_metric):

    p_cons = np.array(list(mask_metric["Accs"].keys()))
    l = len(p_cons)

    linestyles = ["--", "-", ":"]

    proportions = mask_metric["Proportions"]
    sparsities = list(proportions[p_cons[0]].keys())
    fig1, axs = plt.subplots(1, 2, figsize=(20, 5), sharey=False)
    sorted_idxs = [
        [
            [np.argsort(mask_metric["Accs"][p][0.1][c, :, k, 0])[:1] for c in range(1)]
            for k in range(2)
        ]
        for p in p_cons
    ]

    ax = axs[0]
    for n in range(2):
        for t in range(1):
            for i, k in enumerate(sparsities):

                linestyle = linestyles[i]

                mean = np.array(
                    [proportions[p_con][k][..., t, n].mean() for p_con in p_cons[:l]]
                )
                std = np.array(
                    [proportions[p_con][k][..., t, n].std() for p_con in p_cons[:l]]
                )
                plot = ax.plot(
                    p_cons[:l],
                    mean,
                    label=f"Subnetwork {n}, Subtask {t}, {k*100}% weights",
                    linestyle=linestyle,
                )
                col = plot[-1].get_color()
                ax.fill_between(
                    p_cons[:l], mean - std, mean + std, color=col, alpha=0.2
                )
                for p_con in p_cons[:l]:
                    data_points = proportions[p_con][k][..., t, n].mean(1).flatten()
                    # ax.plot([p_con]*len(data_points), data_points, '.', color=col, alpha=0.4)

    ax.legend()
    ax.set_ylabel(
        "Functional Specialization :\n Proportion of selected weights present",
        fontsize=13,
    )
    ax.set_title(f"Weight Mask Metric, Proportions", fontsize=15)
    ax.set_xscale("log")

    for m, metric in enumerate(["Accs"]):
        ax = axs[1 + m]
        for n in range(2):
            for i, k in enumerate([0.1]):
                linestyle = linestyles[n]
                mean = [
                    mask_metric[metric][p_con][k][..., n, -1].mean()
                    for p_con in p_cons[:l]
                ]
                plot = ax.plot(
                    p_cons[:l],
                    mean,
                    label=f"Subtask {n}, {k*100}% weights",
                    linestyle=linestyle,
                )
                std = np.array(
                    [
                        mask_metric[metric][p_con][k][..., n, -1].std()
                        for p_con in p_cons[:l]
                    ]
                )
                col = plot[-1].get_color()
                ax.fill_between(
                    p_cons[:l], mean - std, mean + std, color=col, alpha=0.2
                )
                for p_con in p_cons[:l]:
                    data_points = mask_metric[metric][p_con][k][..., n, -1].flatten()

                    ax.plot(
                        [p_con] * len(data_points),
                        data_points,
                        ".",
                        color=col,
                        alpha=0.4,
                    )

        ax.legend()

        ax.set_ylabel(
            "Functional Specialization :\n Performance of masked model", fontsize=13
        )
        ax.set_title(f"Masked Models Accuracies", fontsize=15)
        ax.set_xscale("log")
        ax.set_xlabel("Proportion of active connections", fontsize=15)

    fig1.suptitle("Weight Mask Metric")

    metrics = lambda p_con: (
        proportions[p_con][k][..., n, n],
        proportions[p_con][k][..., 1 - n, n],
    )
    norm_diff = lambda p_con: (
        (metrics(p_con)[0] - metrics(p_con)[1])
        / (metrics(p_con)[0] + metrics(p_con)[1])
    )

    fig2, axs = plt.subplots(1, 2, figsize=(15, 5))
    x_axis = [p_cons, (1 - p_cons) / (2 * (1 + p_cons))]
    x_labels = ["Proportion of active connections", "Q Modularity Measure"]
    sparsities = list(proportions[p_cons[0]].keys())

    for j, p_cons_Q in enumerate(x_axis):
        ax = axs[j]
        for n in range(2):
            for i, k in enumerate(sparsities[:]):
                linestyle = linestyles[n]
                mean = np.array([norm_diff(p_con).mean() for p_con in p_cons[:l]])
                std = np.array([norm_diff(p_con).std() for p_con in p_cons[:l]])
                plot = ax.plot(
                    p_cons_Q[:l], mean, label=f"Subnetwork {n}", linestyle=linestyle
                )
                col = plot[-1].get_color()
                ax.fill_between(
                    p_cons_Q[:l], mean - std, mean + std, color=col, alpha=0.2
                )
                for p_con in p_cons[:l]:
                    data_points = np.array(norm_diff(p_con)).flatten()

                    # ax.plot([p_con]*len(data_points), data_points, '.', color=col)

        # ax.hlines(xmin=p_cons_Q[0], xmax=p_cons_Q[-1], y=0, color='black', linestyle='-', alpha=0.3)

        ax.set_xlabel(x_labels[j], fontsize=15)

        axs[1].yaxis.tick_right()
        # axs[2].yaxis.set_label_position("right")
        if j == 0:
            ax.set_ylabel(
                "Functional Specialization :\n Atribution Difference", fontsize=13
            )
        if j == 0:
            ax.set_title(f"Weight Mask Metric", fontsize=15)
        ax.legend()

        ax.set_xscale(
            "log" * (p_cons_Q is p_cons) + "linear" * (p_cons_Q is not p_cons)
        )

    fig2.suptitle(f"Weight Mask Metric Diff", fontsize=15)
    fig2.tight_layout()

    return fig1, fig2
