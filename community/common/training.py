import warnings
from community.data.tasks import get_task_family
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm.notebook import tqdm as tqdm_n
from tqdm import tqdm
import wandb
import seaborn as sns
import matplotlib.pyplot as plt

from .init import init_community, init_optimizers
from ..utils.others import (
    check_grad,
    is_notebook,
)
from ..utils.nested import (
    nested_shape,
    nested_round,
    nested_sup,
    nested_len,
    nested_mean,
)
from ..utils.wandb_utils import mkdir_or_save

from ..utils.configs import get_training_dict

from .models.ensembles import ConvCommunity
from .decision import get_decision
from ..data.process import process_data, get_task_target

from deepR.models import step_connections

# ------ Training and Testing functions ------


def get_loss(output, t_target, both=False):
    if both:
        loss = [get_loss(o, t_target) for o in output]
    # print(type(output), type(t_target))
    # print(output, t_target)
    else:
        try:
            loss = F.cross_entropy(output, t_target, reduction="none")
            output = output.unsqueeze(0)

        except (TypeError, RuntimeError) as e:
            loss = [get_loss(o, t) for o, t in zip(output, t_target)]

        # except ValueError:
        # loss = torch.stack([get_loss(o, t_target) for o in output])

    try:
        loss = torch.stack(loss)
    except (RuntimeError, TypeError) as e:
        pass

    return loss


def get_acc(output, t_target, both=False):
    if both:
        acc = [get_acc(o, t_target) for o in output]

    else:
        n_tasks = len(t_target)
        n_out = len(output)

        if n_out == n_tasks:
            try:
                pred = output.argmax(
                    dim=-1, keepdim=True
                )  # get the index of the max log-probability

                correct = pred.eq(t_target.view_as(pred))
                if len(pred.shape) > 2:
                    acc = [get_acc(o, t) for o, t in zip(output, t_target)]
                else:
                    acc = (correct.sum() / t_target.numel()).cpu().data.numpy()

            # task is list, get acc for all pairs
            except AttributeError as e:
                acc = [get_acc(o, t) for o, t in zip(output, t_target)]

            except (TypeError, RuntimeError) as e:
                acc = [get_acc(o, t_target) for o in output]

        else:
            acc = [get_acc(o, t_target) for o in output]

    return np.array(acc)


def get_loss2(output, t_target):
    n_target = nested_shape(t_target)
    n_decisions = nested_shape(output)

    if len(n_decisions) == 1:
        n_decisions = n_decisions[0]
    elif len(n_decisions) == 0:
        n_decisions = 1

    if len(n_target) == 1:
        n_target = n_target[0]
    elif len(n_target) == 0:
        n_target = 1

    if n_target == n_decisions == 1:
        loss = F.cross_entropy(output, t_target, reduction="none")
        output = output.unsqueeze(0)

    elif n_target == 1 and n_decisions != 1:
        t_target = t_target.unsqueeze(0).expand(output.shape[:-1])
        loss = torch.stack(
            [F.cross_entropy(o, t, reduction="none") for o, t in zip(output, t_target)]
        ).T

    elif n_target != 1 and n_decisions == 1:
        loss = torch.stack(
            [F.cross_entropy(output, t, reduction="none") for t in t_target]
        ).T

    elif n_target == n_decisions:
        try:
            loss = torch.stack(
                [
                    F.cross_entropy(o, t, reduction="none")
                    for o, t in zip(output, t_target)
                ]
            ).T
        except RuntimeError:
            res = [get_loss2(o, t) for o, t in zip(output, t_target)]
            loss, t_target = torch.stack([r[0] for r in res]).T, torch.stack(
                [r[1] for r in res]
            )

    else:
        res = [get_loss2(o, t_target) for o in output]
        loss, t_target = torch.stack([r[0] for r in res]).T, torch.stack(
            [r[1] for r in res]
        )

    return loss.mean(), t_target, output


def binary_conn(target, ag):
    n_classes = len(target.unique())
    n_bits = np.ceil(np.log2(n_classes)).astype(int)
    encoding = []
    encoded_target = target[:, ag].clone().detach()
    for d in range(n_bits - 1, -1, -1):
        encoding.append(torch.div(encoded_target, 2**d, rounding_mode="floor"))
        encoded_target -= (
            torch.div(encoded_target, 2**d, rounding_mode="floor") * 2**d
        )
    return torch.stack(encoding, -1)


def train_community(
    model,
    train_loader,
    test_loader,
    optimizers,
    schedulers=None,
    config=None,
    n_epochs=None,
    trials=(True, True),
    use_tqdm=2,
    device=torch.device("cuda"),
    show_all_acc=True,
    ag_masks=None,
):
    """
    Training and testing function for Community of agents
    Args :
        model : community to train
        train_loader, test_loader : dataloaders to train and test on
        optimizers = optimizer_agents, optimizer_connections : optimizers for agents and connections of the community
        schedulers : learning rate schedulers for both optimizers
        trials : (training, testing) bools
        config : config dict or file_path
    """

    optimizer_agents, optimizer_connections = optimizers
    training, testing = trials

    notebook = is_notebook()

    assert (
        config is not None or wandb.run is not None
    ), "Provide training config or run with WandB"

    if config is None:
        config = get_training_dict(wandb.config)

    # ----Config----
    n_epochs = config["n_epochs"] if n_epochs is None else n_epochs
    task = config["task"]
    reg_factor = config["reg_factor"]
    train_connections = config["train_connections"] and config["sparsity"] > 0
    check_gradients = config["check_gradients"]
    global_rewire = config["global_rewire"]
    decision = config["decision"]
    stopping_acc = config["stopping_acc"]
    early_stop = config["early_stop"]
    deepR_params_dict = config["deepR_params_dict"]
    symbols = config["data_type"] == "symbols"
    force_connections = config["force_connections"]
    common_input = config["common_input"]
    n_steps = config["nb_steps"]
    noise_ratio = config["noise_ratio"]
    random_start = config["random_start"]

    n_classes = config["n_classes"]
    n_classes_per_digit = config["n_classes_per_digit"]

    # --------------

    reg_loss = reg_factor > 0.0

    if type(use_tqdm) is int:
        position = use_tqdm
        use_tqdm = True
    elif use_tqdm:
        position = 0

    flatten = not (type(model) is ConvCommunity)
    if train_connections and model.is_community:
        thetas_list = [
            c.thetas[0] for c in model.connections.values() if c.is_deepR_connect
        ]
        sparsity_list = [
            c.sparsity_list[0] for c in model.connections.values() if c.is_deepR_connect
        ]
        if thetas_list == []:
            # print('Empty Thetas List !!')
            warnings.warn("Empty Theta List", Warning)

    descs = ["" for _ in range(2)]
    desc = lambda descs: descs[0] + descs[1]
    train_losses, train_accs = [], []
    test_accs, test_losses = [], []
    deciding_agents = []
    best_loss, best_acc = 1e10, 0.0

    pbar = range(n_epochs + 1)
    if use_tqdm:
        tqdm_f = tqdm_n if notebook else tqdm
        pbar = tqdm_f(pbar, position=position, leave=None, desc="Train Epoch:")

    # dummy fwd for shapes
    data, target = next(iter(train_loader))
    data, target = process_data(
        data,
        target,
        symbols=symbols,
        n_steps=n_steps,
        common_input=common_input,
        task=task,
        noise_ratio=noise_ratio,
        random_start=random_start,
    )
    *_, fconns = model(data.to(device))

    # try:
    #    model = torch.compile(model)
    # except AttributeError:
    #    pass

    for epoch in pbar:
        if training and epoch > 0:
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if type(data) is list:
                    data, target = [d.to(device) for d in data], target.to(device)
                else:
                    data, target = data.to(device), target.to(device)

                # Forward pass

                # Task Selection
                data, target = process_data(
                    data,
                    target,
                    task,
                    flatten=flatten,
                    symbols=symbols,
                    common_input=common_input,
                    n_steps=n_steps,
                    noise_ratio=noise_ratio,
                    random_start=random_start,
                )

                t_target = get_task_target(target, task, n_classes_per_digit)

                optimizer_agents.zero_grad()
                if optimizer_connections:
                    optimizer_connections.zero_grad()

                if force_connections:
                    conns = fconns[-1].detach().unsqueeze(0)
                    for ag in range(2):
                        conns[:, ag, :, model.nonzero_received[ag]] = binary_conn(
                            target, 1 - ag
                        ).float()
                    conns[conns == 0] = -1
                else:
                    conns = None

                output, out_dict = model(data, conns, ag_masks)
                if decision is not None:
                    output, deciding_ags = get_decision(
                        output, *decision, target=t_target
                    )
                    both = decision[1] == "both"
                else:
                    deciding_ags = None
                    both = False

                try:
                    if (
                        deciding_ags is not None
                        and train_loader.batch_size in deciding_ags.shape
                    ):
                        deciding_agents.append(deciding_ags.cpu().data.numpy())
                except AttributeError:
                    deciding_ags = None

                complete_loss = get_loss(output, t_target, both=both)
                loss = nested_mean(complete_loss)

                acc = get_acc(output, t_target, both=both)

                if reg_loss:
                    reg = F.mse_loss(
                        deciding_ags.float().mean(),
                        torch.full_like(deciding_ags.float().mean(), 0.5),
                    )
                    loss += reg * reg_factor

                """
                pred = output.argmax(
                    dim=-1, keepdim=True
                )  # get the index of the max log-probability

                correct = pred.eq(t_target.view_as(pred))

                if output.shape[0] == 1 : # not joint_training : 
                    correct = correct.sum().cpu().data.item()
                    train_accs.append(correct/t_target.numel())
                else : 
                    correct = correct.flatten(start_dim=-2).sum(-1).cpu().data.numpy()
                    train_accs.append(correct/t_target[0].numel())
                

                pred = output.argmax(dim=-1)
                correct = pred.eq(t_target.view_as(pred))
                acc = (
                    (correct.sum(-1) * np.prod(t_target.shape[:-1]) / t_target.numel())
                    .cpu()
                    .data.numpy()
                )
                """
                train_accs.append(acc)

                loss.backward()

                if False:
                    for p in model.parameters():
                        p.grad = torch.sign(p.grad)

                if check_gradients:
                    check_grad(model)

                train_losses.append(loss.cpu().data.item())

                # Apply gradients on agents weights
                optimizer_agents.step()
                if hasattr(model, "scores"):
                    model.update_subnets()

                # Apply gradient for sparse connections and rewire
                """
                if model.is_community and train_connections: 
                    nb_new_con = step_connections(model, optimizer_connections, global_rewire, thetas_list,
                                                  sparsity_list, deepR_params_dict=deepR_params_dict)
                else : 
                    
                """
                nb_new_con = 0
                acc = train_accs[-1]
                descs[0] = str(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.3f}, Accuracy: {}, Dec : {:.3f}%".format(
                        epoch,
                        batch_idx * train_loader.batch_size,
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        torch.round(complete_loss.mean(-1), decimals=1).data
                        if False
                        else torch.round(loss, decimals=1).item(),
                        np.round(100 * np.mean(acc))
                        if not show_all_acc
                        else nested_round(acc),
                        np.mean(deciding_agents),
                    )
                )

                if use_tqdm:
                    pbar.set_description(desc(descs))

        if testing:
            descs[1], loss, acc, _ = test_community(
                model,
                device,
                test_loader,
                config,
                show_all_acc,
                ag_masks,
            )
            if loss < best_loss:
                best_loss = loss
                best_state = copy.deepcopy(model.state_dict())
                best_acc = acc

            test_losses.append(loss)
            test_accs.append(acc)

            if use_tqdm:
                pbar.set_description(desc(descs))

        if schedulers is not None:
            for sch in schedulers:
                if sch:
                    sch.step()

        results = {
            "train_losses": np.array(train_losses),
            "train_accs": np.array(train_accs),
            "test_losses": np.array(test_losses),
            "test_accs": np.array(test_accs),
            "deciding_agents": np.array(deciding_agents),
            "best_state": best_state,
        }

        # Stop training if loss doesn't go down or if stopping_acc is reached
        if epoch >= 4:
            if results["test_losses"][-4:].argmin() == 0 and early_stop:
                # print('Stopping Training (Early Stop), loss hasn\'t improved in 4 epochs')
                return results

        if stopping_acc is not None:
            if nested_sup(acc, stopping_acc):
                return results

    return results


def test_community(
    model,
    device,
    test_loader,
    config,
    show_all_acc=True,
    ag_masks=None,
    verbose=False,
    seed=None,
):
    """
    Testing function for community of agents
    """

    symbols = config["data_type"] == "symbols"
    force_connections = config["force_connections"]

    n_classes = config["n_classes"]
    n_classes_per_digit = config["n_classes_per_digit"]
    common_input = config["common_input"]
    n_steps = config["nb_steps"]

    task = config["task"]
    decision = config["decision"]
    noise_ratio = config["noise_ratio"]
    random_start = config["random_start"]

    model.eval()
    flatten = not (type(model) is ConvCommunity)
    test_loss = 0
    correct = 0
    acc = 0
    deciding_agents = []
    if seed is not None:
        torch.manual_seed(seed)

    data, target = next(iter(test_loader))
    data, target = process_data(
        data,
        target,
        task=task,
        flatten=flatten,
        symbols=symbols,
        common_input=common_input,
        n_steps=n_steps,
        noise_ratio=noise_ratio,
        random_start=random_start,
    )
    *_, fconns = model(data.to(device))
    with torch.no_grad():
        for data, target in test_loader:
            if type(data) is list:
                data, target = [d.to(device) for d in data], target.to(device)
            else:
                data, target = data.to(device), target.to(device)

            data, t_target = process_data(
                data,
                target,
                task=task,
                flatten=flatten,
                symbols=symbols,
                common_input=common_input,
                n_steps=n_steps,
                noise_ratio=noise_ratio,
                random_start=random_start,
            )

            t_target = get_task_target(target, task, n_classes_per_digit)

            if force_connections:
                conns = fconns[-1].detach().unsqueeze(0)
                for ag in range(2):
                    conns[:, ag, :, model.nonzero_received[ag]] = binary_conn(
                        target, 1 - ag
                    ).float()
                conns[conns == 0] = -1
            else:
                conns = None

            output, out_dict = model(data, conns, ag_masks)
            if decision is not None:
                output, deciding_ags = get_decision(output, *decision, target=t_target)
                both = decision[1] == "both"
            else:
                deciding_ags = None
                both = False

            try:
                if (
                    deciding_ags is not None
                    and test_loader.batch_size in deciding_ags.shape
                ):
                    deciding_agents.append(deciding_ags.cpu().data.numpy())
            except AttributeError:
                deciding_ags = None

            complete_loss = get_loss(output, t_target, both=both)
            loss = nested_mean(complete_loss)

            test_loss += loss
            test_acc = get_acc(output, t_target, both=both)

            """
            pred = output.argmax(
                dim=-1, keepdim=True
            )  # get the index of the max log-probability
            c = pred.eq(t_target.view_as(pred))
            if output.shape[0]==1 : #not joint_training : 
                correct += c.sum().cpu().data.item()
                acc += c.sum().cpu().data.item()/t_target.numel()
            else : 
                correct += c.flatten(start_dim=-2).sum(-1).cpu().data.numpy()
                acc += c.flatten(start_dim=-2).sum(-1).cpu().data.numpy()/t_target[0].numel()
            pred = output.argmax(dim=-1)
            c = pred.eq(t_target.view_as(pred))
            test_acc = (
                (c.sum(-1) * np.prod(t_target.shape[:-1]) / t_target.numel())
                .cpu()
                .data.numpy()
            )

            correct += c    
            """
            acc += test_acc

    test_loss /= len(test_loader)
    acc /= len(test_loader)

    deciding_agents = np.array(deciding_agents)

    desc = str(
        " | Test set: Loss: {:.3f}, Accuracy: {}%".format(
            test_loss,
            np.round(100 * np.mean(acc)) if not show_all_acc else nested_round(acc),
        )
    )

    if verbose:
        print(desc)

    return desc, test_loss.cpu().data.item(), acc, deciding_agents


def plot_confusion_mat(
    model, test_loader, config, n_classes, device=torch.device("cuda")
):
    accs = []
    targets, t_targets = [], []

    task = config["task"]
    symbols = config["data_type"] == "symbols"
    decision = config["decision"]
    common_input = config["common_input"]
    n_steps = config["nb_steps"]
    flatten = not (type(model) is ConvCommunity)

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = process_data(
            data,
            target,
            task=task,
            flatten=flatten,
            symbols=symbols,
            common_input=common_input,
            n_steps=n_steps,
        )

        data, t_target = data.to(device), get_task_target(target, task, n_classes).to(
            device
        )

        outputs, states, conns = model(data)
        # print((outputs[-1][0] == outputs[-1][1]).all())
        output, deciding_ags = get_decision(outputs, *decision, target)

        loss = get_loss(output, t_target)

        pred = output.argmax(dim=-1, keepdim=True)
        correct = pred.eq(t_target.view_as(pred)).cpu().data
        targets.append(target.cpu())
        t_targets.append(t_target.cpu())
        accs.append(correct)

    accs, targets = torch.cat(accs), torch.cat(targets)
    n_classes = len(targets.unique())
    t_masks = [(targets == t).all(1) for t in targets.unique(dim=0)]

    acc_per_target = [accs[m].float().mean() for m in t_masks]
    acc_per_target = np.array([accs[m].float().mean() for m in t_masks]).reshape(
        n_classes, n_classes
    )

    # acc_per_target = np.array([[acc_per_target[t1*n_classes + t2].cpu().data.item() for t1 in range(n_classes)] for t2 in range(n_classes)])

    ax = sns.heatmap(
        acc_per_target,
        cmap="inferno",
        annot=acc_per_target.round(1).astype(str),
        annot_kws={"fontsize": 10},
        fmt="s",
    )
    ax.set_title("Confusion Matrix")
    plt.show()


def compute_trained_communities(
    p_cons, loaders, device=torch.device("cuda"), notebook=False, config=None
):
    """
    Trains and saves model for all levels of sparsity of inter-connections
    Args :
        p_cons : list of sparisities of inter-connections
        loaders : training and testing data-loaders
        config : config dict, created in main or from WandB (sweeps)
        save_name : file name to be saved

    """
    if wandb.run is not None:
        config = wandb.config
    else:
        assert config is not None, "Provide configuration dict or run using WandB"

    task = config["task"]
    print(f"Starting training on {task}")
    params_dict, deepR_params_dict = tuple(config["optimization"].values())
    agent_params_dict = config["model"]["agents"]
    connections_params_dict = config["model"]["connections"]

    inverse_task = "digits" in task and config["training"]["inverse_task"]

    l = 0
    save_path = config["saves"]["models_save_path"]
    save_name = config["saves"]["models_save_name"]
    total_path = save_path + save_name

    print(total_path)

    try:
        community_states = torch.load(total_path)
        start = len(community_states.keys())
        print("Warning : file already exists, picking training up from last save")

    except FileNotFoundError:
        community_states = {}
        start = 0

    start = 0

    gdnoises = deepR_params_dict["gdnoise"] * (1 - p_cons)
    lrs_ag = [params_dict["lr"]] * len(p_cons)
    lrs_con = np.geomspace(
        deepR_params_dict["lr"], deepR_params_dict["lr"] / 100, len(p_cons)
    )

    notebook = is_notebook()

    tqdm_f = tqdm_n if notebook else tqdm
    pbar1 = tqdm_f(p_cons[start:], position=0, desc="Model Sparsity : ", leave=None)

    for i, p_con in enumerate(pbar1):
        community_states[p_con] = []
        desc = "Model Trials"
        pbar2 = tqdm_f(
            range(config["training"]["n_tests"]), position=1, desc=desc, leave=None
        )

        for test in pbar2:
            deepR_params_dict["gdnoise"], params_dict["lr"], deepR_params_dict["lr"] = (
                gdnoises[i],
                lrs_ag[i],
                lrs_con[i],
            )

            test_task = task + "inv" * (
                (test >= config["training"]["n_tests"] // 2) and inverse_task
            )
            community = init_community(
                agent_params_dict,
                p_con,
                use_deepR=connections_params_dict["use_deepR"],
                device=device,
            )
            optimizers, schedulers = init_optimizers(
                community, params_dict, deepR_params_dict
            )

            training_dict = get_training_dict(config)

            train_out = train_community(
                community,
                *loaders,
                optimizers,
                schedulers,
                config=training_dict,
                device=device,
                use_tqdm=2,
            )

            best_test_acc = np.max(train_out["test_accs"])
            mean_d_ags = train_out["deciding_agents"].mean()
            community_states[p_con].append(train_out["best_state"])

            pbar2.set_description(
                desc + f" Best Accuracy : {best_test_acc}, Mean Decision : {mean_d_ags}"
            )

            wandb.log(
                {
                    metric_name: metric
                    for metric, metric_name in zip(
                        [best_test_acc, mean_d_ags], ["Best Test Acc", "Mean Decision"]
                    )
                }
            )

        mkdir_or_save(community_states, save_name, save_path)

    wandb.log_artifact(total_path, name="state_dicts", type="model_saves")
