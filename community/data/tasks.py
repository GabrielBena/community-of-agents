import numpy as np
import torch
import itertools
import torchvision.transforms.functional as TF
from copy import deepcopy


def rotation_conflict_task(datas, digits, n_angles=4):
    device = datas.device
    digits = digits.to(device)
    # Use on data before flatening/creating time_steps !
    possible_angles = np.linspace(0, 360 * (1 - 1 / n_angles), n_angles, dtype=int)

    datas_single_t = datas[:, 0], datas[:, 1]
    angle_values = [
        np.random.choice(possible_angles, datas.shape[0]) for data in datas_single_t
    ]
    diff_choice = lambda i: np.random.choice(possible_angles[possible_angles != i])
    v_choice = np.vectorize(diff_choice)
    angle_values[1] = v_choice(angle_values[0])
    angle_values = torch.stack(
        [torch.FloatTensor(a).to(device) for a in angle_values], 0
    ).int()

    rotation = lambda d, angle: TF.rotate(d.unsqueeze(0), angle.data.item())

    rotated_datas = torch.cat(
        [
            torch.stack([rotation(d, a) for d, a in zip(data, angle)])
            for data, angle in zip(datas_single_t, angle_values)
        ],
        1,
    )
    mask = (angle_values[0] < angle_values[1]).to(device)
    target = torch.where(mask, digits[:, 0], digits[:, 1])
    return rotated_datas, target, angle_values


get_digits = lambda target: target.T


def dec2bin(x):
    bits = int((torch.floor(torch.log2(x)) + 1).max())
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(x.dtype)


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


def dec2bin(x):
    bits = int((torch.floor(torch.log2(x)) + 1).max())
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(x.dtype)


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


def get_single_task(task, target, n_classes=None):

    if n_classes is None:
        n_classes = len(target[:, 0].unique())

    target_mult = lambda tgt: torch.stack(
        [t * (n_classes**i) for (i, t) in enumerate(tgt.T)]
    )

    # composed names of task should be with -
    digits = target.T
    try:
        t = int(task)
        return target[..., t]
    except ValueError:
        pass

    if task == "inv":
        return target.flip(-1)

    elif task in ["none", "both", "all"]:
        return target

    elif "parity" in task:
        if "digits" in task:
            parity = (digits.sum(0)) % 2  # 0 when same parity
            if "both" in task:
                return torch.stack(
                    [
                        torch.where(parity.bool(), digits[0], digits[1]),
                        torch.where(parity.bool(), digits[1], digits[0]),
                    ],
                    -1,
                )
            elif "equal" in task:
                tgt = torch.where(parity.bool(), digits[0], digits[1])
                tgt[
                    (digits[0] == digits[1])
                    | (digits[0] == (digits[1] - 1) % n_classes)
                ] = (n_classes + 1)
                return tgt
            elif "sum" in task:
                return parity
            else:
                return torch.where(parity.bool(), digits[0], digits[1])

        else:
            return target % 2

    elif task == "mult":
        return target_mult(target)

    elif "count" in task:

        if "max" in task:
            new_target = torch.where(
                target.argmax(-1).bool(), target[:, 1], target[:, 0]
            )
            new_target[target[:, 0] == target[:, 1]] = 0
        elif "min" in task:
            new_target = torch.where(
                target.argmin(-1).bool(), target[:, 1], target[:, 0]
            )
            new_target[target[:, 0] == target[:, 1]] = 3
        elif "equal" in task:
            new_target = target[:, 0] != target[:, 1]

        return new_target

    elif task == "max":
        return target.max(-1)[0]

    elif task == "min":
        return target.min(-1)[0]

    elif task == "opposite":
        return n_classes - target - 1

    elif task == "sum":
        return target.sum(-1)

    elif task == "bitand":
        return digits[0] & digits[1]

    elif task == "bitor":
        return digits[0] | digits[1]

    elif "bitxor" in task:

        xor = digits[0] ^ digits[1]

        if "last" in task:
            n_last = int(task.split("-")[-1])
            xor = dec2bin(xor)
            xor = xor[..., -n_last:]
            xor = bin2dec(xor, n_last)

        elif "first" in task:
            n_first = int(task.split("-")[-1])
            xor = dec2bin(xor)
            xor = xor[..., :n_first]
            xor = bin2dec(xor, n_first)

        return xor

    else:
        raise ValueError(
            'Task not recognized, try digit number ("0", "1"), "parity", "parity_digits", "sum", "none" '
        )


def get_task_target(target, task, n_classes, temporal_target=False):
    """
    Returns target for different possible tasks
    Args :
        targets : original digits : size (batch x 2)
        task : task to be conducted :
               digit number ("0", "1"), "parity", "parity_digits_10", "parity_digits_100" or "sum" ...
    """

    if temporal_target:
        targets = get_task_target(target, task, n_classes, False)
        return targets.unique(dim=0)

    elif type(task) is list:
        targets = [get_task_target(target, t, n_classes) for t in task]
        try:
            return torch.stack(targets)
        except (ValueError, RuntimeError) as e:
            return targets

    elif task == "family":
        return get_task_family(target, n_classes)[0]

    else:
        new_target = deepcopy(target)

        # Task can be a combination of subtasks, separated by _
        tasks = task.split("_")

        for task in tasks:
            new_target = get_single_task(task, new_target, n_classes)

        if len(new_target.shape) == 2:
            new_target = new_target.T

        return new_target


def get_factors_list(n_digits, device=torch.device("cpu"), include_singles=False):

    accepted = np.arange(n_digits + 1).tolist()
    if not include_singles:
        accepted.remove(1)

    factors_list = [
        torch.tensor(p, device=device)
        for i, p in enumerate(itertools.product(*[[-1, 0, 1] for _ in range(n_digits)]))
        if torch.tensor(p).sum() in accepted and torch.tensor(p).any()
    ]
    key_f = lambda p: (p == 1).sum() + (p == -1).sum() * 0.1

    return sorted(factors_list, key=key_f)


def get_task_family(target, n_classes_per_dig):

    n_digits = target.shape[-1]
    device = target.device

    factors_list = get_factors_list(n_digits, device)

    task_family = torch.stack(
        [
            (target * f).sum(-1) + (f < 0).sum() * (n_classes_per_dig - 1)
            for i, f in enumerate(factors_list)
        ],
        0,
    )

    return task_family, factors_list, n_classes_per_dig * n_digits


# ------ Continual Learning Tasks ------ :


def get_continual_task(data, target, task, seed, n_tasks=None, n_classes=10):

    done = False

    if seed == "skip" or task == "none" or task.split("_")[-1] == "schedule":
        torch.manual_seed(0)
        return data, target
    else:
        seed = int(seed)
    torch.manual_seed(seed)
    conv_data = data.shape[-1] == data.shape[-2]

    if "label_perm" in task:
        perm = torch.randperm(n_classes) if seed != 0 else torch.arange(n_classes)
        permute = lambda x: perm[x]
        target = target.clone().cpu().apply_(permute).to(data.device)
        done = True

    if "pixel_perm" in task:
        flat = data
        n_pixels = flat.shape[-1]
        perm = torch.randperm(n_pixels) if seed != 0 else torch.arange(n_pixels)
        flat = flat[..., perm]
        if conv_data:
            flat = flat[..., perm, :]
        data = flat.view_as(data)
        done = True

    if "rotate" in task:
        if n_tasks is not None:
            angle = 360 // n_tasks
        else:
            angle = 10
        data = torch.stack([TF.rotate(d, seed * angle) for d in data], 0)
        done = True

    if "select" in task:
        target, _ = get_task_target(target, task=seed % 2)
        done = True

    if "sequential" in task:
        if data.shape[1] != 10 and len(data.shape) != 5:
            if len(data.shape) > 4:
                data, target = (
                    data[:, target == seed % n_classes],
                    target[target == seed % n_classes],
                )
            else:
                data, target = (
                    data[target == seed % n_classes],
                    target[target == seed % n_classes],
                )
        else:
            data, target = data[:, seed % n_classes], target[:, seed % n_classes]

        done = True

        assert data.shape[0] > 0, "Empty tensor returned"

    if not done:
        raise ValueError(
            'Continual task setting not recognized !! Try "pixel_perm", "label_perm", or "rotate"'
        )

    torch.manual_seed(torch.random.seed())
    return data, target
