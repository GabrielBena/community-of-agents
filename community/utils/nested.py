import numpy as np
import torch


def nested_shape(output):
    if hasattr(output, "shape"):
        return [output.shape]
    else:
        return [nested_shape(o) for o in output]


def nested_len(output):
    if hasattr(output, "shape"):
        return len(output.shape)
    else:
        return [nested_len(o) for o in output]


def nested_round(acc):
    try:
        round = np.round(np.array(acc) * 100, 0).astype(float)
        if isinstance(round, np.ndarray):
            round = round.tolist()
        return round
    except TypeError:
        return [nested_round(a) for a in acc]


def nested_sup(acc, best_acc):
    try:
        sup = acc > best_acc
        return sup.all()
    except ValueError:
        return np.array([nested_sup(a, best_acc) for a in acc]).all()
    except (AttributeError, TypeError) as e:
        return sup


def nested_mean(losses):
    try:
        return torch.mean(losses)
    except TypeError:
        return torch.stack([nested_mean(l) for l in losses]).mean()
