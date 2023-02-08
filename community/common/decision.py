import torch
import numpy as np
import torch.nn.functional as F

# ------Decision Making Functions ------:


def random_decision(outputs, p=0.5):
    batchs = outputs.shape[1]
    device = outputs.device
    deciding_agents = torch.rand(batchs).to(device) < p
    mask = torch.einsum(
        "ab, a -> ab", torch.ones_like(outputs[0]), deciding_agents
    ).bool()
    outputs = torch.where(mask, outputs[0, ...], outputs[1, ...])
    return outputs, deciding_agents


def max_decision_2(outputs):
    n_agents = outputs.shape[0]
    max_out = lambda i: torch.max(torch.abs(outputs[i, ...]), axis=-1)
    max_outs, deciding_ags = torch.max(
        torch.stack([max_out(i)[0] for i in range(n_agents)]), axis=0
    )
    mask = torch.einsum("bc, b -> bc", torch.ones_like(outputs[0]), deciding_ags).bool()
    outputs = torch.where(mask, outputs[1], outputs[0])

    return outputs, deciding_ags


def max_decision(outputs):

    if isinstance(outputs, torch.Tensor):

        device = outputs.device
        n_agents = outputs.shape[0]
        max_out = lambda i: torch.max(outputs[i, ...], axis=-1)
        _, deciding_ags = torch.max(
            torch.stack([max_out(i)[0] for i in range(n_agents)]), axis=0
        )
        mask_1 = deciding_ags.unsqueeze(0).unsqueeze(-1).expand_as(outputs)
        mask_2 = torch.einsum(
            "b, b... -> b...",
            torch.arange(n_agents).to(device),
            torch.ones_like(outputs),
        )
        mask = mask_1 == mask_2

        return (outputs * mask).sum(0), deciding_ags
    else:
        try:
            outputs = torch.stack([*outputs], 0)
            return max_decision(outputs)
        except TypeError:
            maxs = [max_decision(out) for out in zip(*outputs)]
            return [list(m) for m in zip(*maxs)]


def max_decision_3(outputs):
    device = outputs.device
    mask = (outputs.max(0)[1].sum(-1) > outputs.shape[-1] // 2).bool()
    mask = mask.unsqueeze(-1).expand_as(outputs[0])
    return torch.where(mask, outputs[1], outputs[0]).to(device), mask[..., 0]


def get_decision(outputs, temporal_decision="last", agent_decision="0", target=None):

    outputs = get_temporal_decision(outputs, temporal_decision)

    try:
        if len(outputs.shape) == 2:
            return outputs, None
    except AttributeError:
        pass

    for ag_decision in agent_decision.split("_"):
        outputs, deciding_ags = get_agent_decision(outputs, ag_decision)

    return outputs, deciding_ags


def get_temporal_decision(outputs, temporal_decision):
    n_steps = len(outputs)
    try:
        deciding_ts = int(temporal_decision)
        outputs = outputs[deciding_ts]
    except ValueError:

        if temporal_decision == "last":
            outputs = outputs[-1]
        elif temporal_decision == "sum":
            outputs = torch.sum(outputs, axis=0)
        elif temporal_decision == "mean":
            outputs = torch.mean(outputs, axis=0)
        elif temporal_decision == None:
            outputs = outputs
        elif temporal_decision == "mid-":
            outputs = outputs[n_steps // 2 - 1]
        elif "mid" in temporal_decision:
            outputs = outputs[n_steps // 2]
        else:
            raise ValueError(
                'temporal decision not recognized, try "last", "sum" or "mean", or time_step of decision ("0", "-1" ) '
            )
    return outputs


def get_agent_decision(outputs, agent_decision, target=None):

    try:
        deciding_ags = int(agent_decision)
        outputs = outputs[deciding_ags]
        try:
            deciding_ags = torch.ones(outputs.shape[0]) * deciding_ags
        except AttributeError:
            deciding_ags = 1

    except ValueError:

        if agent_decision == "max":
            outputs, deciding_ags = max_decision(outputs)

        elif agent_decision == "random":
            outputs, deciding_ags = random_decision(outputs)

        elif agent_decision == "sum":
            outputs = outputs.sum(0)
            deciding_ags = None

        elif agent_decision == "combine":
            outputs = torch.cat(
                [
                    torch.stack(
                        [outputs[1, :, i] + outputs[0, :, j] for i in range(10)], dim=-1
                    )
                    for j in range(10)
                ],
                dim=-1,
            )
            deciding_ags = None

        elif agent_decision in ["both", "all"]:
            deciding_ags = None

        elif agent_decision == "loss":
            assert target is not None, "provide target for decision based on min loss"
            loss, min_idxs = torch.stack(
                [F.cross_entropy(out, target, reduction="none") for out in outputs]
            ).min(0)
            min_idxs = min_idxs.unsqueeze(-1).expand_as(outputs[0])
            outputs = torch.where(~min_idxs.bool(), outputs[0], outputs[1])
            deciding_ags = min_idxs
            return outputs, deciding_ags

        else:
            raise ValueError(
                'Deciding agent not recognized, try agent number ("0", "1"), "max", "random", "both" or "parity" '
            )

    return outputs, deciding_ags
