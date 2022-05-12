import torch
import torch.autograd as autograd
import numpy as np

# ------ Weight Masks Models ------

"""
Weight Masks training in PyTorch
Based on  : 
"What's Hidden in a Randomly Weighted Neural Network?"
Vivek Ramanujan, Mitchell Wortsman, Aniruddha Kembhavi, Ali Farhadi, Mohammad Rastegari
(https://arxiv.org/abs/1911.13299)
https://github.com/RAIVNLab/hidden-networks
"""

class GetSubnet_global(autograd.Function):
    """
    Variation of the original sorting function, select top k% scoring parameter globally, instead of layer-per-layer
    Args : 
        score : single parameter score to be selected on (needed for backward pass to have only one corresponding gradient)
        scores : total list of scores, corresponding to all parameters of model
        k : % of weights to be selected
    
    """
    @staticmethod
    def forward(ctx, score, scores, k):
        score_idx = [s is score for s in scores].index(True)
        identities = torch.cat([torch.ones_like(s.flatten())*n for n, s in enumerate(scores)])
        _, idx = torch.cat([s.flatten() for s in scores]).sort()
        j = int((1 - k) * idx.numel())

        idxs_0 = [idx[:j][identities[idx[:j]]==n] for n, _ in enumerate(scores)]
        idxs_1 = [idx[j:][identities[idx[j:]]==n] for n, _ in enumerate(scores)]
        outs = []
        substract = [0]
        substract.extend([s.numel() for s in scores[:-1]])
        for i, (s, idx_0, idx_1) in enumerate(zip(scores, idxs_0, idxs_1)):
            sub = np.sum(substract[:i+1])
            out = s.clone()
            flat_out = out.flatten()
            flat_out[(idx_0-sub).long()] = 0
            flat_out[(idx_1-sub).long()] = 1
            
            #print(out.requires_grad)
            outs.append(out)
            
        return outs[score_idx]

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None, None
    
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None
    