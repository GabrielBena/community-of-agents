import os
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd

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
    
def get_new_name(name) : 
    new_name = ''
    for n in name.split('.') : 
        new_name += n + '_'
    return new_name[:-1]

class Mask(nn.Module):
    """
    weight mask class, to be applied to any torch Module
    Args : 
        model : model to apply mask onto
        sparsity : % of the weights to select in mask
    """
    def __init__(self, model, sparsity=0.5) : 
        super().__init__()
        self.model = copy.deepcopy(model)
        self.sparsity = sparsity
        self.scores = nn.ParameterDict()
        for name, p in self.model.named_parameters():
            n = get_new_name(name)
            if 'bias' not in n : 
                self.scores[n] = nn.Parameter(torch.Tensor(p.size()), requires_grad=True)
                nn.init.kaiming_uniform_(self.scores[n], a=math.sqrt(5))
            else : 
                self.scores[n] = nn.Parameter(torch.Tensor(p.size()), requires_grad=True)
                nn.init.normal_(self.scores[n], 0, 1e-2)
            p.requires_grad = False
        
    def forward(self, x) : 
        subnets = [GetSubnet_global.apply(s, list(self.scores.values()), self.sparsity) for s in self.scores.values()]
        f_model = copy.deepcopy(self.model)
        for i, (name, p) in enumerate(f_model.named_parameters()) : 
            p *= subnets[i]
            
        return f_model(x)
    
    
class Mask_Community(nn.Module):
    """
    weight mask class, to be applied to Community class Module
    Args : 
        model : Community to apply mask onto
        sparsity : % of the weights to select in mask
    """
    def __init__(self, model, sparsity=0.05, scaling=False) : 
        super().__init__()
        self.model = copy.deepcopy(model)
        self.sparsity = sparsity
        self.scores = nn.ParameterDict()
        self.is_community = False
        for name, p in self.model.agents.named_parameters():
            if not 'ih' in name : #Mask not applied to input weights
                n = get_new_name(name)
                if 'bias' not in n : 
                    self.scores[n] = nn.Parameter(torch.Tensor(p.size()), requires_grad=True)
                    nn.init.kaiming_uniform_(self.scores[n], a=math.sqrt(5))
                else : 
                    self.scores[n] = nn.Parameter(torch.Tensor(p.size()), requires_grad=True)
                    nn.init.normal_(self.scores[n], 0, 1e-2)
            
            p.requires_grad = False
                
    def forward(self, x) : 
        subnets = [GetSubnet_global.apply(s, list(self.scores.values()), self.sparsity) for s in self.scores.values()]
        f_model = copy.deepcopy(self.model)
        i = 0
        for (name, p) in f_model.agents.named_parameters() : 
            if not 'ih' in name :
                p *= subnets[i]
                i += 1
        return f_model(x)
    
def get_proportions(masked_model) : 
    scores = list(masked_model.scores.values())
    d_scores = masked_model.scores
    k = masked_model.sparsity
    identities = torch.cat([torch.ones_like(s.flatten())*n for n, s in enumerate(scores)])
    _, idx = torch.cat([s.flatten() for s in scores]).sort()
    j = int((1 - k) * idx.numel())

    idxs_0 = [idx[:j][identities[idx[:j]]==n] for n, _ in enumerate(scores)]
    idxs_1 = [idx[j:][identities[idx[j:]]==n] for n, _ in enumerate(scores)]
    
    proportion_0 = [idxs.numel()/idx.numel() for idxs in idxs_0]
    proportion_1 = [idxs.numel()/idx.numel() for idxs in idxs_1]
    
    return {name : (p_0, p_1) for name, p_0, p_1 in zip(d_scores.keys(), proportion_0, proportion_1)}
    
    
def get_repartitions(masked_model) : 
    scores = list(masked_model.scores.values())
    k = masked_model.sparsity
    identities = torch.cat([torch.ones_like(s.flatten())*n for n, s in enumerate(scores)])
    _, idx = torch.cat([s.flatten() for s in scores]).sort()
    j = int((1 - k) * idx.numel())
    
    top_idxs = identities[idx[j:]]
    top_idxs = (top_idxs/identities.max()).round()
    
    return idx[j:].cpu().data.numpy(), top_idxs.cpu().data.numpy()