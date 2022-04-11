import torch
import numpy as np

#------Decision Making Functions ------: 

def random_decision(outputs, p=0.5) : 
    batchs = outputs.shape[1]
    device = outputs.device
    deciding_agents = torch.rand(batchs).to(device)<p
    mask = torch.einsum('ab, a -> ab', torch.ones_like(outputs[0]), deciding_agents).bool()
    outputs = torch.where(mask, outputs[0, ...], outputs[1, ...])
    return outputs, deciding_agents

def max_decision_2(outputs) :
    n_agents = outputs.shape[0]
    max_out = lambda i : torch.max(torch.abs(outputs[i,...]), axis=-1)
    max_outs, deciding_ags = torch.max(torch.stack([max_out(i)[0] for i in range(n_agents)]), axis=0)
    mask = torch.einsum('bc, b -> bc', torch.ones_like(outputs[0]), deciding_ags).bool()
    outputs = torch.where(mask, outputs[1], outputs[0])
    
    return outputs, deciding_ags

def max_decision(outputs) : 
    device = outputs.device
    n_agents = outputs.shape[0]
    max_out = lambda i : torch.max(outputs[i,...], axis=-1)
    _, deciding_ags = torch.max(torch.stack([max_out(i)[0] for i in range(n_agents)]), axis=0)
    mask_1 = deciding_ags.unsqueeze(0).unsqueeze(-1).expand_as(outputs)
    mask_2 = torch.einsum('b, bcx -> bcx', torch.arange(n_agents).to(device), torch.ones_like(outputs))
    mask = (mask_1 == mask_2)

    return (outputs*mask).sum(0), deciding_ags

def get_decision(outputs, temporal_decision='last', agent_decision='0', parity=None):
    if temporal_decision == 'last' : 
        outputs = outputs[-1]
    elif temporal_decision == 'sum' : 
        outputs = torch.sum(outputs, axis=0)
    elif temporal_decision == 'mean' :
        outputs = torch.mean(outputs, axis=0)
    elif temporal_decision == None : 
        outputs = outputs
    else : 
        raise ValueError('temporal decision not recognized, try "last", "sum" or "mean" ')
        
    if len(outputs.shape) == 2 : 
        return outputs, None
    
    try : 
        deciding_ags = int(agent_decision)
        outputs = outputs[deciding_ags]
        deciding_ags = torch.ones(outputs.shape[0])*deciding_ags
        
    except ValueError : 
            
        if agent_decision == 'max' : 
            if outputs.shape[0] == 2 : 
                outputs, deciding_ags = max_decision_2(outputs)
            else : 
                outputs, deciding_ags = max_decision(outputs)
            
        elif agent_decision == 'random'  : 
            outputs, deciding_ags = random_decision(outputs)
            
        elif agent_decision == 'sum' : 
            outputs = outputs.sum(0)
            deciding_ags = None
        
        elif agent_decision == 'combine' : 
            outputs = torch.cat([torch.stack([outputs[1, :, i] + outputs[0, :, j] for i in range(10)], dim=-1) for j in range(10)], dim=-1)
            deciding_ags = None
        
        elif agent_decision == 'both' : 
            deciding_ags = None
            
        else : 
            raise ValueError('Deciding agent not recognized, try agent number ("0", "1"), "max", "random", "both" or "parity" ')
            
        
    return outputs, deciding_ags