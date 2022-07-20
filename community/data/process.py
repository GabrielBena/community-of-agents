import torch
import numpy as np
from .tasks import rotation_conflict_task, get_task_target

def temporal_data(data, n_steps=2, conv_com=False) : 
    """
    Stack data in time for use with RNNs
    """
    flatten = not conv_com
    is_list = type(data) is list
    if flatten and not is_list: 
        if data.shape[1] == 1 : 
            data = data.flatten(start_dim=1)
        else : 
            data = data.flatten(start_dim=2)
        
    data = [data for _ in range(n_steps)]
    if not is_list : data = torch.stack(data)
    if not is_list and len(data.shape) > 3 and not conv_com: 
        data = data.transpose(1, 2)

    return data

def varying_temporal_data(data, target, n_steps=3, conv_com=False, transpose_and_cat=False) : 
    """
    Stack data in time for use with RNNs
    """
    if not transpose_and_cat : 
            
        flatten = not conv_com
        is_list = type(data) is list
        if flatten and not is_list: 
            if data.shape[1] == 1 : 
                data = data.flatten(start_dim=1)
            else : 
                data = data.flatten(start_dim=2)

        batch_size = data.shape[0]
        def random_shuffle(data, target, step) : 
            samples = np.arange(batch_size)
            if step>0 : np.random.shuffle(samples)
            return data[samples], target[samples]
        
        shuffles = [random_shuffle(data, target, step) for step in range(n_steps)]
        data = torch.stack([s[0] for s in shuffles])
        target = torch.stack([s[1] for s in shuffles])
        if len(data.shape) > 3 : 
            data = data.transpose(1, 2)
        return data, target
    else : 
        perm_b = torch.torch.randperm(data.shape[0])
        data1, data2 = data, data.flip(1)[perm_b]
        target = torch.cat([target.unsqueeze(0).repeat(n_steps, 1, 1), target.flip(1)[perm_b].unsqueeze(0).repeat(n_steps, 1, 1)])
        data1, data2 = temporal_data(data, n_steps, conv_com), temporal_data(data2, n_steps, conv_com)
        data =  torch.cat([data1, data2], 0)
        perm_t = torch.randperm(target.shape[0])
        return data[perm_t], target[perm_t]

def flatten_double_data(data) :
    """
    Flatten double mnist data
    """
    return data.transpose(1, 2).flatten(start_dim=2)


def process_data(data, target, task, conv_com=False, symbols=False, varying_temporal=False, n_steps=2) : 

    if symbols : 
        data =  data.permute(1, 2, 0, 3, 4).float()
        if not conv_com : data = data.flatten(start_dim=-2)         

    elif 'rotation_conflict' in task : 
        try : 
            n_angles = int(task.split('_')[-1])
        except ValueError : 
            n_angles = 4
        data, target, _ = rotation_conflict_task(data, target, n_angles)
        data = temporal_data(data, n_steps=n_steps, conv_com=conv_com)
    else : 
        if not varying_temporal : 
            data = temporal_data(data, n_steps=n_steps, conv_com=conv_com)  
        else : 
            data, target = varying_temporal_data(data, target, n_steps=n_steps, conv_com=conv_com, transpose_and_cat=True)   
            
    target = get_task_target(target, task, temporal_target=(len(target.shape)>2) )

    return data, target