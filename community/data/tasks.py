import numpy as np
import torch
import torchvision.transforms.functional as TF


def get_digits(target, n_classes=10) : 

    return target[..., 0], target[..., 1]

def rotation_conflict_task(datas, digits, n_angles=4) : 
    device = datas.device
    digits = digits.to(device)
    #Use on data before flatening/creating time_steps !
    possible_angles = np.linspace(0, 360*(1 - 1/n_angles), n_angles, dtype=int)

    datas_single_t = datas[:, 0], datas[:, 1]
    angle_values = [np.random.choice(possible_angles, datas.shape[0]) for data in datas_single_t]
    diff_choice = lambda i : np.random.choice(possible_angles[possible_angles != i])
    v_choice = np.vectorize(diff_choice)
    angle_values[1] = v_choice(angle_values[0])
    angle_values = torch.stack([torch.FloatTensor(a).to(device) for a in angle_values], 0).int()

    rotation = lambda d, angle : TF.rotate(d.unsqueeze(0), angle.data.item())

    rotated_datas = torch.cat([torch.stack([rotation(d, a) for d, a in zip(data, angle)]) for data, angle in zip(datas_single_t, angle_values)], 1)
    mask = (angle_values[0] < angle_values[1]).to(device)
    target = torch.where(mask, digits[:, 0], digits[:, 1])
    return rotated_datas, target, angle_values

def get_task_target(target, task='parity_digits_10', temporal_target=False) : 
    """
    Returns target for different possible tasks
    Args : 
        targets : original digits : size (batch x 2)
        task : task to be conducted : 
               digit number ("0", "1"), "parity", "parity_digits_10", "parity_digits_100" or "sum" ...
    """

    n_classes = len(target.unique())

    if temporal_target : 

        return get_task_target(target, task, n_classes, False).unique(dim=0)

    else : 

        new_target = None

        #Task can be a combination of subtasks, separated by _
        tasks = task.split('_')

        if 'inv' in tasks : 
            new_target = target.flip(-1)
            digits = digits_0, digits_1 = get_digits(new_target, n_classes)
        else : 
            digits = digits_0, digits_1 = get_digits(target, n_classes)

        parity = (digits_0 + digits_1)%2  #0 when same parity
        target_100 = (digits_0*10 + digits_1)

        if 'parity' in tasks : 
            if 'digits' in tasks : 
                new_target = torch.where(parity.bool(), digits_0, digits_1)
            elif 'both' in tasks : 
                new_target = torch.stack(torch.where(parity.bool(), digits_0, digits_1), torch.where(parity.bool(), digits_1, digits_0))
            else : 
                new_target = parity

        elif '100_class' in tasks: 
            new_target = target_100

        elif 'count' in tasks : 

            if 'max' in tasks : 
                new_target = torch.where(target.argmax(-1).bool(), target[:, 1], target[:, 0])
                new_target[target[:, 0] == target[:, 1]] = 0
            elif 'min' in tasks : 
                new_target = torch.where(target.argmin(-1).bool(), target[:, 1], target[:, 0])
                new_target[target[:, 0] == target[:, 1]] = 3
            elif 'equal' in tasks : 
                new_target = target[:, 0] != target[:, 1]

        elif task == 'none' : 
            new_target = target.transpose(0, 1)

        if 'sum' in tasks : 
            new_target = target.sum(-1)

        try : 
            task = int(tasks[-1])
            if new_target is None : 
                new_target = target
            new_target = new_target[..., task] 
            
        except ValueError : 
            'continue'
            
        if new_target is None : 
            raise ValueError('Task recognized, try digit number ("0", "1"), "parity", "parity_digits", "parity_digits_100" or "sum" ')
                    
        return new_target
#------ Continual Learning Tasks ------ : 

def get_continual_task(data, target, task, seed, n_tasks=None, n_classes=10) : 

    done = False
    if seed == 'skip' or task == 'none' or task.split('_')[-1] == 'schedule': 
        torch.manual_seed(0)
        return data, target
    else : 
        seed = int(seed)
    torch.manual_seed(seed)        
    conv_data = (data.shape[-1] == data.shape[-2])

    if  'label_perm' in task  : 
        perm = torch.randperm(n_classes)if seed != 0 else torch.arange(n_classes)
        permute = lambda x : perm[x]
        target = target.clone().cpu().apply_(permute).to(data.device)
        done = True

    if 'pixel_perm' in task: 
        flat = data
        n_pixels = flat.shape[-1]
        perm = torch.randperm(n_pixels)if seed != 0 else torch.arange(n_pixels)
        flat = flat[..., perm]
        if conv_data : flat = flat[..., perm, :]
        data = flat.view_as(data)
        done = True
        
    if 'rotate' in task :
        if n_tasks is not None : 
            angle = 360//n_tasks
        else : 
            angle = 10
        data = torch.stack([TF.rotate(d, seed*angle) for d in data], 0)      
        done = True

    if 'select' in task : 
        target, _ = get_task_target(target, task=seed%2)
        done = True

    if 'sequential' in task : 
        if data.shape[1] != 10 and len(data.shape) != 5 : 
            if len(data.shape) > 4 : 
                data, target = data[:, target==seed%n_classes], target[target==seed%n_classes]
            else : 
                data, target = data[target==seed%n_classes], target[target==seed%n_classes]
        else : 
             data, target = data[:, seed%n_classes], target[:, seed%n_classes]
             
        done=True

        assert data.shape[0] > 0, "Empty tensor returned"

    if not done :
        raise ValueError('Continual task setting not recognized !! Try "pixel_perm", "label_perm", or "rotate"')

    torch.manual_seed(torch.random.seed())
    return data, target
