import numpy as np
import torch
import torchvision.transforms.functional as TF


def get_digits(target, n_classes=10) : 
    if len(target.shape) > 1 : 
        return target[..., 0], target[..., 1]
    else :
        d1 = torch.div(target, n_classes, rounding_mode='floor')
        return d1, target - d1*n_classes

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

        digits = digits_1, digits_2 = get_digits(target, n_classes)
        parity = (digits_1 + digits_2)%2  #0 when same parities
        global_target = (digits_1*10 + digits_2)
        global_target_inv = (digits_1 + digits_2*10)

        try : 
            task = int(task)
            return digits[task] 
            
        except ValueError : 

            if task == 'parity' : 
                return parity

            if task == 'inv' : 
                return torch.stack(digits[::-1])
                
            elif 'parity_digits_100' in task : 
                if not 'inv' in task : 
                    return global_target*(1-parity) + global_target_inv*parity
                else : 
                    return global_target*(parity) + global_target_inv*(1 - parity)    
            
            elif 'both_parity_digits' in task : 
                target_1 = digits_1*(1-parity) + digits_2*parity
                target_2 = digits_1*(parity) + digits_2*(1-parity)
                return torch.stack((target_1, target_2))

            elif 'parity_digits' in task :
                if not 'inv' in task :
                    return digits_1*(1-parity) + digits_2*parity
                else :
                    return digits_1*(parity) + digits_2*(1-parity)

            elif task == 'sum' : 
                return digits[0] + digits[1]

            elif 'opposite' == task : 
                return target.flip(1)

            elif '100_class' in task: 
                if 'inv' in task : 
                    return global_target_inv
                else : 
                    return global_target

            elif task == 'none' : 
                return target.transpose(0, 1)

            elif task == 'both' : 
                return torch.stack([digits_1*(1-parity) + digits_2*parity, digits_1*(parity) + digits_2*(1-parity)])

            elif task == 'both_100' : 
                return torch.stack([global_target*(1-parity) + global_target_inv*parity, global_target*(parity) + global_target_inv*(1-parity)])

            elif task == '20_class_parity_based' : 
                parities = [d%2 for d in digits]
                targets = [d + p*10 for d, p in zip(digits, parities[::-1])]
                return torch.stack(targets)
                
            else : 
                raise ValueError('Task recognized, try digit number ("0", "1"), "parity", "parity_digits", "parity_digits_100" or "sum" ')
                       
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
