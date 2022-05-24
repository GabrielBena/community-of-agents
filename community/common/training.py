import warnings
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm.notebook import tqdm as tqdm_n
from tqdm import tqdm
import wandb

from .init import init_community, init_optimizers
from .utils import check_grad, is_notebook
from .wandb_utils import get_training_dict, mkdir_or_save_torch

from .models.ensembles import ConvCommunity
from .decision import get_decision
from ..data.process import process_data

from deepR.models import step_connections
                    
#------ Training and Testing functions ------
            
def train_community(model, train_loader, test_loader, optimizers, schedulers=None,
                    config=None, trials=(True, True), joint_training=False,
                    use_tqdm=True, device=torch.device('cuda')):
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

    assert config is not None or wandb.run is not None, 'Provide training config or run with WandB'

    if config is None : 
        config = get_training_dict(wandb.config)
    
    #----Config----
    n_epochs = config['n_epochs']
    task = config['task']
    reg_factor = config['reg_factor']
    train_connections = config['train_connections']
    check_gradients = config['check_gradients']
    global_rewire = config['global_rewire']
    decision_params = config['decision_params']
    min_acc = config['stopping_acc']
    early_stop = config['early_stop']
    deepR_params_dict = config['deepR_params_dict']
    #--------------

    reg_loss = reg_factor>0.

    if type(use_tqdm) is int : 
        position = use_tqdm
        use_tqdm = True
    elif use_tqdm : 
        position = 0

    conv_com = type(model) is ConvCommunity
    if model.is_community and train_connections: 
        thetas_list = [c.thetas[0] for c in model.connections.values() if c.is_deepR_connect]
        sparsity_list = [c.sparsity_list[0] for c in model.connections.values() if c.is_deepR_connect]
        if thetas_list == [] : 
            #print('Empty Thetas List !!')
            warnings.warn("Empty Theta List", Warning)
    
    descs = ['' for _ in range(2)]
    desc = lambda descs : descs[0] + descs[1]  
    train_losses, train_accs = [], []
    test_accs, test_losses = [], []
    deciding_agents = []
    best_loss, best_acc = 1e10, 0.
    
    pbar = range(n_epochs)
    if use_tqdm : 
        tqdm_f = tqdm_n if notebook else tqdm
        pbar = tqdm_f(pbar, position=position, leave=None, desc='Train Epoch:')
            
    for epoch in pbar : 
        if training : 

            model.train()        
            for batch_idx, (data, target) in enumerate(train_loader):
                if type(data) is list : 
                    data, target = [d.to(device) for d in data], target.to(device)
                else : 
                    data, target = data.to(device), target.to(device)

                #Forward pass

                #Task Selection
                data, target = process_data(data, target, task, conv_com)

                optimizer_agents.zero_grad()
                if optimizer_connections : optimizer_connections.zero_grad()

                output, *_ = model(data)
                output, deciding_ags = get_decision(output, *decision_params, target=target)

                #if deciding_ags is not None and deciding_ags.shape[0]==train_loader.batch_size: deciding_agents.append(deciding_ags.cpu().data.numpy())
                
                if len(output.shape)==2 : 
                    output = output.unsqueeze(0)     
                if len(target.shape)==1 : 
                    target = target.unsqueeze(0)
                
                if joint_training : 
                    target = target.expand([2] + list(target.shape[1:]))
                    take_min_loss = False
                elif target.shape[0] != output.shape[0] : 
                    take_min_loss = True
                else : 
                    take_min_loss = False

                #print(output.shape, target.shape, take_min_loss)

                target = target.contiguous()

                assert target.shape == output.shape[:-1] or take_min_loss or joint_training, print('Target and output shapes not compatible :', target.shape, output.shape)
                
                factors = [1, 1]

                if take_min_loss : 
                    loss, min_idxs = torch.stack([F.cross_entropy(output[0], tgt, reduction='none') for tgt in target]).min(0)
                    loss = loss.mean()
                else  : 
                    loss = torch.stack([F.cross_entropy(out, tgt)*f for (out, tgt, f) in zip(output, target, factors)]).mean()

                if reg_loss : 
                    reg = F.mse_loss(deciding_ags.float().mean(), torch.full_like(deciding_ags.float().mean(), 0.5))
                    loss += reg*reg_factor

                pred = output.argmax(dim=-1, keepdim=True)  # get the index of the max log-probability
                if take_min_loss : 
                    target = torch.where(~min_idxs.bool(), target[0], target[1])

                correct = pred.eq(target.view_as(pred))
                if not joint_training : 
                    correct = correct.sum().cpu().data.item()
                    train_accs.append(correct/target.numel())
                else : 
                    correct = correct.flatten(start_dim=1).sum(1).cpu().data.numpy()
                    train_accs.append(correct/target[0].numel())
                
                loss.backward()

                if check_gradients : 
                    check_grad(model)

                train_losses.append(loss.cpu().data.item())
                
                #Apply gradients on agents weights
                optimizer_agents.step()

                #Apply gradient for sparse connections and rewire
                if model.is_community and train_connections: 
                    nb_new_con = step_connections(model, optimizer_connections, global_rewire, thetas_list,
                                                  sparsity_list, deepR_params_dict=deepR_params_dict)
                else : 
                    nb_new_con = 0
                acc = train_accs[-1]
                descs[0] = str('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.3f}, Accuracy: {}%, New Cons: {}'.format(
                        epoch, batch_idx * train_loader.batch_size, len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item(),
                        (np.round(100*a) for a in acc) if type(acc) is list else np.round(100*acc),
                        nb_new_con))

                if use_tqdm: 
                    pbar.set_description(desc(descs))
            
        if testing : 
            descs[1], loss, acc, deciding_ags = test_community(model, device, test_loader, decision_params=decision_params, task=task, joint_training=joint_training)  
                                                 
            if loss < best_loss : 
                best_loss = loss
                best_state = copy.deepcopy(model.state_dict())

            try : 
                if acc > best_acc : 
                    best_acc = acc
            except ValueError : 
                 if( acc > best_acc).all() : 
                    best_acc = acc

            test_losses.append(loss)
            test_accs.append(acc)
            deciding_agents.append(deciding_ags)

        else : 
            best_state = None

        if use_tqdm : 
            pbar.set_description(desc(descs))
            
        if schedulers is not None : 
            for sch in schedulers : sch.step()

        results = {
            'train_losses' : np.array(train_losses),
            'train_accs' : np.array(train_accs),
            'test_losses' : np.array(test_losses),
            'test_accs' : np.array(test_accs),
            'deciding_agents' : np.array(deciding_agents),
            'best_state' : best_state
        }

        # Stop training if loss doesn't go down or if min_acc is reached
        if epoch >= 4 : 
            if results['test_losses'][-4:].argmin() == 0 and early_stop : 
                #print('Stopping Training (Early Stop), loss hasn\'t improved in 4 epochs')
                return results    
            if min_acc is not None :                 
                try :  
                    if (best_acc>=min_acc) :
                        #print(f'Stopping Training, Minimum accuracy of {min_acc} reached')
                        return results
                except ValueError : 
                    if (best_acc>=min_acc).all() :
                        #print(f'Stopping Training, Minimum accuracy of {min_acc} reached')
                        return results

    return results
                   
def test_community(model, device, test_loader, decision_params=('last', 'max'), task='parity_digits', verbose=False, seed=None, joint_training=False):
    """
    Testing function for community of agents
    """
    model.eval()
    conv_com = type(model) is ConvCommunity
    test_loss = 0
    correct = 0
    acc = 0
    deciding_agents = []
    if seed is not None : 
        torch.manual_seed(seed)
    with torch.no_grad():
        for data, target in test_loader:

            if type(data) is list : 
                data, target = [d.to(device) for d in data], target.to(device)
            else : 
                data, target = data.to(device), target.to(device)
                            
            data, target = process_data(data, target, task, conv_com)

            output, *_ = model(data)
            output, deciding_ags = get_decision(output, *decision_params, target=target)
            if deciding_ags is not None and deciding_ags.shape[0]==test_loader.batch_size: deciding_agents.append(deciding_ags.cpu().data.numpy())
            
            if len(output.shape)==2 : 
                    output = output.unsqueeze(0)     
            if len(target.shape)==1 : 
                target = target.unsqueeze(0)
            if joint_training : 
                target = target.expand([2] + list(target.shape[1:]))
                take_min_loss = False
            elif target.shape[0] != output.shape[0] : 
                try : 
                    target = target.transpose(0, 1)
                    assert target.shape == output.shape[:-1]
                    take_min_loss = False
                except AssertionError : 
                    target = target.transpose(0, 1)
                    take_min_loss = True
            else : 

                take_min_loss = False

            assert target.shape == output.shape[:-1] or take_min_loss or joint_training, print(target.shape, output.shape)
                
            if take_min_loss : 
                loss, min_idxs = torch.stack([F.cross_entropy(output[0], tgt, reduction='none') for tgt in target]).min(0)
                test_loss += loss.sum()
                target = torch.where(~min_idxs.bool(), target[0], target[1])
            else : 
                test_loss += torch.sum(torch.stack([F.cross_entropy(out, tgt, reduction='sum') for (out, tgt) in zip(output, target)]))

            pred = output.argmax(dim=-1, keepdim=True)  # get the index of the max log-probability

            c = pred.eq(target.view_as(pred))
            if not joint_training : 
                correct += c.sum().cpu().data.item()
                acc += c.sum().cpu().data.item()/target.numel()
            else : 
                correct += c.flatten(start_dim=1).sum(1).cpu().data.numpy()
                acc += c.flatten(start_dim=1).sum(1).cpu().data.numpy()/target[0].numel()

    test_loss /= len(test_loader.dataset) * target.shape[0]
    acc /= len(test_loader)

    deciding_agents = np.array(deciding_agents)
    
    desc = str(' | Test set: Loss: {:.3f}, Accuracy: {}/{} ({}%), Mean decider: {:.2f}'.format(
            test_loss, np.sum(correct), (len(test_loader.dataset)*(target.shape[0])),
            (np.round(100*a) for a in acc) if type(acc) is list else np.round(100*acc),
            deciding_agents.mean()))
    
    if verbose : print(desc)
        
    return desc, test_loss.cpu().data.item(), acc, deciding_agents

def compute_trained_communities(p_cons, loaders, device=torch.device('cuda'), notebook=False, 
                                config=None) : 
    """
    Trains and saves model for all levels of sparsity of inter-connections
    Args : 
        p_cons : list of sparisities of inter-connections
        loaders : training and testing data-loaders
        config : config dict, created in main or from WandB (sweeps)
        save_name : file name to be saved
        
    """
    if wandb.run is not None :
        config = wandb.config
    else : 
        assert config is not None, 'Provide configuration dict or run using WandB'

    task = config['task']
    print(f'Starting training on {task}')
    params_dict, deepR_params_dict = tuple(config['optimization'].values())
    agent_params_dict = config['model_params']['agents_params']
    connections_params_dict = config['model_params']['connections_params']

    inverse_task = 'digits' in task and config['training']['inverse_task']

    l = 0
    save_path =  config['saves']['models_save_path']
    save_name = config['saves']['models_save_name'] 
    total_path = save_path + save_name

    print(total_path)

    try :
        community_states = torch.load(total_path)
        start = len(community_states.keys())
        print('Warning : file already exists, picking training up from last save')
        
    except FileNotFoundError : 
        community_states = {}
        start = 0

    start = 0
    
    gdnoises = deepR_params_dict['gdnoise']*(1-p_cons)
    lrs_ag = [params_dict['lr']]*len(p_cons)
    lrs_con = np.geomspace(deepR_params_dict['lr'], deepR_params_dict['lr']/100, len(p_cons)) 

    notebook = is_notebook()

    tqdm_f = tqdm_n if notebook else tqdm
    pbar1 = tqdm_f(p_cons[start:], position=0, desc='Model Sparsity : ', leave=None)

    for i, p_con in enumerate(pbar1) : 
        community_states[p_con] = []
        desc = 'Model Trials'
        pbar2 = tqdm_f(range(config['training']['n_tests']), position=1, desc=desc, leave=None)

        for test in pbar2 : 
            
            deepR_params_dict['gdnoise'], params_dict['lr'], deepR_params_dict['lr'] = gdnoises[i], lrs_ag[i], lrs_con[i]  
            
            test_task = task + 'inv'*((test >= config['training']['n_tests']//2) and inverse_task)
            community = init_community(agent_params_dict, p_con, use_deepR=connections_params_dict['use_deepR'], device=device)
            optimizers, schedulers = init_optimizers(community, params_dict, deepR_params_dict)

            training_dict = get_training_dict(config)
            
            train_out = train_community(community, *loaders, optimizers, schedulers,                          
                                                     config=training_dict, device=device, use_tqdm=2)
            
            best_test_acc = np.max(train_out['test_accs'])
            mean_d_ags = train_out['deciding_agents'].mean()
            community_states[p_con].append(train_out['best_state'])
            
            pbar2.set_description(desc + f' Best Accuracy : {best_test_acc}, Mean Decision : {mean_d_ags}')
            
            wandb.log({metric_name : metric for metric, metric_name in zip([best_test_acc, mean_d_ags], ['Best Test Acc', 'Mean Decision'])})

        mkdir_or_save_torch(community_states, save_name, save_path)
        
    wandb.log_artifact(total_path, name='state_dicts', type='model_saves') 