import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import torch
import wandb
from random import seed as seed_r
from numpy.random import seed as seed_np
from os import environ
from pathlib import Path
from torch.random import manual_seed as seed_tcpu
from torch.cuda import manual_seed_all as seed_tgpu


#------ Connection counting utility functions -----

def nb_connect_community(n_in, n_ag, n_hid, p_con, n_ins=None, out=None) :
    """
    Number of active connecyions in a community of specified parameters
    """
    if out is not None : 
        n_out, p_out = out
        n_con_out = n_out*p_out*n_ag*n_hid
    else : 
        n_con_out = 0
    if n_ins is not None : 
        n_in_total = np.sum(np.array(n_ins))
    else : 
        n_in_total = n_ag*n_in
    return n_in_total*n_hid + n_ag*(1 + (n_ag-1)*p_con)*n_hid**2 + n_con_out

def nb_connect_agent(n_in, n_hid) : 
    """
    Number of active connections in a single agent
    """
    return n_in*n_hid + n_hid**2
   
def calculate_hidden_single_agent(n_in, n_ag, n_hid, p_con, ins=None, out=None) : 
    """
    Calculates number of connections for single agent to be equivalent to a community
    """
    funct = lambda h : nb_connect_agent(n_in*n_ag, h) - nb_connect_community(n_in, n_ag, n_hid, p_con, ins, out)
    n_hidden = fsolve(funct, n_hid)
    return int(n_hidden)

def get_nb_connections(community) : 
    """
    Returns dictionnary of number of active connections of community
    """
    nb_connections = {}
    for tag, c in zip(community.connections.keys(), community.connections.values()) : 
        if c.is_sparse_connect : 
            nb_connected = (c.thetas[0]>0).int().sum()
            nb_connections[tag] = nb_connected
    
    return nb_connections



def get_output_shape(input_shape, kernel_size=[3,3], stride = [1,1], padding=[1,1], dilation=[0,0]):
    if not hasattr(kernel_size, '__len__'):
        kernel_size = [kernel_size, kernel_size]
    if not hasattr(stride, '__len__'):
        stride = [stride, stride]
    if not hasattr(padding, '__len__'):
        padding = [padding, padding]
    if not hasattr(dilation, '__len__'):
        dilation = [dilation, dilation]
    im_height = input_shape[-2]
    im_width = input_shape[-1]
    height = int((im_height + 2 * padding[0] - dilation[0] *
                  (kernel_size[0] - 1) - 1) // stride[0] + 1)
    width = int((im_width + 2 * padding[1] - dilation[1] *
                  (kernel_size[1] - 1) - 1) // stride[1] + 1)
    return [height, width]

    
#------ Plotting utils ------

def plot_running_data(data, ax=None, m=1, **plot_kwargs) :
    try : 
        x, metric = data
    except ValueError : 
        x, metric = range(len(data)), data
    if ax is None :
        ax = plt.subplot(111)
    running = np.convolve(metric, np.ones((m))/(m), mode='valid')
    running_x = np.convolve(x, np.ones((m))/(m), mode='valid')
    ax.plot(running_x, running, **plot_kwargs)
    plt.legend()


def plot_grid(imgs, labels=None, row_title=None, figsize=None, save_loc=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]
        if labels is not None :  labels = [labels]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False, figsize=figsize)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            im = ax.imshow(np.asarray(img[0]), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            if labels is not None :  ax.set_title(labels[row_idx][col_idx])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    fig.colorbar(im)
    
    
    plt.tight_layout()
    if save_loc is not None : 
        plt.savefig(save_loc)
    plt.show()




#------ WandB Utils ------
def get_wandb_artifact(config=None, project='Spec_vs_Sparsity', name='correlations', type='metric', process_config=False, run_id=None, ensure_latest=False) : 
    entity = 'gbena'
    
    api = wandb.Api()

    if process_config and config is not None: 
        config = get_new_config(config, 'config')
    elif config is not None : 
        config['state'] = 'finished'

    print(config)

    runs = api.runs(f'{entity}/{project}', filters=config) # Filtered

    #print(config)
    assert len(runs) > 0, f'No runs found for current filters'

    if run_id is not None : 
        artifacts = [r.logged_artifacts() for r in runs if r.id == run_id]
    else : 
        if len(runs) != 1 : 
            print(f'Warning : {len(runs)} runs found for current filters')#, taking last one by default as no run id is specified')
        artifacts = [r.logged_artifacts() for r in runs]
    
    wanted_artifacts =[[art for art in artifact if name in art.name] for artifact in artifacts]
    wanted_artifacts = [art for art in wanted_artifacts if len(art) > 0]
    assert len(wanted_artifacts) > 0, f'No artifacts found for name {name} or type {type} for {len(runs)} currently filtered runs'
    
    if len(wanted_artifacts) != 1 : 
        print(f'Warning : {len(wanted_artifacts)} runs containing wanted artifact, taking last one by default as no run id or precise name is specified')  
    wanted_artifact = wanted_artifacts[0]

    if len(wanted_artifact) != 1 : 
        print(f'Warning : {len(wanted_artifacts)} artifacts found for single run, taking last one by default')
    wanted_artifact = wanted_artifact[0]

    if ensure_latest : 
        assert 'latest' in wanted_artifact.aliases, 'Artifact found is not the latest, disable ensure_latest to return anyway'

    wanted_artifact.download()
    try : 
        wandb.use_artifact(wanted_artifact.name)
    except wandb.Error : 
        pass
    return torch.load(wanted_artifact.file()), wanted_artifacts, runs

def get_new_config(config, key_prefix='config') : 
    new_config = {}
    for k1, v1 in config.items() : 
        if type(v1) is dict : 
            sub_config = get_new_config(v1, k1)
            new_config.update({key_prefix + '.' + k : v for k,v in sub_config.items()})
        else : 
            new_config[key_prefix + '.' + k1] = v1
    return new_config


def mkdir_or_save_torch(to_save, save_name, save_path) : 
    try : 
        torch.save(to_save, save_path + save_name)
    except FileNotFoundError : 
        path = Path(save_path)
        path.mkdir(save_path, parents=True)
        torch.save(to_save, save_path + save_name)


def get_training_dict(config)  : 

    training_dict = {
        'n_epochs' : config['training']['n_epochs'], 
        'task' : config['task'],
        'global_rewire' : config['model_params']['global_rewire'], 
        'check_gradients' : False, 
        'reg_factor' : 0.,
        'train_connections' : True,
        'decision_params' : config['training']['decision_params'],
        'early_stop' : config['training']['early_stop'] ,
        'deepR_params_dict' : config['optimization']['connections'],
    }

    return training_dict

# ------ Others ------

def rescue_code(function):
    import inspect
    get_ipython().set_next_input("".join(inspect.getsourcelines(function)[0]))

def set_seeds(seed = 42):
    seed_r(seed) 
    seed_np(seed)
    environ['PYTHONHASHSEED'] = str(seed)
    seed_tcpu(seed)
    seed_tgpu(seed)

def check_grad(model, task_id = '0') : 
    for n, p in model.named_parameters() : 
        if 'k_params' in n or 'all_scores' in n : 
            if task_id in n : 
                return check_ind_grad(n, p)
        else : 
            check_ind_grad(n, p)

def check_ind_grad(n, p) : 
    if p.grad is not None : 
        if (p.grad == 0).all() : 
            ''
            print(f'{n}, Zero Grad')
        #else : print(f'{n} : {p.grad}')
    elif p.requires_grad : 
        ''
        print(f'{n}, None Grad')

def is_notebook() : 
    try  : 
        get_ipython()
        notebook = True 
    except NameError : 
        notebook = False
    return notebook