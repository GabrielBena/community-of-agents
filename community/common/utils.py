import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from random import seed as seed_r
from numpy.random import seed as seed_np
from os import environ
from torch.random import manual_seed as seed_tcpu
from torch.cuda import manual_seed_all as seed_tgpu
from community.data.tasks import get_task_target

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



from PIL import Image
import shutil, os

def create_gifs(data, target=None, name='gif', input_size=30, task=None) : 

    if task : target = get_task_target(target, task)
    
    try : 
        shutil.rmtree('gifs/')
        os.mkdir('gifs')
    except FileNotFoundError : 
        os.mkdir('gifs')
        'continue'

    img_list = lambda i : data[..., i, :, :].cpu().data.numpy()*255

    def create_gif(img_list, l, w, name) : 
            
        images_list = [Image.fromarray(img.reshape(w, l)).resize((256, 256)) for img in img_list]
        images_list = images_list # + images_list[::-1] # loop back beginning

        images_list[0].save(
            f'{name}.gif', 
            save_all=True, 
            append_images=images_list[1:],
            loop=0)

    data_size = data.shape[2]
    if target is None : 
        target = ['' for _ in range(data_size)]
    else : 
        target = target.cpu().data.numpy()

    [create_gif(img_list(i), input_size, 2*input_size, 'gifs/' + f'{name}_{i}_{target[i]}') for i in range(min(10, len(target))) ];

    
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