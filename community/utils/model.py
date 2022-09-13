import numpy as np
from scipy.optimize import fsolve

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
