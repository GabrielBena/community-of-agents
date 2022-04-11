import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from itertools import chain
from collections import namedtuple
import warnings
import sys, os
import copy

DCLL_path = 'DCLL/decolle-public/'
sys.path.append(DCLL_path)
sys.path.append(DCLL_path + 'decolle')

import decolle
import base_model, lenet_decolle_model
from decolle.utils import train, test, accuracy, load_model_from_checkpoint, save_checkpoint, write_stats, get_output_shape

DeepR_path = 'DEEP-R/'
sys.path.append(DeepR_path)
import deepR_torch as deepR

class SmoothStep(torch.autograd.Function):
    '''
    Modified from: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    '''

    @staticmethod
    def forward(aux, x, thr=0):
        aux.save_for_backward(x)
        return (x >=thr).type(x.dtype)

    def backward(aux, grad_output):
        # grad_input = grad_output.clone()
        input, = aux.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= -.5] = 0
        grad_input[input > .5] = 0
        return grad_input
    
smooth_step = SmoothStep().apply
sigmoid = nn.Sigmoid()

class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """
    
    scale = 100.0 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input, thr=0):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > thr] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad
    
# here we overwrite our naive spike function by the "SurrGradSpike" nonlinearity which implements a surrogate gradient
super_spike  = SurrGradSpike.apply

def state_detach(state):
    for s in state:
        s.detach_()
        
class LIFLayerConnect(nn.Module):
    NeuronState = namedtuple('NeuronState', ['P', 'Q', 'R', 'S'])
    #sg_function = smooth_step
    sg_function = super_spike

    def __init__(self, layer, ad_layer=None, rec_layer=None, alpha=.9, alpharp=.65, wrp=1.0, beta=.85, deltat=1000, do_detach=True):
        '''
        deltat: timestep in microseconds (not milliseconds!)
        '''
        super().__init__()
        self.base_layer = layer
        self.deltat = deltat
        self.dt = deltat/1e-6
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.tau_m = torch.nn.Parameter(1. / (1 - self.alpha), requires_grad=False)
        self.tau_s = torch.nn.Parameter(1. / (1 - self.beta), requires_grad=False)
        self.alpharp = alpharp
        self.wrp = wrp
        self.state = None
        self.do_detach = do_detach

    def cuda(self, device=None):
        '''
        Handle the transfer of the neuron state to cuda
        '''
        self = super().cuda(device)
        self.state = None
        self.base_layer = self.base_layer.cuda()
        return self

    def cpu(self, device=None):
        '''
        Handle the transfer of the neuron state to cpu
        '''
        self = super().cpu(device)
        self.state = None
        self.base_layer = self.base_layer.cpu()
        return self

    @staticmethod
    def reset_parameters(layer):
        if hasattr(layer, 'out_channels'):
            conv_layer = layer
            n = conv_layer.in_channels
            for k in conv_layer.kernel_size:
                n *= k
            stdv = 1. / np.sqrt(n) / 250
            conv_layer.weight.data.uniform_(-stdv * 1e-2, stdv * 1e-2)
            if conv_layer.bias is not None:
                conv_layer.bias.data.uniform_(-stdv, stdv)
        elif hasattr(layer, 'out_features'): 
            if hasattr(layer, 'thetas') : 
                layer.thetas[0].data[:]*=1e-4
            elif hasattr(layer, 'weight') : 
                layer.weight.data[:]*=0
            if layer.bias is not None:
                layer.bias.data.uniform_(-1e-3,1e-3)
        else:
            warnings.warn('Unhandled layer type, not resetting parameters')
    
    @staticmethod
    def get_out_channels(layer):
        '''
        Wrapper for returning number of output channels in a LIFLayer
        '''
        if hasattr(layer, 'out_features'):
            return layer.out_features
        elif hasattr(layer, 'out_channels'): 
            return layer.out_channels
        elif hasattr(layer, 'get_out_channels'): 
            return layer.get_out_channels()
        else: 
            raise Exception('Unhandled base layer type')
    
    @staticmethod
    def get_out_shape(layer, input_shape):
        if hasattr(layer, 'out_channels'):
            return get_output_shape(input_shape, 
                                    kernel_size=layer.kernel_size,
                                    stride = layer.stride,
                                    padding = layer.padding,
                                    dilation = layer.dilation)
        elif hasattr(layer, 'out_features'): 
            return []
        elif hasattr(layer, 'get_out_shape'): 
            return layer.get_out_shape()
        else: 
            raise Exception('Unhandled base layer type')

    def init_state(self, Sin_t):
        dtype = Sin_t.dtype
        device = self.get_device()
        input_shape = list(Sin_t.shape)
        out_ch = self.get_out_channels(self.base_layer)
        out_shape = self.get_out_shape(self.base_layer, input_shape)
        self.state = self.NeuronState(P=torch.zeros(input_shape).type(dtype).to(device),
                                      Q=torch.zeros(input_shape).type(dtype).to(device),
                                      R=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      S=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device))

    def init_parameters(self):
        self.reset_parameters(self.base_layer)
        

    def forward(self, Sin_t):
        if self.state is None:
            self.init_state(Sin_t)

        state = self.state
        Q = self.beta * state.Q + (self.tau_s) * Sin_t
        P = self.alpha * state.P + (self.tau_m) * state.Q  
        R = state.R
        #R = self.alpharp * state.R - state.S * self.wrp
        
        U = self.base_layer(P)
        
        S = state.S
        #S = self.sg_function(U)

        self.state = self.NeuronState(P=P, Q=Q, R=R, S=S)
        if self.do_detach: 
            state_detach(self.state)
        return state.S, U

    def get_output_shape(self, input_shape):
        layer = self.base_layer
        if hasattr(layer, 'out_channels'):
            im_height = input_shape[-2]
            im_width = input_shape[-1]
            height = int((im_height + 2 * layer.padding[0] - layer.dilation[0] *
                          (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1)
            weight = int((im_width + 2 * layer.padding[1] - layer.dilation[1] *
                          (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1)
            return [height, weight]
        else:
            return layer.out_features
    
    def get_device(self):
        return self.base_layer.weight[0].device
        
class LIFRecLayer(nn.Module):
    NeuronState = namedtuple('NeuronState', ['P', 'Q', 'R', 'S'])
    #sg_function = smooth_step
    sg_function = super_spike

    def __init__(self, layer, rec_layer, alpha=.9, alpharp=.65, wrp=1.0, beta=.85, deltat=1000, do_detach=True):
        '''
        deltat: timestep in microseconds (not milliseconds!)
        '''
        super().__init__()
        self.base_layer = layer
        self.rec_layer = rec_layer
        
        self.deltat = deltat
        self.dt = deltat/1e-6
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.tau_m = torch.nn.Parameter(1. / (1 - self.alpha), requires_grad=False)
        self.tau_s = torch.nn.Parameter(1. / (1 - self.beta), requires_grad=False)
        
        self.alpha_rec = torch.tensor(alpha)
        self.beta_rec = torch.tensor(beta)
        self.tau_m_rec = torch.nn.Parameter(1. / (1 - self.alpha), requires_grad=False)
        self.tau_s_rec = torch.nn.Parameter(1. / (1 - self.beta), requires_grad=False)
        self.rec_mask = torch.eye(self.rec_layer.weight.data.shape[0], dtype = bool).to(self.get_device())

        
        self.alpharp = alpharp
        self.wrp = wrp
        self.state = None
        self.rec_state = None
        self.do_detach = do_detach

    def cuda(self, device=None):
        '''
        Handle the transfer of the neuron state to cuda
        '''
        self = super().cuda(device)
        self.state = None
        self.rec_state = None
        self.base_layer = self.base_layer.cuda()
        self.rec_layer = self.rec_layer.cuda()
        return self

    def cpu(self, device=None):
        '''
        Handle the transfer of the neuron state to cpu
        '''
        self = super().cpu(device)
        self.state = None
        self.rec_state = None
        self.base_layer = self.base_layer.cpu()
        self.rec_layer = self.rec_layer.cpu()
        
        return self

    @staticmethod
    def reset_parameters(layer):
        if hasattr(layer, 'out_channels'):
            conv_layer = layer
            n = conv_layer.in_channels
            for k in conv_layer.kernel_size:
                n *= k
            stdv = 1. / np.sqrt(n) / 250
            conv_layer.weight.data.uniform_(-stdv * 1e-2, stdv * 1e-2)
            if conv_layer.bias is not None:
                conv_layer.bias.data.uniform_(-stdv, stdv)
        elif hasattr(layer, 'out_features'): 
            layer.weight.data[:]*=0
            #layer.weight.data.uniform_(-1e-4,1e-4)
            if layer.bias is not None:
                layer.bias.data.uniform_(-1e-3,1e-3)
        else:
            warnings.warn('Unhandled layer type, not resetting parameters')
    
    @staticmethod
    def get_out_channels(layer):
        '''
        Wrapper for returning number of output channels in a LIFLayer
        '''
        if hasattr(layer, 'out_features'):
            return layer.out_features
        elif hasattr(layer, 'out_channels'): 
            return layer.out_channels
        elif hasattr(layer, 'get_out_channels'): 
            return layer.get_out_channels()
        else: 
            raise Exception('Unhandled base layer type')
    
    @staticmethod
    def get_out_shape(layer, input_shape):
        if hasattr(layer, 'out_channels'):
            return get_output_shape(input_shape, 
                                    kernel_size=layer.kernel_size,
                                    stride = layer.stride,
                                    padding = layer.padding,
                                    dilation = layer.dilation)
        elif hasattr(layer, 'out_features'): 
            return []
        elif hasattr(layer, 'get_out_shape'): 
            return layer.get_out_shape()
        else: 
            raise Exception('Unhandled base layer type')

    def init_state(self, Sin_t):
        dtype = Sin_t.dtype
        device = self.base_layer.weight.device
        input_shape = list(Sin_t.shape)
        out_ch = self.get_out_channels(self.base_layer)
        out_shape = self.get_out_shape(self.base_layer, input_shape)
        self.state = self.NeuronState(P=torch.zeros(input_shape).type(dtype).to(device),
                                      Q=torch.zeros(input_shape).type(dtype).to(device),
                                      R=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      S=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device))
        
        
    def init_rec_state(self, Sh_t) : 
        dtype = Sh_t.dtype
        device = self.rec_layer.weight.device
        input_shape = list(Sh_t.shape)
        out_ch = self.get_out_channels(self.rec_layer)
        out_shape = self.get_out_shape(self.rec_layer, input_shape)
        self.rec_state = self.NeuronState(P=torch.zeros(input_shape).type(dtype).to(device),
                                      Q=torch.zeros(input_shape).type(dtype).to(device),
                                      R=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      S=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device))

    def init_parameters(self):
        self.reset_parameters(self.base_layer)
        self.reset_parameters(self.rec_layer)
        
    def compute_state(self, Sin_t) : 
        if self.state is None:
            self.init_state(Sin_t)

        state = self.state
        for s in state : 
            #print('Base : ', s.shape)
            ''
        Q = self.beta * state.Q + self.tau_s * Sin_t
        P = self.alpha * state.P + self.tau_m * state.Q  
        R = self.alpharp * state.R - state.S * self.wrp
        U = self.base_layer(P) + R
        return P, Q, R, U
    
    def compute_rec_state(self, Sh_t) : 

        self.rec_layer.weight.data[self.rec_mask] *= 0
        
        if self.rec_state is None:
            self.init_rec_state(Sh_t)

        state = self.rec_state

        Q = self.beta_rec * state.Q + self.tau_s_rec * Sh_t
        P = self.alpha_rec * state.P + self.tau_m_rec * state.Q  
        R = state.R
        U = self.rec_layer(P)
        return P, Q, R, U
    
    
    def forward(self, Sin_t, Uconnect=0):
        
        base_state = self.compute_state(Sin_t)
        rec_state = self.compute_rec_state(self.state.S)        
        
        U = base_state[-1] + rec_state[-1] + Uconnect
        S = self.sg_function(U)

        self.state = self.NeuronState(P=base_state[0], Q=base_state[1], R=base_state[2], S=S)
        self.rec_state = self.NeuronState(P=rec_state[0], Q=rec_state[1], R=rec_state[2], S=S)

        if self.do_detach: 
            base_model.state_detach(self.state)
            base_model.state_detach(self.rec_state)
            
        return S, U

    def get_output_shape(self, input_shape):
        layer = self.base_layer
        if hasattr(layer, 'out_channels'):
            im_height = input_shape[-2]
            im_width = input_shape[-1]
            height = int((im_height + 2 * layer.padding[0] - layer.dilation[0] *
                          (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1)
            weight = int((im_width + 2 * layer.padding[1] - layer.dilation[1] *
                          (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1)
            return [height, weight]
        else:
            return layer.out_features
    
    def get_device(self):
        return self.base_layer.weight.device

class LIFLayer(nn.Module):
    NeuronState = namedtuple('NeuronState', ['P', 'Q', 'R', 'S'])
    #sg_function = smooth_step
    sg_function = super_spike

    def __init__(self, layer, ad_layer=None, rec_layer=None, alpha=.9, alpharp=.65, wrp=1.0, beta=.85, deltat=1000, do_detach=True):
        '''
        deltat: timestep in microseconds (not milliseconds!)
        '''
        super(LIFLayer, self).__init__()
        self.base_layer = layer
        self.deltat = deltat
        self.dt = deltat/1e-6
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.tau_m = torch.nn.Parameter(1. / (1 - self.alpha), requires_grad=False)
        self.tau_s = torch.nn.Parameter(1. / (1 - self.beta), requires_grad=False)
        self.alpharp = alpharp
        self.wrp = wrp
        self.state = None
        self.do_detach = do_detach

    def cuda(self, device=None):
        '''
        Handle the transfer of the neuron state to cuda
        '''
        self = super().cuda(device)
        self.state = None
        self.base_layer = self.base_layer.cuda()
        return self

    def cpu(self, device=None):
        '''
        Handle the transfer of the neuron state to cpu
        '''
        self = super().cpu(device)
        self.state = None
        self.base_layer = self.base_layer.cpu()
        return self

    @staticmethod
    def reset_parameters(layer):
        if hasattr(layer, 'out_channels'):
            conv_layer = layer
            n = conv_layer.in_channels
            for k in conv_layer.kernel_size:
                n *= k
            stdv = 1. / np.sqrt(n) / 250
            conv_layer.weight.data.uniform_(-stdv * 1e-2, stdv * 1e-2)
            if conv_layer.bias is not None:
                conv_layer.bias.data.uniform_(-stdv, stdv)
        elif hasattr(layer, 'out_features'): 
            layer.weight.data[:]*=0
            #layer.weight.data.uniform_(-0.1, 0.1)
            if layer.bias is not None:
                layer.bias.data.uniform_(-1e-3,1e-3)
        else:
            warnings.warn('Unhandled layer type, not resetting parameters')
    
    @staticmethod
    def get_out_channels(layer):
        '''
        Wrapper for returning number of output channels in a LIFLayer
        '''
        if hasattr(layer, 'out_features'):
            return layer.out_features
        elif hasattr(layer, 'out_channels'): 
            return layer.out_channels
        elif hasattr(layer, 'get_out_channels'): 
            return layer.get_out_channels()
        else: 
            raise Exception('Unhandled base layer type')
    
    @staticmethod
    def get_out_shape(layer, input_shape):
        if hasattr(layer, 'out_channels'):
            return get_output_shape(input_shape, 
                                    kernel_size=layer.kernel_size,
                                    stride = layer.stride,
                                    padding = layer.padding,
                                    dilation = layer.dilation)
        elif hasattr(layer, 'out_features'): 
            return []
        elif hasattr(layer, 'get_out_shape'): 
            return layer.get_out_shape()
        else: 
            raise Exception('Unhandled base layer type')

    def init_state(self, Sin_t):
        dtype = Sin_t.dtype
        device = self.base_layer.weight.device
        input_shape = list(Sin_t.shape)
        out_ch = self.get_out_channels(self.base_layer)
        out_shape = self.get_out_shape(self.base_layer, input_shape)
        self.state = self.NeuronState(P=torch.zeros(input_shape).type(dtype).to(device),
                                      Q=torch.zeros(input_shape).type(dtype).to(device),
                                      R=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      S=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device))

    def init_parameters(self, Sin_t=0):
        self.reset_parameters(self.base_layer)
        
    def forward(self, Sin_t, U_connect=0):
        if self.state is None:
            self.init_state(Sin_t)

        state = self.state
        Q = self.beta * state.Q + (self.tau_s) * Sin_t
        P = self.alpha * state.P + (self.tau_m) * state.Q  
        R = self.alpharp * state.R - state.S * self.wrp
        U = self.base_layer(P) + U_connect + R
        S = self.sg_function(U)
        self.state = self.NeuronState(P=P, Q=Q, R=R, S=S)
        if self.do_detach: 
            state_detach(self.state)
        return S, U

    def get_output_shape(self, input_shape):
        layer = self.base_layer
        if hasattr(layer, 'out_channels'):
            im_height = input_shape[-2]
            im_width = input_shape[-1]
            height = int((im_height + 2 * layer.padding[0] - layer.dilation[0] *
                          (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1)
            weight = int((im_width + 2 * layer.padding[1] - layer.dilation[1] *
                          (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1)
            return [height, weight]
        else:
            return layer.out_features
    
    def get_device(self):
        return self.base_layer.weight.device

class SpikingAgent(nn.Module):
    """
    Agent class, basic building block for a community of agents. Based on an RSNN cell
    Params : 
        n_in, n_hidden, n_out : input, hidden and output dimensions
        tag : string describing the agent, used to store connections in a dictionnary
        use_readout : whether to use a readout layer
    """
    def __init__(self, n_in, n_hidden, n_out, tag, method='rtrl', layer_type = LIFRecLayer):
        super().__init__()

        self.dims = [n_in, *n_hidden, n_out]
        self.tag = tag
        
        self.LIF_layers = nn.ModuleList()
        self.readout_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        
        self.method = method
        for d1, d2 in zip(self.dims[:-2], self.dims[1:-1]) : 
            base_layer = nn.Linear(d1, d2)
            rec_layer = nn.Linear(d2, d2, bias=False)
            layer = layer_type(base_layer,
                                rec_layer,
                                do_detach= True if method == 'rtrl' else False)
            
            layer.init_parameters()
            self.LIF_layers.append(layer)
            
            readout = nn.Linear(d2, n_out)
            for param in readout.parameters():
                param.requires_grad = False
            self.reset_lc_parameters(readout, .5)
            self.readout_layers.append(readout)
                    
                    
                    
    def forward(self, S_in, U_connect=0) :
        """
        forward function of the Agent
        Params : 
            S_in : input data
            S_connect : sparse connections from other agents
        """
        s_out = []
        r_out = []
        u_out = []
        
        S_in = S_in.view(S_in.size(0), -1)
        input = (S_in, U_connect)
        for i, (lif, ro) in enumerate(zip(self.LIF_layers, self.readout_layers)) : 
            
            s, u = lif(*input)
            
            if i == len(self.LIF_layers) : 
                s_ = sigmoid(u)
            else : 
                s_ = lif.sg_function(u)
                
            r_ = ro(s_.reshape(s_.size(0), -1))
            
            s_out.append(s_) 
            r_out.append(r_)
            u_out.append(u)
            
            input = [s_.detach() if lif.do_detach else s_]
            
        return r_out, (s_out, u_out)  
    
    
    def init(self, data_batch, burnin) : 
        
        for l in self.LIF_layers : 
            l.state = None
            l.rec_state = None
        with torch.no_grad():
            for i in range(burnin) : 
                self.forward(data_batch[i, ...])
                
                
    def reset_lc_parameters(self, layer, lc_ampl):
        stdv = lc_ampl / np.sqrt(layer.weight.size(1))
        layer.weight.data.uniform_(-stdv, stdv)
        self.reset_lc_bias_parameters(layer,lc_ampl)
        
    def reset_lc_bias_parameters(self, layer, lc_ampl):
        stdv = lc_ampl / np.sqrt(layer.weight.size(1))
        if layer.bias is not None:
            layer.bias.data.uniform_(-stdv, stdv)
        
class SpikingCommunity(nn.Module) : 
    """
    Community, composed of sparsely connected agents. 
    Params : 
        agents : list of agents composing the community
        sparse_connections : (n_agents x n_agents) matrix specifying the sparsity of the connections in-between agents
    """
    def __init__(self, agents, sparse_connections):
        super().__init__()
        self.agents = nn.ModuleList()
       
        for ag in agents : 
            self.agents.append(ag)
        self.n_agents = len(agents)
        self.sparse_connections = sparse_connections
        self.method = self.agents[0].method
        self.init_connections()
    
    @property
    def agents_states(self) : 
        return [ag.LIF_layers[-1].state for ag in self.agents]
        
        
    # Initializes connections in_between agents with the given sparsity matrix
    def init_connections(self) : 
        
        self.tags = np.empty((self.n_agents, self.n_agents), dtype=object)
        self.connected = np.zeros((self.n_agents, self.n_agents))
        self.connections = nn.ModuleDict()
        
        for i, ag1 in enumerate(self.agents) : 
            i = int(ag1.tag)
            for j, ag2 in enumerate(self.agents) : 
                j = int(ag2.tag)
                if i==j: continue
                else : 
                    p_con = self.sparse_connections[i, j]
                    if p_con >0 : 
                        #State-to-State connections : 
                        n_in = ag1.dims[-2]
                        n_out = ag2.dims[1]
                        
                        #Output-to-Input connections:
                        #n_in = ag1.dims[-1]
                        #n_out = ag2.dims[0]
                        
                        dims = [n_in, n_out]
                        sparsity_list = [p_con]
                        connection_layer = deepR.Sparse_Connect(dims, sparsity_list)
                        #connection_layer = nn.Linear(n_in, n_out, bias=False)
                        connection = LIFLayerConnect(connection_layer, None,  True if self.method == 'rtrl' else False)
                        
                        connection.init_parameters()
                        
                        self.tags[i, j] = ag1.tag+ag2.tag
                        self.connections[self.tags[i, j]] = connection
                        self.connected[i, j] = 1

    def init(self, data_batch, burnin):
        '''
        Necessary to reset the state of the network whenever a new batch is presented
        '''
        for ag in self.agents:
            for l in ag.LIF_layers : 
                l.state = None
                l.rec_state = None

        for c in self.connections.values() : 
            c.state = None         

        with torch.no_grad():
            for i in range(burnin) : 
                self.forward(data_batch[i, ...])
                ''
                
    #Forward function of the community
    """
    def forward(self, x):
        outputs = []
        spikes = []
        mems = []
        
        #Split_data checks if the data is provided on a per-agent manner or in a one-to-all manner. 
        #Split_data can be a double list of len n_timesteps x n_agents or a tensor with second dimension n_agents
        split_data = ((type(x) is torch.Tensor and len(x.shape)>3) or type(x) is list)            
        
        for t, x_t in enumerate(x) : 
            if split_data:
                assert len(x_t) == len(self.agents), 'If data is provided per agent, make sure second dimension matches number of agents'
                
            output = [None for ag in self.agents]
            spike = [None for ag in self.agents]
            mem = [None for ag in self.agents]
            
            for ag1 in self.agents :
                i = int(ag1.tag)
                if split_data : 
                    inputs = x_t[i].clone()
                else : 
                    inputs = x_t.clone()
                    
                #Receive sparse connections from other agents 
                if t > 0 : 
                    inputs_connect = 0
                    for ag2 in self.agents :
                        j = int(ag2.tag)
                        if self.connected[j, i]==1: 
                            #State-to-state connections j->i
                            sparse_out = self.connections[ag2.tag+ag1.tag](spikes[t-1][j])
                            inputs_connect += sparse_out[0]

                            #Output-to-Input connections j->i
                            #sparse_out = self.connections[self.tags[j, i]](outputs[n-1][j])
                            #inputs += sparse_out[0]

                    out, (s, u) = ag1(inputs, inputs_connect)
                    
                else : 
                    out, (s, u) = ag1(inputs)
                    
                output[i] = out
                spike[i] = s
                mem[i] = u
                
            #Store states and outputs of agent
            spikes.append(spike)
            mems.append(mem)
            output = torch.stack(output)            
            outputs.append(output)
            
        outputs = torch.stack(outputs)
        
        return outputs, (spikes, mems)
    """
    def forward(self, x_t):

        #Split_data checks if the data is provided on a per-agent manner or in a one-to-all manner. 
        #Split_data can be a double list of len n_timesteps x n_agents or a tensor with second dimension n_agents
        split_data = ((type(x_t) is torch.Tensor and len(x_t.shape)>2) or type(x_t) is list)            
        
        if split_data:
            assert len(x_t) == len(self.agents), 'If data is provided per agent, make sure second dimension matches number of agents'

        output = [None for ag in self.agents]
        spike = [None for ag in self.agents]
        mem = [None for ag in self.agents]
        for i, ag1 in enumerate(self.agents) :
            if split_data : 
                inputs = x_t[i]
            else : 
                inputs = x_t
                
            #Receive sparse connections from other agents 

            inputs_connect = 0
            for j, ag2 in enumerate(self.agents) :
                con_tag = ag2.tag+ag1.tag
                if self.connected[j, i]==1 and self.agents_states[j] is not None : 
                    #State-to-state connections j->i
                    sparse_out = self.connections[con_tag](self.agents_states[j].S)
                    inputs_connect += sparse_out[-1]
                    ''
                    #Output-to-Input connections j->i
                    #sparse_out = self.connections[self.tags[j, i]](outputs[n-1][j])
                    #inputs += sparse_out[0]

            out, (s, u) = ag1(inputs, inputs_connect)

            output[i] = torch.stack(out)
            spike[i] = torch.stack(s)
            mem[i] = torch.stack(u)

        #Store states and outputs of agent
        spike = torch.stack(spike)
        mem = torch.stack(mem) 
        output = torch.stack(output)            
                    
        return output.transpose(0, 1), (spike, mem)
    
    @property
    def nb_connections(self) : 
        """
        Returns dictionnary of number of active connections of community
        """
        nb_connections = {}
        for tag, c in zip(self.connections.keys(), self.connections.values()) : 
            if type(c) is LIFLayerConnect : c = c.base_layer
            
            nb_connected = (c.thetas[0]>0).int().sum()
            nb_connections[tag] = nb_connected

        return nb_connections
    
    def get_trainable_named_parameters(self, layer=None):
        if layer is None:
            params = dict()
            for k,p in self.named_parameters():
                if p.requires_grad:
                    params[k]=p

            return params
        else:
            return self.LIF_layers[layer].named_parameters()
        
        
    def get_trainable_parameters(self, layer=None):
        if layer is None:
            return chain(*[l.parameters() for l in self.LIF_layers])
        else:
            return self.LIF_layers[layer].parameters()

def apply_grad(theta, params):
    """
    Update parameter's gradient and compute random-walk step as well as regularizing step.
    """
    device = theta.device
    lr = params['lr']
    l1 = params['l1']
    is_con = (theta>0)
    
    if theta.grad is not None :
        gdnoise = params['gdnoise']
        noise_update = (torch.randn_like(theta)*gdnoise).to(device)
        theta.data += is_con*noise_update*lr
        theta.data -= is_con*l1*lr
        theta.grad *= is_con
        
    else : 
        print('None Grad')
               

        

