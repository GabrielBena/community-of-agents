import torch
import numpy as np
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

DeepR_path = '/home/gb21/Code/re-creations/DEEP-R/'
import sys, os
sys.path.append(DeepR_path)
import deepR_torch as deepR

sys.path.append('../')

from common.models import Agent, get_output_shape, ConvCommunity

from dual_utils import *


relu = nn.ReLU()
tanh = nn.Tanh()
sigmoid = nn.Sigmoid()
#------ Surrogate Gradient------


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
        out[input >= thr] = 1.0
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
        
super_spike  = SurrGradSpike.apply

#------Mask and Scores ------
class GetSubnet_2(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        flat_out = out.flatten()
        top_k = (flat_out.numel()*k).int().item()
        min_k = flat_out.numel() - top_k
        
        top_values, idx_1 = torch.topk(flat_out, top_k)
        _, idx_0 = torch.topk(flat_out, min_k, largest=False)
        # flat_out and out access the same memory.

        flat_out[idx_1] = 1
        flat_out[idx_0] = 0

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10) -> torch.Tensor:
    uniform = logits.new_empty([2]+list(logits.shape)).uniform_(0, 1)

    noise = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()
    res = torch.sigmoid((logits + noise) / tau)

    if hard:
        res = ((res > 0.5).type_as(res) - res).detach() + res

    return res

def sigmoid(logits: torch.Tensor, mode: str = "simple", tau: float = 1, eps: float = 1e-10):
    if mode=="simple":
        return torch.sigmoid(logits)
    elif mode in ["soft", "hard"]:
        return gumbel_sigmoid(logits, tau, hard=mode=="hard", eps=eps)
    else:
        assert False, "Invalid sigmoid mode: %s" % mode

def getSubnet_sig(scores, train=False):
    out = scores.clone()

    if train : 
        return gumbel_sigmoid(out, hard=True)
    else : 
        #return (out >= 0.).float()
        return torch.maximum(out, torch.zeros_like(out).to(out.device))


#------ Community Models------

class SupSupCommunity(ConvCommunity) : 
    """
    Community of mind, composed of sparsely connected agents, with a global CNN input layer. 
    A dictionnary of masks over agents is contructed to face multiple tasks
    Params : 
        agents : list of agents composing the community
        sparse_connections : (n_agents x n_agents) matrix specifying the sparsity of the connections in-between agents
        sparse_out = (n_out, p_out) : if not None, use a sparse connection of n_out neurons and sparsity p_out 
                                as a global readout connected to all agents
                                
        sparse_in = (n_in, p_in) : if not None, use sparse connections of n_in neurons and sparsity p_in
                                   for every agent to replace original weights
        
        use_deciding_agent : Wheter to use a separate agent to take decsions, connected to all other in a straightforward manner
    """
    def __init__(self,
                 input_shape,
                 Nhid=[4],
                 kernel_size=[7],
                 stride=[1],
                 pool_size=[2],
                 num_conv_layers=1,
                 decouple=False,
                 p_connect=0.1,
                 n_hid=20,
                 n_out=10,
                 n_out_sup=0,
                 distance_based_sparsity=False,
                 cell_type=nn.RNN,
                 use_deciding_agent=False,
                 use_sparse_readout=False,
                 mask_samples=1,
                 temporal_masks=False,
                 train_deterministic=False
                ):
        
        super().__init__(input_shape,
                 Nhid,
                 kernel_size,
                 stride,
                 pool_size,
                 num_conv_layers,
                 decouple,
                 p_connect,
                 n_hid,
                 n_out+n_out_sup,
                 distance_based_sparsity,
                 cell_type,
                 use_deciding_agent,
                 use_sparse_readout)
        
        for p in self.agents.parameters() : 
            p.requires_grad = True
            
        self.n_out = (n_out, n_out_sup)
        self.connection_scores = nn.ParameterDict()

        self.output_scores = nn.ParameterDict()

        self.task_ids = []
        
        self.newly_learnt = []
        self.mask_samples = mask_samples
        self.temporal_masks = temporal_masks

        self.is_train=True
        self.train_deterministic = train_deterministic
    
    @property 
    def all_scores(self) : 
        return list(self.connection_scores.values()) + list(self.output_scores.values())

    @property
    def num_tasks(self) : 
        return len(self.connection_scores.keys())

    @property
    def task_ints(self) : 
        return np.array(self.task_ids, dtype=int)

    def all_connection_masks(self, n_samples=0) : 
        if len(self.task_ids)>0 :
            return torch.stack([self.get_connection_mask(t, n_samples)[0]for t in self.task_ids])
        else : 
            return None

    def all_output_masks(self, n_samples=0) : 
        if len(self.task_ids)>0 :
            return torch.stack([self.get_output_mask(t, n_samples)[0]for t in self.task_ids])
        else : 
            return None

    def init_connection_score(self, task_id, n_steps=None) : 
        
        if self.temporal_masks : 
            connection_score = nn.Parameter(torch.Tensor(torch.Size([n_steps-1]+list(self.connected.shape))), requires_grad=True)
        else : 
            connection_score = nn.Parameter(torch.Tensor(torch.Size(self.connected.shape)), requires_grad=True)
        
        #nn.init.kaiming_uniform_(connection_score, a=np.sqrt(5))    

        sig_inv = lambda p : np.log((p/(1-p)))

        #init_ = sig_inv(1/self.n_agents)
        #init_ = sig_inv(0.25)
        init_ = 0.

        nn.init.kaiming_normal_(connection_score, a=0.)
        #nn.init.constant_(connection_score, init_)

        self.connection_scores[task_id] = connection_score
        self.task_ids.append(task_id)
        self.newly_learnt.append(task_id)        
        if not hasattr(self, 'self_connection_mask') : 
            self.self_connection_mask = torch.eye(connection_score.shape[-1]).bool()       
            if self.temporal_masks : 
                self.self_connection_mask = (self.self_connection_mask.unsqueeze(0).expand([n_steps-1] + list(self.self_connection_mask.shape))).bool()
        return connection_score
    
    def init_out_score(self, task_id) : 
        
        output_score = nn.Parameter(torch.Tensor(torch.Size([self.n_agents])), requires_grad=True)

        sig_inv = lambda p : (p/(1-p)).log()
        #init_ = sig_inv(torch.tensor(1/np.sqrt(self.n_agents)))
        init_ = 0.
        #nn.init.constant_(output_score, init_)
        nn.init.normal_(output_score, std=1e-2)

        self.output_scores[task_id] = output_score
        return output_score
        
    def train(self, training=True) : 
        self.is_train = training
        super().train(training)

    def get_output_mask(self, task_id, n_samples=1) : 
        output_score = self.output_scores[task_id]  
        if n_samples > 0:
            mask = output_score.unsqueeze(0).expand(n_samples, *output_score.shape) if n_samples > 1 else output_score
            return gumbel_sigmoid(mask, hard=True), output_score
        else:
            #return (output_score >= 0).float(), output_score  
            return super_spike(output_score), output_score       

    def get_connection_mask(self, task_id, n_samples=1) : 
        connection_score = self.connection_scores[task_id]  
        mask = connection_score.clone()
        #mask[self.self_connection_mask] -= 100
        if n_samples > 0 :
            if n_samples > 1 : 
                mask = mask.unsqueeze(0).expand(n_samples, *mask.shape)
            return gumbel_sigmoid(mask, hard=True), connection_score
        else:
            #return (mask >= 0).float(), connection_score
            return super_spike(mask), connection_score

    def forward(self, x, task_id=None) : 
        
        if task_id is None :
            if len(self.task_ids) == 0 : 
                task_id = '0'
            else : 
                task_id, *_ = oneshot_task_inference(self, x)

        n_samples = self.mask_samples if self.is_train else 0

        if task_id == '-1' : 
            #alpha_mask = self.alphas>0

            self.output_masks = (torch.stack([self.get_output_mask(t, n_samples)[0]*a*(a>0.)
                        for t, a in zip(self.task_ids, self.alphas)])).sum(0)

            self.connection_masks = (torch.stack([self.get_connection_mask(t, n_samples)[0]*a*(a>0.)
                            for t, a in zip(self.task_ids, self.alphas)])).sum(0)

        elif task_id != 'skip': 
            n_steps = x.shape[0]
            batch_size = x.shape[1]
            if task_id not in self.task_ids : 
                self.init_connection_score(task_id, n_steps)
                self.init_out_score(task_id)

            self.output_masks, output_score = self.get_output_mask(task_id, n_samples)
            self.connection_masks, connection_score = self.get_connection_mask(task_id, n_samples)
            
            n1 = batch_size//2 if self.train_deterministic else batch_size
            n2 = batch_size - n1

            if n_samples > 1 : 
                batch_mask = lambda mask : torch.stack([mask[n%n_samples] for n in range(n1)], -1).to(mask.device)
                self.connection_masks, self.output_masks = batch_mask(self.connection_masks), batch_mask(self.output_masks)

            if self.train_deterministic and n_samples >0: 
                deterministic_masks = self.get_output_mask(task_id, n_samples=0)[0], self.get_connection_mask(task_id, n_samples=0)[0]
                batch_mask = lambda mask : torch.stack([mask for n in range(n2)]).transpose(0, -1).to(mask.device)
                self.connection_masks = torch.cat([self.connection_masks, batch_mask(deterministic_masks[1])], -1)
                self.output_masks = torch.cat([self.output_masks, batch_mask(deterministic_masks[0])], -1)

        return super().forward(x, supsup=(task_id != 'skip'), temporal_masks=self.temporal_masks)
   
# ------ Attention Based Models ------

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model_read, d_model_write, d_model_out, d_k, d_v, dropout=0.1, ind_queries=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.ind_queries=ind_queries

        self.w_qs = nn.Linear(d_model_read, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model_write, n_head * d_v, bias=False)
        self.w_ks = nn.Linear(d_model_write, n_head * d_k, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model_out, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model_out, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        #print(q.shape, k.shape, v.shape)

        #residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        if not self.ind_queries : q = self.w_qs(q)
        q = q.view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        #q += residual

        q = self.layer_norm(q)

        return q, attn
    
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class GlobalWorkspaceCommunity(nn.Module) : 

    """
    Community of mind, composed of sparsely connected agents, with a global CNN input layer. 
    Communication between agents is ensured via a shared global workspace (Goyal 2020)
    Params : 
        agents : list of agents composing the community
        sparse_connections : (n_agents x n_agents) matrix specifying the sparsity of the connections in-between agents
        sparse_out = (n_out, p_out) : if not None, use a sparse connection of n_out neurons and sparsity p_out 
                                as a global readout connected to all agents
                                
        sparse_in = (n_in, p_in) : if not None, use sparse connections of n_in neurons and sparsity p_in
                                   for every agent to replace original weights
        conv_agents : list of ConvAgents to create first layer
        
        use_deciding_agent : Wheter to use a separate agent to take decsions, connected to all other in a straightforward manner
    """
    def __init__(self,
                 input_shape,
                 Nhid=[4],
                 kernel_size=[7],
                 stride=[1],
                 pool_size=[2],
                 num_conv_layers=1,
                 decouple=False,
                 p_connect=0.1,
                 n_hid = 20,
                 n_out = 10,
                 distance_based_sparsity=False,
                 cell_type=nn.RNN,
                 use_deciding_agent=False,
                 use_sparse_readout=False,
                 use_attention=False):
        
        
        # If only one value provided, then it is duplicated for each layer
        if len(kernel_size) == 1:   kernel_size = kernel_size * num_conv_layers
        if stride is None: stride=[1]
        if len(stride) == 1:        stride = stride * num_conv_layers
        if pool_size is None: pool_size = [1]
        if len(pool_size) == 1: pool_size = pool_size * num_conv_layers
        if Nhid is None:          self.Nhid = Nhid = []
                
        super().__init__()
        self.is_community = True
        
        self.input_shape = input_shape
        
        self.decouple = decouple
        if not decouple : 
            Nhid = [input_shape[0]] + Nhid
        else : 
            Nhid = [1] + Nhid
            
        feature_height = self.input_shape[1]
        feature_width = self.input_shape[2]
        padding = (np.array(kernel_size) - 1) // 2
        
        
        self.cnns = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        for i in range(num_conv_layers) : 
            feature_height, feature_width = get_output_shape(
                [feature_height, feature_width], 
                kernel_size = kernel_size[i],
                stride = stride[i],
                padding = padding[i],
                dilation = 1)
            
            feature_height //= pool_size[i]
            feature_width //= pool_size[i]
            
            if decouple : 
                cnn = nn.ModuleList()
                for _ in range(input_shape[0]) : 
                    cnn.append(nn.Conv2d(Nhid[i], Nhid[i+1], kernel_size[i], stride[i], padding[i]))
                
            else : 
                cnn = nn.Conv2d(Nhid[i], Nhid[i+1], kernel_size[i], stride[i], padding[i])
            self.cnns.append(cnn)
            
            pool = nn.MaxPool2d(kernel_size=pool_size[i])
            self.pool_layers.append(pool)
            
        n_in = feature_height*feature_width
        n_agents = Nhid[-1]
        if decouple : 
            n_agents *= input_shape[0]
            
        self.dims = [n_in, n_hid, n_out]

        self.competing_agents = True
        self.use_gating = True

        if use_attention : 
            self.attention_layers = nn.ModuleDict()

            self.n_mem = n_agents//2
            self.mem_size = n_in

            d_k, d_v = 64, 64

            if self.competing_agents : 
                    
                self.n_head = 6

                self.attention_layers['agent_attention'] = MultiHeadAttention(self.n_head, n_hid, n_in, self.mem_size, d_k, d_v, ind_queries=True)
                query_params = (n_hid, self.n_head, d_k)           

                self.attention_layers['workspace_writing'] = MultiHeadAttention(1, self.mem_size, self.mem_size, self.mem_size, d_k, d_v, ind_queries=False)

            else : 
                self.n_head = 4
                query_params = None         

                self.attention_layers['workspace_writing']  = MultiHeadAttention(1, self.mem_size, n_hid, self.mem_size, d_k, d_v, ind_queries=False)

            self.attention_layers['workspace_broadcast']  = MultiHeadAttention(4, n_hid, self.mem_size, n_hid, d_k, d_v, ind_queries=False)

            if self.use_gating : 
                self.gating_weights = nn.ParameterDict()
                self.gating_weights['input'] = nn.Parameter(torch.rand(n_in, self.mem_size))
                self.gating_weights['I'] = nn.Parameter(torch.rand(self.mem_size, self.mem_size))
                self.gating_weights['F'] = nn.Parameter(torch.rand(self.mem_size, self.mem_size))
                
        else : 
            query_params = None
        
        self.individual_readouts = not(use_sparse_readout or use_deciding_agent)
        agents = [Agent(n_in, n_hid, n_out, str(n), cell_type=cell_type,
                     use_readout=self.individual_readouts, query_params=query_params) for n in range(n_agents)]        
        self.agents = nn.ModuleList()
       
        for ag in agents : 
            self.agents.append(ag)
        self.n_agents = len(agents)
        
        self.p_connect = p_connect
        self.sparse_connections = (np.ones((n_agents, n_agents)) - np.eye(n_agents))*p_connect  
        if distance_based_sparsity : 
            self.sparse_connections[n_agents//2:, :n_agents//2] *= 0.5
            self.sparse_connections[:n_agents//2, n_agents//2:] *= 0.5
            
        self.use_deciding_agent = use_deciding_agent
        self.use_sparse_readout = use_sparse_readout
        self.use_attention = use_attention
        
        self.init_connections()
        
    def init_sparse_readout(self, n_out=None, p_out=None) : 
        if n_out is None : 
            n_out = self.dims[-1]
        if p_out is None : 
            p_out = self.p_connect
        
        n_in = np.sum(np.array([ag.dims[-2] for ag in self.agents]))
        if self.decouple : 
            readout = nn.ModuleList()
            n_in //= 2
            for i in range(2) : 
                decoupled_readout = nn.Linear(n_in, n_out)
                readout.append(decoupled_readout)
            self.connections['readout'] = readout
        else : 
            readout = deepR.Sparse_Connect([n_in, n_out], [p_out])
            #readout = nn.Linear(n_in, n_out)
            self.connections['readout'] = readout
        
    def init_deciding_agent(self, n_in, n_hid, n_out) :
        cell_type = self.agents[0].cell.__class__
        decider = Agent(n_in, n_hid, n_out, 'decider', True, cell_type=cell_type)
        self.decider = decider
        
    # Initializes connections in_between agents with the given sparsity matrix
    def init_connections(self, sparse_connections=None) :   
        self.connections = nn.ModuleDict()
                      
        if self.use_sparse_readout : 
            if self.use_deciding_agent : 
                n_hid = self.dims[-2]
                n_hid *= 2
                self.init_deciding_agent(n_hid, n_hid, self.dims[-1])
                self.init_sparse_readout(n_hid, 0.25)
            else : 
                self.init_sparse_readout(self.dims[-1], 0.25)    
        else :
            if self.use_deciding_agent : 
                n_in = np.sum(np.array([ag.dims[-2] for ag in self.agents]))
                n_hid = self.dims[-2]
                n_hid *= 2
                self.init_deciding_agent(n_in, n_hid, self.dims[-1])
                        
    
    def gated_writing(self, X, M_t, M_attention) : 

        X_sum = torch.mean(torch.stack([relu(torch.matmul(x, self.gating_weights['input'])) for x in X]), dim=0)
        K = X_sum + tanh(M_t)
        I = sigmoid(torch.matmul(K, self.gating_weights['I']))
        F = sigmoid(torch.matmul(K, self.gating_weights['F']))
        M_t = I*tanh(M_attention) + F*M_t

        return M_t

    def process_input(self, x_t) : 
        for l, (cnn, pool) in enumerate(zip(self.cnns, self.pool_layers)) : 
            if self.decouple : 
                if l == 0 : x_t = [x_t[:, i, ...].unsqueeze(1) for i in range(2)]
                x_t = [cnn[i](x_t_d) for i, x_t_d in enumerate(x_t)]
                x_t = [pool(x_t_d) for x_t_d in x_t]
                
            else : 
                x_t = cnn(x_t)
                x_t = pool(x_t)
        if self.decouple : x_t = torch.cat(x_t, axis=1)
        
        #print(x_t.shape)
        x_t = x_t.flatten(start_dim=-2).transpose(0, 1)
        return x_t

    #Forward function of the community
    def forward(self, x):

        if self.use_attention : 
            self.workspace = torch.zeros((x.shape[1], self.n_mem, self.mem_size)).to(self.device)
        outputs = []
        states = []
        states_decider = []
        #Split_data checks if the data is provided on a per-agent manner or in a one-to-all manner. 
        #Split_data can be a double list of len n_timesteps x n_agents or a tensor with second dimension n_agents        
        for n, x_t in enumerate(x) : 

            x_t = self.process_input(x_t)
                
            if self.use_deciding_agent : 
                state_decider = None
                #state_decider = torch.zeros((1, x_t.shape[1], self.decider.dims[-2])).to(self.decider.device)
                #print(state_decider.shape)
            
            if n == 0 : 
                state = [None for ag in self.agents]
                output = [None for ag in self.agents]
                for i, ag1 in enumerate(self.agents) :
                    inputs = x_t[i]#.clone()

                    out, h = ag1(inputs)
                        
                    output[i] = out
                    state[i] = h
                
                #Store states and outputs of agent
                state, output = torch.cat(state), torch.cat(output)
                final_state = state

            else :
                if self.use_attention : 
                    if self.competing_agents : 

                        ag_queries = torch.stack([ag.w_qs(s) for ag, s in zip(self.agents, state)])
                        ag_queries = ag_queries.transpose(0, 1)
                        attention_modulated_inputs, scores = self.attention_layers['agent_attention'](ag_queries, x_t.transpose(0, 1), x_t.transpose(0, 1))
                        
                        _, indices = torch.topk(torch.diagonal(scores.mean(axis=1), dim1=1, dim2=2), self.n_mem, dim=-1)

                        expanded_indices = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), attention_modulated_inputs.size(-1))
                        expanded_indices_state = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), state.size(-1)).transpose(0, 1)
                        expanded_indices_out = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), output.size(-1)).transpose(0, 1)

                        selected_values = torch.gather(attention_modulated_inputs, 1, expanded_indices)
                        
                        R = torch.cat([self.workspace, selected_values], dim=1)

                        attention_modulated_inputs = attention_modulated_inputs.transpose(0, 1)
                        new_state, new_output = [None for ag in self.agents], [None for ag in self.agents]
                        for i, ag1 in enumerate(self.agents) :
                            inputs = attention_modulated_inputs[i]
                            out, h = ag1(inputs, state[i].unsqueeze(0))
                            new_state[i], new_output[i] = h, out

                        new_state, new_output = torch.cat(new_state), torch.cat(new_output)
                        state, output = state.clone(), output.clone()

                        new_state = torch.gather(new_state, 1, expanded_indices_state)
                        new_output = torch.gather(new_output, 1, expanded_indices_out)

                        state.scatter_(1, expanded_indices_state, new_state)

                        if self.individual_readouts : 
                            output.scatter_(1, expanded_indices_out, new_output)
                        else : 
                            output = state

                    else : 
                        R = state.transpose(0, 1)

                    
                    workspace_attention, workspace_scores = self.attention_layers['workspace_writing'](self.workspace, R, R)

                    if self.use_gating : 
                        self.workspace, workspace_attention = self.workspace.transpose(0, 1), workspace_attention.transpose(0, 1)
                        self.workspace = torch.stack([self.gated_writing(x_t, m, m_attention) for (m, m_attention) in zip(self.workspace, workspace_attention)], axis=1)
                    else : 
                        self.workspace = workspace_attention
                        
                    self.workspace = self.workspace.transpose(0, 1)

                    broadcast_states = []
                    for s in state : 
                        broadcast_state = []
                        for m in self.workspace : 
                            new_s, _ = self.attention_layers['workspace_broadcast'](s.unsqueeze(1), m.unsqueeze(1), m.unsqueeze(1))
                            broadcast_state.append(new_s.transpose(0, 1))
                        broadcast_state = torch.cat(broadcast_state)
                        broadcast_states.append(broadcast_state)

                    broadcast_states = torch.stack(broadcast_states)
                    state = state + broadcast_states.sum(axis=1)
                    
                    self.workspace = self.workspace.transpose(0, 1)
                else : 
                    state = state
            
            states.append(state)
            
            if 'readout' in self.connections.keys() : 
                if self.decouple : 
                    n_decouple = self.n_agents//2
                    output = [output[:n_decouple].transpose(0, 1).flatten(start_dim=1),
                                         output[n_decouple:].transpose(0, 1).flatten(start_dim=1)]
                    
                    output = [r(o) for r, o in zip(self.connections['readout'], output)]
                    #print([o.shape for o in output])
                    output = torch.stack(output)
                        
                else : 
                    output = output.transpose(0, 1).flatten(start_dim=1)
                    output = self.connections['readout'](output)
                    if self.use_deciding_agent : 
                        output, state_decider = self.decider(output, state_decider)
                        output = output[0]
                        states_decider.append(state_decider)                
            else : 
                if self.use_deciding_agent : 
                    output = output.transpose(0, 1).flatten(start_dim=1)
                    output, state_decider = self.decider(output, state_decider)
                    output = output[0]
                    states_decider.append(state_decider)
            
            outputs.append(output)
        outputs = torch.stack(outputs)
        
        return outputs, (states, 0)
    
    @property
    def device(self) : 
        return self.agents[0].device
    
    @property
    def nb_connections(self) : 
        """
        Returns dictionnary of number of active connections of community
        """
        nb_connections = {}
        for tag, c in zip(self.connections.keys(), self.connections.values()) : 
            if type(c) is deepR.Sparse_Connect : 
                nb_connected = (c.thetas[0]>0).int().sum()
                nb_connections[tag] = nb_connected

        return nb_connections

