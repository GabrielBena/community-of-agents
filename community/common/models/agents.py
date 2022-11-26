import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from copy import deepcopy
import numpy as np
import scipy


class LeakyRNN(nn.RNNCell):
    def __init__(
        self,
        input_size,
        hidden_size,
        l=1,
        alpha=1,
        bias=True,
        nonlinearity="relu",
        batch_first=False,
    ):
        super(LeakyRNN, self).__init__(
            input_size, hidden_size, bias=bias, nonlinearity=nonlinearity
        )
        self.alpha = alpha

    def forward(self, x_in, h=None):
        states = []
        if len(x_in.shape) < 3:
            x_in = x_in.unsqueeze(0)

        for x_t in x_in:

            if h is None:
                output = super(LeakyRNN, self).forward(x_t)
                h = self.alpha * output
            else:
                output = super(LeakyRNN, self).forward(x_t, h)
                h = (1 - self.alpha * h) + self.alpha * output

            states.append(h)

        states = torch.stack(states)
        return states, h


cell_types_dict = {str(t): t for t in [nn.RNN, nn.LSTM, nn.GRU]}


class Agent(nn.Module):
    """
    Agent class, basic building block for a community of agents. Based on an RNN/LSTM cell
    Params :
        n_in, n_hidden, n_out : input, hidden and output dimensions
        tag : string describing the agent, used to store connections in a dictionnary
        use_readout : whether to use a readout layer
        use bottleneck : wheter to use a bottleneck, for bottleneck retraining metric
        train_in_out : whether to train input and readout layers respectively
        cell_type : architecture to use as reccurent cell
    """

    def __init__(
        self,
        n_in,
        n_hidden,
        n_layers,
        n_out,
        tag,
        n_readouts=1,
        train_in_out=(True, True),
        cell_type=nn.RNN,
        use_bottleneck=False,
        ag_dropout=0.0,
        density=1.0,
    ):

        super().__init__()

        self.dims = [n_in, n_hidden, n_out]

        self.tag = str(tag) if type(tag) is not str else tag

        if type(cell_type) is tuple:
            cell_type = cell_type[0]
        if type(cell_type) is str:
            cell_type = cell_types_dict[cell_type]

        self.cell = cell_type(n_in, n_hidden, n_layers, batch_first=False)
        self.cell_type = cell_type
        for n, p in self.cell.named_parameters():
            if "weight" in n:
                init.xavier_normal_(p, init.calculate_gain("tanh", p))

        self.use_bottleneck = use_bottleneck
        self.dropout = nn.Dropout(ag_dropout) if ag_dropout > 0 else None

        if self.use_bottleneck:
            if n_out == 100:
                n_bot = 10
            else:
                n_bot = 5
            bottleneck = [nn.Linear(n_hidden, n_bot)]
            readout = [nn.Linear(n_bot, n_out)]
        else:
            readout = [nn.Linear(n_hidden, n_out)]

        readout[0].weight.requires_grad = train_in_out[1]
        readout[0].bias.requires_grad = train_in_out[1]

        self.use_readout = n_readouts is not None
        if self.use_readout:

            self.multi_readout = n_readouts > 1
            if self.multi_readout:
                readout.extend([deepcopy(readout[0]) for _ in range(n_readouts - 1)])

            self.readout = nn.ModuleList(readout)

            self.init_readout_weights(self.readout)

        self.cell_params("weight_ih_l0").requires_grad = train_in_out[0]
        self.cell_params("bias_ih_l0").requires_grad = train_in_out[0]
        # self.cell.weight_ih_l0.requires_grad = train_in_out[0]
        # self.cell.bias_ih_l0.requires_grad = train_in_out[1]

    @property
    def w_in(self):
        return self.cell.weight_ih_l0

    @property
    def w_rec(self):
        return self.cell.weight_hh_l0

    def init_readout_weights(self, readout):
        try:
            nn.init.kaiming_uniform_(readout.weight, nonlinearity="relu")
        except AttributeError:
            [self.init_readout_weights(r) for r in self.readout]

    def forward(self, x_in, x_h=None, x_connect=0, softmax=False):
        """
        forward function of the Agent
        Params :
            x_in : input data
            x_h : previous state of the agent
            x_connect : sparse connections from other agents
            softmax : whether to use softmax layer
        """
        if hasattr(self, "sparse_in"):
            th, sign = self.sparse_in
            weight = self.cell_params("weight_ih_l0")
            weight = th * sign * (th > 0)

        if len(x_in.shape) < 3:
            x_in = x_in.unsqueeze(0)

        if self.dropout:
            x_in = self.dropout(x_in)

        if x_h is None:
            x, h = self.cell(x_in)
        else:
            if (
                type(self.cell) is nn.RNN
                or type(self.cell) is LeakyRNN
                or type(self.cell) is nn.GRU
            ):
                h = x_h + x_connect

            elif type(self.cell) is nn.LSTM:
                h = x_h[0] + x_connect, x_h[1]

            x, h = self.cell(x_in, h)

        output = x
        if self.dropout:
            output = self.dropout(output)
        if self.use_bottleneck:
            output = self.bottleneck(output)

        if self.use_readout:
            output = torch.cat([r(output) for r in self.readout])

        if softmax:
            output = F.log_softmax(output, dim=-1)

        return output, h

    def cell_params(self, name):
        if hasattr(self.cell, name):
            return getattr(self.cell, name)
        elif hasattr(self.cell, name + "_l0"):
            return getattr(self.cell, name + "_l0")
        elif hasattr(self.cell, name[:-3]):
            return getattr(self.cell, name[:-3])

    @property
    def device(self):
        return self.cell_params("weight_ih_l0").device


#### Sparse init from https://openreview.net/forum?id=2dgB38geVEU :

# functions for initializing the sparse matrices

# function that checks the Theorem 1 condition for an input square matrix
def check_cond(W):
    W_diag_only = np.diag(np.diag(W))
    W_diag_pos_only = W_diag_only.copy()
    W_diag_pos_only[W_diag_pos_only < 0] = 0.0
    W_abs_cond = np.abs(W - W_diag_only) + W_diag_pos_only
    max_eig_abs_cond = np.max(np.real(np.linalg.eigvals(W_abs_cond)))
    if max_eig_abs_cond < 1:
        return True
    else:
        return False


# sampling function to use for each non-zero element in a generated matrix
# (uniform but between -1 and 1 instead of default 0 to 1)
def uniform_with_neg(x):
    return np.random.uniform(low=-1.0, high=1.0, size=x)


# function that creates a square matrix of a given size with a given density and distribution (+ scalar for the distribution)
# also zeroes out the diagonal after generation
# the uniform_with_neg function is used as the sampling function by default


def generate_random_sparse_matrix(
    num_units, density, dist_multiplier, dist_func=uniform_with_neg
):
    test = scipy.sparse.random(
        num_units, num_units, density=density, format="csr", data_rvs=dist_func
    )
    np_test = test.toarray()
    np_test = dist_multiplier * np_test
    np.fill_diagonal(np_test, 0)
    return np_test


# create a full set of RNN modules using the above functions
# will only keep a given random RNN if it meets the condition for theorem 1
def create_modules(module_sizes, density, dist_multiplier, post_select_multiplier):
    modules = []
    for m in module_sizes:
        okay = False
        while not okay:
            cur_matrix = generate_random_sparse_matrix(m, density, dist_multiplier)
            okay = check_cond(cur_matrix)
        cur_matrix = (
            post_select_multiplier * cur_matrix
        )  # once reach here the current matrix is one of the ones selected
        modules.append(cur_matrix)
    return modules
