import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from copy import deepcopy


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
        use_readout=True,
        train_in_out=(True, False),
        cell_type=nn.RNN,
        use_bottleneck=False,
        ag_dual_readout=False,
        ag_dropout=0.0,
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

        self.use_readout = use_readout
        if self.use_bottleneck:
            if n_out == 100:
                n_bot = 10
            else:
                n_bot = 5
            self.bottleneck = nn.Linear(n_hidden, n_bot)
            self.readout = nn.Linear(n_bot, n_out)
        else:
            self.readout = nn.Linear(n_hidden, n_out)

        self.readout.weight.requires_grad = train_in_out[1]
        self.readout.bias.requires_grad = train_in_out[1]

        self.dual_readout = ag_dual_readout
        if ag_dual_readout:
            self.readout = nn.ModuleList([self.readout, deepcopy(self.readout)])
        else:
            self.readout = nn.ModuleList([self.readout])

        self.cell_params("weight_ih_l0").requires_grad = train_in_out[0]
        self.cell_params("bias_ih_l0").requires_grad = train_in_out[0]
        # self.cell.weight_ih_l0.requires_grad = train_in_out[0]
        # self.cell.bias_ih_l0.requires_grad = train_in_out[1]

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
