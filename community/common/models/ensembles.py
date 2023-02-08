from unittest.loader import VALID_MODULE_NAME
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .connections import MaskedLinear, Sparse_Connect
from .agents import Agent
from ...utils.model import get_output_shape
from copy import deepcopy


class Community(nn.Module):
    """
    Community, composed of sparsely connected agents.
    Params :
        agents : list of agents composing the community
        sparse_connections : (n_agents x n_agents) matrix specifying the sparsity of the connections in-between agents
    """

    def __init__(
        self,
        agents,
        sparsity,
        n_readouts=1,
        readout_from=None,
        readout_n_hid=None,
        use_deepR=True,
        binarize=False,
        comms_start="1",
        comms_dropout=0.0,
        comms_out_scale=0.1,
    ):

        super().__init__()

        self.agents = nn.ModuleList(agents)
        self.n_agents = len(agents)
        self.n_layers = agents[0].cell.num_layers
        self.agent_dims = agents[0].dims

        if sparsity > 1:
            sparsity /= 100

        sparse_connections = (
            np.ones((self.n_agents, self.n_agents)) - np.eye(self.n_agents)
        ) * sparsity

        self.sparse_connections = sparse_connections
        self.use_deepR = use_deepR
        self.comms_dropout = comms_dropout
        self.binarize = binarize
        self.comms_out_scale = comms_out_scale

        self.init_connections()
        self.is_community = True

        self.use_common_readout = n_readouts is not None

        if self.use_common_readout:
            self.initialize_readout(n_readouts, readout_from, readout_n_hid)

        self.comms_start = comms_start

    def gather(self, l, rf):
        return [l[i] for i in rf]

    def initialize_readout(self, n_readouts, readout_from, n_hid=None):

        self.multi_readout = type(n_readouts) is list or n_readouts > 1

        if readout_from is None:
            if type(n_readouts) is list:
                readout_from = [list(range(self.n_agents)) for r in n_readouts]
            else:
                readout_from = [list(range(self.n_agents)) for r in range(n_readouts)]
        else:
            if type(n_readouts) is list:
                assert len(readout_from) == len(
                    n_readouts
                ), f"Provide correct readout scheme {n_readouts, readout_from}"
            else:
                assert (
                    len(readout_from) == n_readouts
                ), f"Provide correct readout scheme {n_readouts, readout_from}"

        self.readout_from = readout_from

        readout_dims = self.get_readout_dimensions(n_readouts, readout_from, n_hid)
        self.readout = self.create_readout_from_dims(readout_dims)

    def init_readout_weights(self, readout):
        try:
            nn.init.kaiming_uniform_(readout.weight, nonlinearity="relu")
        except AttributeError:
            [self.init_readout_weights(r) for r in self.readout]

    @property
    def min_t_comms(self):
        if self.comms_start == "start":
            min_t_comms = 1
        elif self.comms_start == "mid":
            min_t_comms = self.nb_steps // 2
        elif self.comms_start == "last":
            min_t_comms = self.nb_steps - 1
        elif isinstance(self.comms_start, int):
            min_t_comms = self.comms_start
        else:
            raise NotImplementedError
        return min_t_comms

    def get_readout_dimensions(self, n_readout, readout_from, n_hid):

        if type(n_readout) is list:

            return [
                self.get_readout_dimensions(nr, rf, n_hid)
                for nr, rf in zip(n_readout, readout_from)
            ]

        else:
            try:
                readout_dims = [
                    [
                        np.sum([ag.dims[-2] for ag in self.gather(self.agents, rf)]),
                        self.agents[0].dims[-1],
                    ]
                    for rf in readout_from
                ]
            except TypeError:
                readout_dims = [
                    [
                        np.sum(
                            [
                                ag.dims[-2]
                                for ag in self.gather(self.agents, readout_from)
                            ]
                        ),
                        self.agents[0].dims[-1],
                    ]
                    for _ in range(n_readout)
                ]

        def insert_dim(readout_dims, dim, idx=1):
            try:
                [insert_dim(r, dim) for r in readout_dims]
            except (TypeError, AttributeError) as e:
                readout_dims.insert(idx, dim)

        if n_hid is not None:
            insert_dim(readout_dims, n_hid)

        return readout_dims

    def create_readout_from_dims(self, readout_dims):

        try:
            readout = nn.ModuleList(
                [self.create_readout_from_dims(r_dim) for r_dim in readout_dims]
            )
        except (TypeError, IndexError) as e:
            readout = [
                nn.Linear(d1, d2) for d1, d2 in zip(readout_dims[:-1], readout_dims[1:])
            ]

            if len(readout) == 1:
                readout = readout[0]
            else:
                readout.insert(1, nn.ReLU())
                readout = nn.Sequential(*readout)

        return readout

    # Initializes connections in_between agents with the given sparsity matrix
    def init_connections(self):

        self.tags = np.empty((self.n_agents, self.n_agents), dtype=object)
        self.connected = np.zeros((self.n_agents, self.n_agents))
        self.connections = nn.ModuleDict()

        for i, ag1 in enumerate(self.agents):
            i = int(ag1.tag)
            for j, ag2 in enumerate(self.agents):
                j = int(ag2.tag)
                if i == j:
                    continue
                else:
                    p_con = self.sparse_connections[i, j]
                    if p_con > 0:
                        # State-to-State connections :
                        n_in = ag1.dims[-2]
                        n_out = ag2.dims[-2]

                        # Output-to-Input connections:
                        # n_in = ag1.dims[-1]
                        # n_out = ag2.dims[0]

                        dims = [n_in, n_out]
                        sparsity_list = [p_con]
                        if self.use_deepR:
                            connection = Sparse_Connect(
                                dims, sparsity_list, self.comms_dropout, self.binarize
                            )
                        else:
                            connection = MaskedLinear(
                                *dims,
                                p_con,
                                dropout=self.comms_dropout,
                                binarize=self.binarize,
                                out_scale=self.comms_out_scale,
                            )
                        self.tags[i, j] = ag1.tag + ag2.tag
                        self.connections[self.tags[i, j]] = connection
                        self.connected[i, j] = 1

    # Forward function of the community
    def forward(self, x, forced_comms=None, state_masks=None):

        if state_masks is not None:
            state_masks = torch.tensor(state_masks).to(x.device)

        # Split_data checks if the data is provided on a per-agent manner or in a one-to-all manner.
        # data can be a double list of len n_timesteps x n_agents or a tensor with second dimension n_agents
        split_data = (type(x) is torch.Tensor and len(x.shape) > 3) or type(x) is list
        self.nb_steps = len(x)
        ag_outputs = [[] for ag in self.agents]
        outputs = []
        states = [[None] for ag in self.agents]
        connections = [[] for ag in self.agents]

        for t, x_t in enumerate(x):
            if split_data:
                assert len(x_t) == len(
                    self.agents
                ), "If data is provided per agent, make sure second dimension matches number of agents"
            ag_states = []
            for ag1 in self.agents:
                i = int(ag1.tag)
                if split_data:
                    inputs = x_t[i].clone()
                else:
                    inputs = x_t.clone()

                # Receive sparse connections from other agents

                inputs_connect = 0

                if t >= self.min_t_comms:
                    if forced_comms is None:
                        for ag2 in self.agents:
                            j = int(ag2.tag)
                            if self.connected[j, i] == 1:
                                # State-to-state connections j->i
                                sparse_out = self.connections[ag2.tag + ag1.tag](
                                    states[j][-1]
                                )
                                inputs_connect += sparse_out[0]

                                # Output-to-Input connections j->i
                                # sparse_out = self.connections[self.tags[j, i]](outputs[n-1][j])
                                # inputs += sparse_out[0]
                    else:
                        inputs_connect = forced_comms[:, ag1]

                    # out, h = ag1(inputs, states[t-1][i], inputs_connect)

                out, h = ag1(inputs, states[i][-1], inputs_connect)
                ag_outputs[i].append(out)

                if state_masks is not None:
                    h *= state_masks[i]

                ag_states.append(h)
                connections[i].append(inputs_connect)
            # Store states and outputs of agent
            for i, state in enumerate(ag_states):
                states[i].append(state)

            if self.use_common_readout:

                def readout_process(readout, readout_from, input):

                    try:
                        gathered_input = torch.cat(
                            [s[-1] for s in self.gather(input, readout_from)], -1
                        )
                        out = readout(gathered_input)[0]
                    except (NotImplementedError, TypeError) as e:
                        out = torch.stack(
                            [
                                readout_process(r, rf, input)
                                for r, rf in zip(readout, readout_from)
                            ]
                        )

                    return out

                """
                try:
                    out = readout_process(self.readout, self.readout_from, states)
                except RuntimeError:  # Use bottleneck as out
                    out = readout_process(self.readout, self.readout_from, out)
                """
                common_out = readout_process(
                    self.readout, self.readout_from, ag_outputs
                )
                outputs.append(common_out)
            else:
                outputs = ag_outputs

        if not self.use_common_readout:
            outputs = torch.stack([torch.stack(o) for o in outputs], 1)
        else:
            outputs = torch.stack(outputs, 0)

        if self.n_layers > 1:
            states = torch.stack(
                [torch.stack([s[-1] for s in st[1:]]) for st in states], 1
            )
        else:
            states = torch.stack([torch.cat(s[1:]) for s in states], 1)
        try:
            connections = torch.stack(
                [torch.stack(c[self.min_t_comms :]) for c in connections], 1
            )
        except TypeError:
            connections = torch.tensor(connections)

        return outputs.squeeze(), states, ag_outputs, connections

    @property
    def w_rec_global(self):
        n_hid = self.agent_dims[1]
        w_rec = torch.zeros((self.n_agents * n_hid, self.n_agents * n_hid))

        for i, ag1 in enumerate(self.agents):
            for j, ag2 in enumerate(self.agents):
                if i != j:
                    conns = self.connections[f"{ag1.tag}{ag2.tag}"]
                    w_rec[
                        i * n_hid : (i + 1) * n_hid, j * n_hid : (j + 1) * n_hid
                    ] = conns.w

            w_rec[i * n_hid : (i + 1) * n_hid, i * n_hid : (i + 1) * n_hid] = ag1.w_rec
        return w_rec

    @property
    def nb_connections(self):
        """
        Returns dictionnary of number of active connections of community
        """
        if self.use_deepR:
            return {
                tag: (c.thetas[0] > 0).int().sum()
                for tag, c in zip(self.connections.keys(), self.connections.values())
            }
        else:
            return {
                tag: (c.w_mask > 0).int().sum()
                for tag, c in zip(self.connections.keys(), self.connections.values())
            }


class ConvCommunity(nn.Module):
    """
    Community of mind, composed of sparsely connected agents, with a global CNN input layer.
    Params :
        input_shape : shape of input to determine
        use_deciding_agent : Wheter to use a separate agent to take decsions, connected to all other in a straightforward manner
    """

    def __init__(
        self,
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
        distance_based_sparsity=False,
        cell_type=nn.RNN,
        use_deciding_agent=False,
        use_sparse_readout=False,
        use_attention=False,
    ):

        # If only one value provided, then it is duplicated for each layer
        if len(kernel_size) == 1:
            kernel_size = kernel_size * num_conv_layers
        if stride is None:
            stride = [1]
        if len(stride) == 1:
            stride = stride * num_conv_layers
        if pool_size is None:
            pool_size = [1]
        if len(pool_size) == 1:
            pool_size = pool_size * num_conv_layers
        if Nhid is None:
            self.Nhid = Nhid = []

        super().__init__()
        self.is_community = True

        self.input_shape = input_shape

        self.decouple = decouple
        if not decouple:
            Nhid = [input_shape[0]] + Nhid
        else:
            Nhid = [1] + Nhid

        feature_height = self.input_shape[1]
        feature_width = self.input_shape[2]
        padding = (np.array(kernel_size) - 1) // 2

        self.cnns = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        for i in range(num_conv_layers):
            feature_height, feature_width = get_output_shape(
                [feature_height, feature_width],
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i],
                dilation=1,
            )

            feature_height //= pool_size[i]
            feature_width //= pool_size[i]

            if decouple:
                cnn = nn.ModuleList()
                for _ in range(input_shape[0]):
                    cnn.append(
                        nn.Conv2d(
                            Nhid[i], Nhid[i + 1], kernel_size[i], stride[i], padding[i]
                        )
                    )

            else:
                cnn = nn.Conv2d(
                    Nhid[i], Nhid[i + 1], kernel_size[i], stride[i], padding[i]
                )
            self.cnns.append(cnn)

            pool = nn.MaxPool2d(kernel_size=pool_size[i])
            self.pool_layers.append(pool)

        n_in = feature_height * feature_width
        n_agents = Nhid[-1]
        if decouple:
            n_agents *= input_shape[0]

        self.dims = [n_in, n_hid, n_out]
        use_readout = not (use_deciding_agent or use_sparse_readout)

        agents = [
            Agent(
                n_in,
                n_hid,
                1,
                n_out,
                str(n),
                cell_type=cell_type,
                use_readout=use_readout,
            )
            for n in range(n_agents)
        ]
        self.agents = nn.ModuleList()

        for ag in agents:
            self.agents.append(ag)
        self.n_agents = len(agents)

        self.p_connect = p_connect
        self.sparse_connections = (
            np.ones((n_agents, n_agents)) - np.eye(n_agents)
        ) * p_connect
        if distance_based_sparsity:
            self.sparse_connections[n_agents // 2 :, : n_agents // 2] *= 0.5
            self.sparse_connections[: n_agents // 2, n_agents // 2 :] *= 0.5

        self.use_deciding_agent = use_deciding_agent
        self.use_sparse_readout = use_sparse_readout
        self.use_attention = use_attention

        self.init_connections()

    def init_sparse_readout(self, n_out=None, p_out=None):
        if n_out is None:
            n_out = self.dims[-1]
        if p_out is None:
            p_out = self.p_connect

        n_in = np.sum(np.array([ag.dims[-2] for ag in self.agents]))
        if self.decouple:
            readout = nn.ModuleList()
            n_in //= 2
            for i in range(2):
                decoupled_readout = nn.Linear(n_in, n_out)
                # decoupled_readout = deepR.Sparse_Connect([n_in, n_out], [p_out])
                readout.append(decoupled_readout)
            self.connections["readout"] = readout
            readout.is_sparse_connect = False
        else:
            readout = Sparse_Connect([n_in, n_out], [p_out])
            # readout = nn.Linear(n_in, n_out)
            self.connections["readout"] = readout

    def init_deciding_agent(self, n_in, n_hid, n_out):
        cell_type = self.agents[0].cell.__class__
        decider = Agent(n_in, n_hid, n_out, "decider", True, cell_type=cell_type)
        self.decider = decider

    # Initializes connections in_between agents with the given sparsity matrix
    def init_connections(self, sparse_connections=None):

        if sparse_connections is None:
            sparse_connections = self.sparse_connections

        self.tags = np.empty((self.n_agents, self.n_agents), dtype=object)
        self.connected = np.zeros((self.n_agents, self.n_agents))
        self.connections = nn.ModuleDict()

        for i, ag1 in enumerate(self.agents):
            for j, ag2 in enumerate(self.agents):
                if i == j:
                    continue
                else:
                    p_con = sparse_connections[i, j]
                    if p_con > 0:
                        # State-to-State connections :

                        if self.use_bottleneck:
                            n_in = ag1.dims[-3]
                            n_out = ag2.dims[-3]
                        else:
                            n_in = ag1.dims[-2]
                            n_out = ag2.dims[-2]
                        # Output-to-Input connections:
                        # n_in = ag1.dims[-1]
                        # n_out = ag2.dims[0]

                        dims = [n_in, n_out]
                        sparsity_list = [p_con]
                        connection = Sparse_Connect(dims, sparsity_list)
                        self.tags[i, j] = ag1.tag + "-" + ag2.tag
                        self.connections[self.tags[i, j]] = connection
                        self.connected[i, j] = 1

        if self.use_sparse_readout:
            if self.use_deciding_agent:
                n_hid = self.dims[-2]
                n_hid *= 2
                self.init_deciding_agent(n_hid, n_hid, self.dims[-1])
                self.init_sparse_readout(n_hid, 0.25)
            else:
                self.init_sparse_readout(self.dims[-1], 0.25)
        else:
            if self.use_deciding_agent:
                n_in = np.sum(np.array([ag.dims[-2] for ag in self.agents]))
                n_hid = self.dims[-2]
                n_hid *= 2
                self.init_deciding_agent(n_in, n_hid, self.dims[-1])

    def process_input(self, x_t):
        for l, (cnn, pool) in enumerate(zip(self.cnns, self.pool_layers)):
            if self.decouple:
                if l == 0:
                    x_t = [x_t[:, i, ...].unsqueeze(1) for i in range(2)]
                x_t = [cnn[i](x_t_d) for i, x_t_d in enumerate(x_t)]
                x_t = [pool(x_t_d) for x_t_d in x_t]

            else:
                x_t = cnn(x_t)
                x_t = pool(x_t)

        if self.decouple:
            x_t = torch.cat(x_t, axis=1)

        # print(x_t.shape)
        x_t = x_t.flatten(start_dim=-2).transpose(0, 1)
        return x_t

    def mask_mult(self, data, mask):
        if len(mask.shape) > 0:
            data = torch.einsum("abc, b -> abc", data, mask.to(data.device))
        else:
            data = data * mask
        return data

    # Forward function of the community
    def forward(self, x, supsup=False, temporal_masks=False):
        outputs = []
        states = []
        states_decider = []
        # Split_data checks if the data is provided on a per-agent manner or in a one-to-all manner.
        # Split_data can be a double list of len n_timesteps x n_agents or a tensor with second dimension n_agents
        for n, x_t in enumerate(x):

            x_t = self.process_input(x_t)

            output = [None for ag in self.agents]
            state = [None for ag in self.agents]
            if self.use_deciding_agent:
                state_decider = torch.zeros(
                    (1, x_t.shape[1], self.decider.dims[-2])
                ).to(self.decider.device)
                # print(state_decider.shape)

            for i, ag1 in enumerate(self.agents):
                inputs = x_t[i]

                # Receive sparse connections from other agents
                if n > 0:
                    inputs_connect = 0
                    if not supsup:
                        for j, ag2 in enumerate(self.agents):
                            if self.connected[j, i] == 1:
                                # State-to-state connections j->i
                                sparse_out = self.connections[self.tags[j, i]](
                                    states[n - 1][j]
                                )
                                # print(sparse_out.shape)
                                inputs_connect += sparse_out[:]

                                # Output-to-Input connections j->i
                                # sparse_out = self.connections[self.tags[j, i]](outputs[n-1][j])
                                # inputs += sparse_out[0]

                        out, h = ag1(inputs, states[n - 1][i], inputs_connect)
                    else:
                        if temporal_masks:
                            mask = self.connection_masks[n - 1]
                        else:
                            mask = self.connection_masks

                        connected_to = mask.max(axis=1)[0]
                        connected_from = mask.max(axis=0)[0]
                        # use_ag = (connected_from[i])
                        # use_ag = self.output_masks[i]
                        use_ag = mask[i, i]
                        for j, ag2 in enumerate(self.agents):
                            connected_mask = mask[j, i]
                            connected_global = self.connected[j, i]
                            batch_mask = len(connected_mask.shape) > 0

                            if batch_mask:
                                connected = connected_global
                            else:
                                connected = connected_global * connected_mask
                            if connected:
                                # State-to-state connections j->i
                                sparse_out = self.connections[self.tags[j, i]](
                                    states[n - 1][j]
                                )
                                # print(sparse_out.shape, connected_local.shape)
                                sparse_out = self.mask_mult(sparse_out, connected_mask)
                                inputs_connect += sparse_out[:]

                        out, h = outs = ag1(inputs, states[n - 1][i], inputs_connect)
                        out, h = self.mask_mult(outs[0], use_ag), outs[1]
                else:
                    use_ag = self.output_masks[i] if supsup else torch.tensor(1)
                    out, h = outs = ag1(inputs)
                    # out, h = self.mask_mult(outs[0], use_ag), outs[1]

                # print(out.shape)

                output[i] = out
                state[i] = h

            # Store states and outputs of agent
            states.append(state)
            output = torch.cat(output)

            # print(output.shape)
            if self.use_attention:
                output = output.transpose(0, 1)
                output, scores = self.attention_layer(
                    output, state_decider.transpose(0, 1), state_decider.transpose(0, 1)
                )
                output = output.transpose(0, 1)
                # return scores

            if "readout" in self.connections.keys():
                if self.decouple:
                    n_decouple = self.n_agents // 2
                    output = [
                        output[:n_decouple].transpose(0, 1).flatten(start_dim=1),
                        output[n_decouple:].transpose(0, 1).flatten(start_dim=1),
                    ]

                    output = [r(o) for r, o in zip(self.connections["readout"], output)]
                    # print([o.shape for o in output])
                    output = torch.stack(output)

                else:
                    output = output.transpose(0, 1).flatten(start_dim=1)
                    output = self.connections["readout"](output)
                    if self.use_deciding_agent:
                        output, state_decider = self.decider(output, state_decider)
                        output = output[0]
                        states_decider.append(state_decider)
            else:
                if self.use_deciding_agent:
                    output = output.transpose(0, 1).flatten(start_dim=1)
                    output, state_decider = self.decider(output, state_decider)
                    output = output[0]
                    states_decider.append(state_decider)

            outputs.append(output)

        outputs = torch.stack(outputs)

        return outputs, (states, 0)

    @property
    def nb_connections(self):
        """
        Returns dictionnary of number of active connections of community
        """
        nb_connections = {}
        for tag, c in zip(self.connections.keys(), self.connections.values()):
            if type(c) is Sparse_Connect:
                nb_connected = (c.thetas[0] > 0).int().sum()
                nb_connections[tag] = nb_connected

        return nb_connections
