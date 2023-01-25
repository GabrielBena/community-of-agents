import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from deepR.models import weight_sampler_strict_number
from community.spiking.surrogate import super_spike


class MaskedLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sparsity: float,
        bias: bool = False,
        w_mask=None,
        weight_scale=1.0,
        out_scale=0.1,
        dropout=0.0,
        binarize=False,
        device=None,
        dtype=None,
    ) -> None:

        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        assert (w_mask is not None) or (
            sparsity is not None
        ), "Provide either weight mask or sparsity"

        if w_mask is not None:
            assert isinstance(w_mask, torch.Tensor) or isinstance(w_mask, np.ndarray)
            if isinstance(w_mask, np.ndarray):
                w_mask = torch.from_numpy(w_mask, requires_grad=False)
            self.register_buffer("w_mask", w_mask)
            self.sparsity = (w_mask > 0).float().mean()

        else:
            self.sparsity = sparsity
            n_in, n_out = in_features, out_features
            self.nb_non_zero = int(sparsity * n_in * n_out)

            w_mask = np.zeros((n_in, n_out), dtype=bool)
            # ind_in = rd.choice(np.arange(in_features),size=self.nb_non_zero)
            # ind_out = rd.choice(np.arange(out_features),size=self.nb_non_zero)

            ind_in, ind_out = np.unravel_index(
                np.random.choice(
                    np.arange(n_in * n_out), self.nb_non_zero, replace=False
                ),
                (n_in, n_out),
            )
            w_mask[ind_in, ind_out] = True
            w_mask = torch.tensor(w_mask)
            self.register_buffer("w_mask", w_mask)

        self.weight.requires_grad = True
        self.nb_neurons = in_features
        self.use_dropout = dropout > 0.0
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)

        self.binarize = binarize

        self.weight_scale = weight_scale
        self.out_scale = out_scale
        self.reset_parameters_()

        self.is_deepR_connect = False

    @property
    def w(self):
        return self.weight * self.w_mask

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # print(input.count_nonzero() / input.numel())
        h = F.linear(input, self.weight * self.w_mask, self.bias)
        if self.use_dropout:
            h = self.dropout(h)
        if self.binarize:
            h = super_spike(h)
        # print(h.count_nonzero(dim=1).max())
        return h

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, masked".format(
            self.in_features, self.out_features, self.bias is not None
        )

    def reset_parameters_(self, method="xavier") -> None:
        if method == "xavier":
            nn.init.xavier_normal_(
                self.weight, nn.init.calculate_gain("relu", self.weight)
            )
        else:

            if self.nb_neurons != 0:
                bound = self.weight_scale * np.sqrt(
                    1.0 / (self.nb_neurons)
                )  # *self.sparsity))
                nn.init.kaiming_normal_(self.weight)
                self.weight.data *= bound


class Sparse_Connect(nn.Module):
    """
    Sparse network to be trained with DeepR, used as connections in a global model
    Args :
        dims : dimensions of the network
        sparsity_list : sparsities of the different layers
    """

    def __init__(self, dims, sparsity_list, dropout=0.0, binarize=False):
        super().__init__()
        self.thetas = torch.nn.ParameterList()
        self.weight = torch.nn.ParameterList()
        self.signs = torch.nn.ParameterList()
        self.sparsity_list = sparsity_list
        self.out_features = dims[-1]
        self.bias = None
        self.nb_non_zero_list = [
            int(d1 * d2 * p) for (d1, d2, p) in zip(dims[:-1], dims[1:], sparsity_list)
        ]
        for d1, d2, nb_non_zero in zip(dims[:-1], dims[1:], self.nb_non_zero_list):
            w, w_sign, th, _ = weight_sampler_strict_number(d1, d2, nb_non_zero)
            self.thetas.append(th)
            self.weight.append(w)
            self.signs.append(w_sign)
            assert (w == w_sign * th * (th > 0)).all()

        self.is_deepR_connect = True
        self.use_dropout = dropout > 0.0
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)

        self.binarize = binarize

    def forward(self, x, relu=False):
        if type(x) is tuple:
            x = x[0]
        if len(x.shape) > 3:
            print(x.shape)
            # x = x.transpose(1, 2).flatten(start_dim=2)
        for i, (th, sign) in enumerate(zip(self.thetas, self.signs)):

            w = th * sign * (th > 0)
            # x = F.linear(x, w)
            x = torch.matmul(x, w)
            if relu:
                x = F.relu(x)

        if self.use_dropout:
            x = self.dropout(x)
        if self.binarize:
            x = super_spike(x)
        if self.out_scale:
            x *= self.out_scale
        return x
