import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaskedLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, sparsity: float,  bias: bool = True, w_mask=None, weight_scale=None,
                 device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        assert (w_mask is not None) or (sparsity is not None), 'Provide either weight mask or sparsity'

        if w_mask is not None:
            assert isinstance(w_mask, torch.Tensor) or isinstance(w_mask, np.ndarray)
            if isinstance(w_mask, np.ndarray):
                w_mask = torch.from_numpy(w_mask, requires_grad=False)
            self.register_buffer('w_mask', w_mask)
            self.sparsity = (w_mask>0).float().mean()

        else : 
            self.sparsity = sparsity
            n_in, n_out = in_features, out_features
            self.nb_non_zero = int(sparsity*n_in*n_out)
        
            w_mask = np.zeros((n_in, n_out),dtype=bool)
            #ind_in = rd.choice(np.arange(in_features),size=self.nb_non_zero)
            #ind_out = rd.choice(np.arange(out_features),size=self.nb_non_zero)

            ind_in, ind_out = np.unravel_index(np.random.choice(np.arange(n_in*n_out), self.nb_non_zero, replace=False), (n_in, n_out))
            w_mask[ind_in,ind_out] = True
            w_mask = torch.tensor(w_mask)
            self.register_buffer('w_mask', w_mask)

        self.weight.requires_grad = True
        self.nb_neurons = in_features
        
        self.weight_scale = 1. if weight_scale is None else weight_scale
        self.reset_parameters_()

        self.is_deepR_connect = False
            
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight*self.w_mask, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, masked'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def reset_parameters_(self) -> None:
        if self.nb_neurons != 0:
            bound = self.weight_scale*np.sqrt(1. / (self.nb_neurons))#*self.sparsity))
            nn.init.kaiming_uniform_(self.weight)
            self.weight.data *= bound
