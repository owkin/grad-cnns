'''Create PyTorch "multi-model" containing multiple pointers to a given model'''

# Author(s): Gaspar Rochette <gaspar.rochette@ens.fr>
#            Andre Manoel <andre.manoel@owkin.com>

from copy import deepcopy

import torch
import torch.nn as nn

def replicate_model(net_class, batch_size):
    class MultiNet(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.batch_size = batch_size
            self.model = net_class(**kwargs)

            if hasattr(self.model, 'input_size'):
                self.input_size = self.model.input_size

            models = []
            for _ in range(self.batch_size):
                models.append(deepcopy(self.model))
                for p, p_true in zip(models[-1].parameters(), self.model.parameters()):
                    p.data = p_true.data
            self.models = nn.ModuleList(models)

            self.parameters = self.model.parameters

        def forward(self, x):
            if self.training:
                assert x.size(0) == self.batch_size
                x = x.split(1, dim=0)
                y = torch.cat([model(x_i) for x_i, model in zip(x, self.models)], dim=0)
            else:
                y = self.model(x)

            return y

        def reduce_batch(self):
            """Puts per-example gradient in p.bgrad and aggregate on p.grad"""
            params = zip(*[model.parameters() for model in self.models])  # group per-layer
            for p, p_multi in zip(self.model.parameters(), params):
                p.bgrad = torch.stack([p_.grad for p_ in p_multi], dim=0)
                for p_ in p_multi:
                    p_.grad = None

        def reassign_params(self):
            """Reassign parameters of sub-models to those of the main model"""
            for model in self.models:
                for p, p_true in zip(model.parameters(), self.model.parameters()):
                    p.data = p_true.data

        def get_detail(self, b): pass  # for compatibility with crb.nn.Module

    return MultiNet
