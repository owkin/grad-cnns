'''Generate differentially-private optimizers in PyTorch'''

# Author: Andre Manoel <andre.manoel@owkin.com>
# License: BSD 3 clause

import torch

def make_optimizer(cls, noise_multiplier, l2_norm_clip):
    '''Returns differentially-private version of any PyTorch optimizer

    Parameters
    ----------
    cls : torch.optim object
        PyTorch optimizer to convert.
    noise_multiplier: float
        Ratio between gaussian noise and maximum gradient variations
    l2_norm_clip: float
        Global L2 norm threshold for model gradient.
    '''

    class DPOptimizer(cls):
        def __init__(self, params, **kwargs):
            self.noise_multiplier = noise_multiplier
            self.l2_norm_clip = l2_norm_clip

            super().__init__(params, **kwargs)

        def step(self, closure=None):
            '''Modify gradients then call `step` on original optimizer'''

            for group in self.param_groups:
                # Compute norm of sample-wise gradients
                # NOTE: p.bgrad must exist in order for this to work
                squared_norms = torch.stack([(p.bgrad.view(len(p.bgrad), -1) \
                        ** 2).sum(dim=1) for p in group['params']])
                grad_norm = torch.sqrt(squared_norms.sum(dim=0))

                for p in group['params']:
                    # Clip gradients
                    factor = (l2_norm_clip / len(p.bgrad)) / grad_norm
                    factor[factor > 1.0] = 1.0
                    clipped_grad = torch.tensordot(factor, p.bgrad,
                            dims=([0], [0])).clone()

                    # Replace gradients w/ noisy, norm-bounded version
                    std_noise = noise_multiplier * l2_norm_clip
                    p.grad = clipped_grad + (std_noise / len(p.bgrad)) * \
                            torch.randn_like(clipped_grad)

            super().step(closure)

    return DPOptimizer
