'''
Add hook to PyTorch's layers for using crb_backward functions for backprop
'''

# Author(s): Gaspar Rochette <gaspar.rochette@ens.fr>
# License: BSD 3 clause

import torch.nn as nn

from . import crb_backward


class Module(nn.Module):
    def __init__(self):
        '''Similar to torch.nn.Module but can compute per-example gradients.'''

        super().__init__()
        self.detail = False
        self.unfold_convolution_ = False

    def zero_grad(self):
        super().zero_grad()
        for p in self.parameters():
            p.bgrad = None

    def get_detail(self, b):
        '''Switches between batch-grad only and per-example computation.'''

        if b != self.detail:
            self.detail = b
            batchnorms = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
            for module in self.children():
                if isinstance(module, Module):
                    module.get_detail(b)
                elif isinstance(module, batchnorms) and b:
                    err = 'Can\'t compute per-example gradients with batchnorm layer.'
                    raise TypeError(err)

    def unfold_convolution(self, b):
        '''This is only defined for comparison purposes in the case of Conv2d.'''

        if b != self.unfold_convolution_:
            self.unfold_convolution_ = b
            for module in self.children():
                if isinstance(module, Module):
                    module.unfold_convolution(b)

    def save_for_detail(self):
        return self.detail and self.training


class ModuleList(nn.ModuleList, Module): pass


class Sequential(nn.Sequential, Module): pass


class Linear(nn.Linear, Module):
    '''Similar to torch.nn.Module but can compute per-example gradients.

    This class can only be used by itself or wrapped in a bnn.Module
    object (a nn.Module object would not use the extra functionalities).
    '''

    def forward(self, input):
        # compute output
        output = super().forward(input)

        if self.save_for_detail():
            def save_bgrad_hook(grad_output):
                w, b = crb_backward.linear_backward(input, grad_output,
                                            bias=self.bias is not None)
                self.weight.bgrad = w
                if self.bias is not None:
                    self.bias.bgrad = b
            output.register_hook(save_bgrad_hook)

        return output


class Conv1d(nn.Conv1d, Module):
    '''Similar to torch.nn.Module but can compute per-example gradients.

    This class can only be used by itself or wrapped in a bnn.Module
    object (a nn.Module object would not use the extra functionalities).
    '''

    def forward(self, input):
        output = super().forward(input)

        if self.save_for_detail():
            def save_bgrad_hook(grad_output):
                w, b = crb_backward.conv1d_backward(
                    input, grad_output,
                    self.in_channels, self.out_channels,
                    self.weight.shape[-1:],
                    bias=self.bias is not None,
                    stride=self.stride, dilation=self.dilation,
                    padding=self.padding, groups=self.groups
                )
                self.weight.bgrad = w
                if self.bias is not None:
                    self.bias.bgrad = b

            output.register_hook(save_bgrad_hook)

        return output


class Conv2d(nn.Conv2d, Module):
    '''Similar to torch.nn.Module but can compute per-example gradients.

    This class can only be used by itself or wrapped in a bnn.Module
    object (a nn.Module object would not use the extra functionalities).
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.unfold_convolution_:
            self.hook_fun = crb_backward.conv2d_backward
        else:
            self.hook_fun = crb_backward.conv2d_backward_unfold


    def forward(self, input):
        output = super().forward(input)

        if self.save_for_detail():
            def save_bgrad_hook(grad_output):
                w, b = self.hook_fun(
                    input, grad_output,
                    self.in_channels, self.out_channels,
                    self.weight.shape[-2:],
                    bias=self.bias is not None,
                    stride=self.stride, dilation=self.dilation,
                    padding=self.padding, groups=self.groups
                )
                self.weight.bgrad = w
                if self.bias is not None:
                    self.bias.bgrad = b

            output.register_hook(save_bgrad_hook)

        return output


class InstanceNorm1d(nn.InstanceNorm1d, Module):
    '''Similar to torch.nn.Module but can compute per-example gradients.

    This class can only be used by itself or wrapped in a bnn.Module
    object (a nn.Module object would not use the extra functionalities).
    '''

    def forward(self, input):
        output = super().forward(input)

        if self.save_for_detail() and self.affine:
            def save_bgrad_hook(grad_output):
                w, b = crb_backward.instance_norm_backward(input, grad_output)
                self.weight.bgrad = w
                self.bias.bgrad = b

            output.register_hook(save_bgrad_hook)

        return output


class InstanceNorm2d(nn.InstanceNorm2d, Module):
    '''Similar to torch.nn.Module but can compute per-example gradients.

    This class can only be used by itself or wrapped in a bnn.Module
    object (a nn.Module object would not use the extra functionalities).
    '''

    def forward(self, input):
        output = super().forward(input)

        if self.save_for_detail() and self.affine:
            def save_bgrad_hook(grad_output):
                w, b = crb_backward.instance_norm_backward(input, grad_output)
                self.weight.bgrad = w
                self.bias.bgrad = b

            output.register_hook(save_bgrad_hook)

        return output
