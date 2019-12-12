'''
Recode backprop for linear and convolutional layers, ensuring trick is applied
'''

# Author(s): Gaspar Rochette <gaspar.rochette@ens.fr>
# License: BSD 3 clause

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_backward(input, grad_output, bias=True):
    '''Computes per-example gradients for nn.Linear layers.

    This function is used in the internal behaviour of bnn.Linear.
    '''
    weight_bgrad = torch.bmm(grad_output.unsqueeze(2), input.unsqueeze(1))
    bias_bgrad = grad_output if bias else None

    return weight_bgrad, bias_bgrad


def conv_backward(input, grad_output, in_channels, out_channels, kernel_size,
                  bias=True, stride=1, dilation=1, padding=0, groups=1, nd=1):
    '''Computes per-example gradients for nn.Conv1d and nn.Conv2d layers.

    This function is used in the internal behaviour of bnn.Linear.
    '''

    # Change format of stride from int to tuple if necessary.
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * nd
    if isinstance(stride, int):
        stride = (stride,) * nd
    if isinstance(dilation, int):
        dilation = (dilation,) * nd
    if isinstance(padding, int):
        padding = (padding,) * nd

    # Get some useful sizes
    batch_size = input.size(0)
    input_shape = input.size()[-nd:]
    output_shape = grad_output.size()[-nd:]

    # Reshape to extract groups from the convolutional layer
    # Channels are seen as an extra spatial dimension with kernel size 1
    input_conv = input.view(1, batch_size * groups, in_channels // groups, *input_shape)

    # Compute convolution between input and output; the batchsize is seen
    # as channels, taking advantage of the `groups` argument
    grad_output_conv = grad_output.view(-1, 1, 1, *output_shape)

    stride = (1, *stride)
    dilation = (1, *dilation)
    padding = (0, *padding)

    if nd == 1:
        convnd = F.conv2d
        s_ = np.s_[..., :kernel_size[0]]
    elif nd == 2:
        convnd = F.conv3d
        s_ = np.s_[..., :kernel_size[0], :kernel_size[1]]
    elif nd == 3:
        raise NotImplementedError('3d convolution is not available with current per-example gradient computation')

    conv = convnd(
        input_conv, grad_output_conv,
        groups=batch_size * groups,
        stride=dilation,
        dilation=stride,
        padding=padding
    )

    # Because of rounding shapes when using non-default stride or dilation,
    # convolution result must be truncated to convolution kernel size
    conv = conv[s_]

    # Reshape weight gradient to correct shape
    new_shape = [batch_size, out_channels, in_channels // groups, *kernel_size]
    weight_bgrad = conv.view(*new_shape).contiguous()

    # Compute bias gradient
    grad_output_flat = grad_output.view(batch_size, grad_output.size(1), -1)
    bias_bgrad = torch.sum(grad_output_flat, dim=2) if bias else None

    return weight_bgrad, bias_bgrad


def conv1d_backward(*args, **kwargs):
    '''Computes per-example gradients for nn.Conv1d layers.

    This function is used in the internal behaviour of bnn.Linear.
    '''
    return conv_backward(*args, nd=1, **kwargs)


def conv2d_backward(*args, **kwargs):
    '''Computes per-example gradients for nn.Conv2d layers.

    This function is used in the internal behaviour of bnn.Linear.
    '''
    return conv_backward(*args, nd=2, **kwargs)


def conv2d_backward_unfold(input, grad_output, in_channels, out_channels, kernel_size,
                           bias=True, stride=1, dilation=1, padding=0, groups=1):
    '''Computes per-example gradients for nn.Conv2d layers with `unfold` option.

    This function is used in the internal behaviour of bnn.Linear.
    '''

    batch_size = input.size(0)
    input_shape = input.size()[-2:]
    output_shape = grad_output.size()[-2:]

    input_unfolded = F.unfold(input, output_shape)
    grad_output_unfolded = F.unfold(grad_output, output_shape)

    input_unfolded = input_unfolded.view(batch_size, 1, in_channels, -1, *kernel_size)
    grad_output_unfolded = grad_output_unfolded.view(batch_size, out_channels, 1, -1, 1, 1)
    weight_bgrad = torch.sum(grad_output_unfolded * input_unfolded, dim=3)

    bias_bgrad = torch.sum(torch.sum(grad_output, dim=3), dim=2) if bias else None

    return weight_bgrad, bias_bgrad


def instance_norm_backward(input, grad_output):
    norm_input = F.instance_norm(input)  # shape (B, C, N1, N2)

    batch_size, channels = input.size(0), input.size(1)
    weight_bgrad = torch.sum(
        (norm_input * grad_output).view(batch_size, channels, -1),
        dim=2
    )

    bias_bgrad = torch.sum(
        grad_output.view(batch_size, channels, -1),
        dim=2
    )

    return weight_bgrad, bias_bgrad
