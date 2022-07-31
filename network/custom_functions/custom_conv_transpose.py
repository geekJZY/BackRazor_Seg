import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import logging

from mesa import custom_quant
from mesa import native
from mesa import packbit

import actnn.cpp_extension.backward_func as ext_backward_func

from torch.nn.modules.utils import _pair
from .sparse_matrix import sparsify, unsparsify
from pdb import set_trace

# Uniform Quantization based Convolution
class conv_transpose2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, mask, stride, padding, output_padding, groups, dilation,
                clip_val, level, iteration, ema_decay, quant_groups, shift):
        shape_x, mask_x, sparse_x = sparsify(input, mask, with_batch_size=False)

        # custom_quant.Quant.forward(ctx, x, clip_val, level, iteration, ema_decay, quant_groups, shift)

        ctx.save_for_backward(shape_x, mask_x, sparse_x, weight, bias)
        ctx.other_args = (input.shape, stride, padding, output_padding, dilation, groups)

        return F.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)

    @staticmethod
    def backward(ctx, grad_output):
        q_input_shape, stride, padding, output_padding, dilation, groups = ctx.other_args
        padding = _pair(padding)
        output_padding = _pair(output_padding)
        stride = _pair(stride)
        dilation = _pair(dilation)

        shape_x, mask_x, sparse_x, weight, bias = ctx.saved_tensors
        input = unsparsify(shape_x, mask_x, sparse_x)
        # x = custom_quant.Quant.restore(ctx)

        grad_input, grad_weight = ext_backward_func.cudnn_convolution_transpose_backward(
            input, grad_output, weight, padding, output_padding, stride, dilation, groups,
            True, False, False, [ctx.needs_input_grad[0], ctx.needs_input_grad[1]])

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum([0, 2, 3])
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, \
               None, None, None, None, None, None, None


# class conv_transpose2d(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, scheme=None):
#         return conv_transposend.run_forward(2, F.conv_transpose2d, ctx, input, weight, bias, stride,
#                                             padding, output_padding, groups, dilation, scheme)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         return conv_transposend.run_backward(2, ctx, grad_output, [0, 2, 3], _pair)


class SparseConvTranspose2d(nn.ConvTranspose2d, custom_quant.Quant):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros', args=None, logger=None, quant_groups=1,
                 masker=None, act_prune=False):
        super(SparseConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                                    padding, output_padding, groups, bias, dilation, padding_mode)
        custom_quant.Quant.__init__(self, args=args, logger=logger, quant_groups=quant_groups)
        self.masker = masker
        self.act_prune = act_prune
        # print("act_prune is {}".format(act_prune))
        # self.tag = 'conv'

    def __repr__(self):
        return self.__str__()

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size, self.stride, self.padding,
                                              self.kernel_size, self.dilation)  # type: ignore
        if self.masker is not None and self.training:
            mask = self.masker(x)
            if self.act_prune:
                # apply mask to activation in the forward
                x = x * mask

            y = conv_transpose2d.apply(x, self.weight, self.bias, mask, self.stride, self.padding, output_padding,
                                       self.groups, self.dilation, self.clip_val, self.level, self.iteration,
                                       self.ema_decay, self.quant_groups, self.shift)
        else:
            y = F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, output_padding, self.groups,
                                   self.dilation)
        return y
