import torch
import torch.nn as nn
import torch.distributed
from torch.autograd.function import Function
from .sparse_matrix import sparsify, unsparsify
from pdb import set_trace

import actnn.cpp_extension.backward_func as ext_backward_func

class batch_norm(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias, mask, half, quantize,
                training, exponential_average_factor, eps):

        shape_x, mask_x, sparse_x = sparsify(input, mask, with_batch_size=False)

        if half:
            sparse_x = sparse_x.half()

        if training:
            output, save_mean, save_var, reserve = ext_backward_func.cudnn_batch_norm(
                input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)
        else:
            output, save_mean, save_var = ext_backward_func.native_batch_norm(
                input, weight, bias, running_mean, running_var, training, exponential_average_factor, eps)
            reserve = None

        # ctx.other_args = input.shape
        ctx.saved = (shape_x, mask_x, sparse_x, weight, running_mean, running_var, save_mean, save_var, training, eps, reserve)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # if not ctx.needs_input_grad[3]:
        #     assert not ctx.needs_input_grad[0] and not ctx.needs_input_grad[4]
        #     return None, None, None, None, None, None, None, None, None
        shape_x, mask_x, sparse_x, weight, running_mean, running_var, save_mean, save_var, training, eps, reserve = ctx.saved

        # q_input_shape = ctx.other_args

        sparse_x = sparse_x.to(weight.dtype)
        input = unsparsify(shape_x, mask_x, sparse_x, with_batch_size=False)
        del sparse_x, ctx.saved

        # empty_cache(config.empty_cache_threshold)

        if training:
            input = input.contiguous()
            grad_input, grad_weight, grad_bias = ext_backward_func.cudnn_batch_norm_backward(
                input, grad_output, weight, running_mean, running_var, save_mean, save_var, eps, reserve)
        else:
            grad_input, grad_weight, grad_bias = ext_backward_func.native_batch_norm_backward(
                grad_output, input, weight, running_mean, running_var, save_mean, save_var, training, eps,
                [ctx.needs_input_grad[0], ctx.needs_input_grad[3], ctx.needs_input_grad[4]]
            )

        return grad_input, None, None, grad_weight, grad_bias, None, None, None, None, None, None


class sync_batch_norm(Function):
    @staticmethod
    def forward(self, input, weight, bias, mask, half, quantize, running_mean, running_var,
                eps, momentum, process_group, world_size):
        input = input.contiguous()

        count = torch.empty(1,
                            dtype=running_mean.dtype,
                            device=input.device).fill_(input.numel() // input.size(1))

        # calculate mean/invstd for input.
        mean, invstd = torch.batch_norm_stats(input, eps)

        num_channels = input.shape[1]
        # C, C, 1 -> (2C + 1)
        combined = torch.cat([mean, invstd, count], dim=0)
        # world_size * (2C + 1)
        combined_list = [
            torch.empty_like(combined) for k in range(world_size)
        ]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        torch.distributed.all_gather(combined_list, combined, process_group, async_op=False)
        combined = torch.stack(combined_list, dim=0)
        # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
        mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)

        size = count_all.view(-1).long().sum()
        if size == 1:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))

        # calculate global mean & invstd
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            count_all.view(-1)
        )

        shape_x, mask_x, sparse_x = sparsify(input, mask, with_batch_size=False)
        # self.saved = quantized
        self.save_for_backward(weight, mean, invstd, count_all, shape_x, mask_x, sparse_x)
        # self.other_args = input.shape
        self.process_group = process_group

        # apply element-wise normalization
        return torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()

        # quantized = self.saved
        # q_input_shape = self.other_args
        # saved_input = dequantize_activation(quantized, q_input_shape)
        # del quantized, self.saved

        weight, mean, invstd, count_tensor, shape_x, mask_x, sparse_x = self.saved_tensors
        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group

        sparse_x = sparse_x.to(weight.dtype)
        saved_input = unsparsify(shape_x, mask_x, sparse_x, with_batch_size=False)

        # calculate local stats as well as grad_weight / grad_bias
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
            grad_output,
            saved_input,
            mean,
            invstd,
            weight,
            self.needs_input_grad[0],
            self.needs_input_grad[1],
            self.needs_input_grad[2]
        )

        if self.needs_input_grad[0]:
            # synchronizing stats used to calculate input gradient.
            # TODO: move div_ into batch_norm_backward_elemt kernel
            num_channels = sum_dy.shape[0]
            combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
            torch.distributed.all_reduce(
                combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)
            sum_dy, sum_dy_xmu = torch.split(combined, num_channels)

            divisor = count_tensor.sum()
            mean_dy = sum_dy / divisor
            mean_dy_xmu = sum_dy_xmu / divisor
            # backward pass for gradient calculation
            grad_input = torch.batch_norm_backward_elemt(
                grad_output,
                saved_input,
                mean,
                invstd,
                weight,
                mean_dy,
                mean_dy_xmu
            )

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not self.needs_input_grad[1]:
            grad_weight = None

        if weight is None or not self.needs_input_grad[2]:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None


class SparseSyncBatchNorm(nn.SyncBatchNorm):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group=None,
        group=0,
        masker=None,
        quantize=False,
        half=False
    ) -> None:
        super(SparseSyncBatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats, process_group)
        self.masker = masker
        self.quantize = quantize
        self.half = half

        assert self.masker is not None
        assert not self.quantize

    def forward(self, input):
        # currently only GPU input is supported
        if not input.is_cuda:
            raise ValueError('SyncBatchNorm expected input tensor to be on GPU')

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        # If buffers are not to be tracked, ensure that they won't be updated
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)
        running_mean = self.running_mean if not self.training or self.track_running_stats else None
        running_var = self.running_var if not self.training or self.track_running_stats else None

        need_sync = bn_training
        if need_sync:
            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            mask = self.masker(input)

            return batch_norm().apply(
                input, running_mean, running_var, self.weight, self.bias, mask, self.half, self.quantize,
                bn_training, exponential_average_factor, self.eps)
        else:
            if not self.ddp_gpu_size:
                raise AttributeError('SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel')

            assert bn_training

            mask = self.masker(input)

            return sync_batch_norm().apply(
                input, self.weight, self.bias, mask, self.half, self.quantize, running_mean, running_var,
                self.eps, exponential_average_factor, process_group, world_size)


class SparseBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,
                 masker=None, quantize=False, half=False):
        super(SparseBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.masker = masker
        self.quantize = quantize
        self.half = half

        assert self.masker is not None
        assert not self.quantize

    def forward(self, input):
        if not self.training:
            return super(SparseBatchNorm2d, self).forward(input)

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        mask = self.masker(input)

        return batch_norm().apply(
            input, self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, mask, self.half, self.quantize,
            bn_training, exponential_average_factor, self.eps)
