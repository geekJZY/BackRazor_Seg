import torch
from torch import nn

from pdb import set_trace

import sys
sys.path.append(".")

import argparse
import builtins
import torch
from torch import nn, optim
from torch.distributed import Backend
from torch.nn.parallel.distributed import DistributedDataParallel

from network.custom_functions.masker import Masker
from network.custom_functions.custom_conv import SparseConv2d

from torch.nn import Conv2d


import network
from network.upernet import UperNet
import utils

import torch
import torch.nn as nn

from network.custom_functions.masker import Masker
from network.custom_functions.custom_conv import SparseConv2d
from network.custom_functions.custom_sync_bn import SparseBatchNorm2d

from pdb import set_trace

def replace_conv2d(model, **kwargs):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_conv2d(module, **kwargs)

        if isinstance(module, nn.Conv2d):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            transposed = module.transposed
            output_padding = module.output_padding
            groups = module.groups
            padding_mode = module.padding_mode

            new_module = SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                      dilation=dilation, groups=groups, bias=(module.bias is not None), **kwargs)
            new_module.load_state_dict(module.state_dict(), strict=False)

            assert not transposed
            assert padding_mode == "zeros"
            setattr(model, n, new_module)


def replace_BN(model, **kwargs):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_BN(module, **kwargs)

        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            eps = module.eps
            momentum = module.momentum
            affine = module.affine
            track_running_stats = module.track_running_stats

            new_module = SparseBatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                                           track_running_stats=track_running_stats, **kwargs)
            new_module.load_state_dict(module.state_dict())

            setattr(model, n, new_module)


def testGradDif(model1, model2):
    max_dif = -1
    max_dif_name = None

    grad_dict_model1 = {k: v.grad for k, v in model1.named_parameters() if v.requires_grad}
    grad_dict_model2 = {k: v.grad for k, v in model2.named_parameters() if v.requires_grad}

    for name1, grad1 in grad_dict_model1.items():
        if grad1 is not None:
            assert grad_dict_model2[name1] is not None
            # set_trace()

        dif = torch.norm(grad1 - grad_dict_model2[name1])
        print("grad name is {}, value is {}".format(name1, dif))
        if dif > max_dif:
            max_dif = dif
            max_dif_name = name1

    print("max grad name is {}, value is {}".format(max_dif_name, max_dif))


class ConfigMemoryTest(object):
    def __init__(self):
        self.hidden_size = 384
        self.quantize = False

        self.transformer = dict()
        self.transformer["attention_dropout_rate"] = 0
        self.transformer["num_heads"] = 12

        self.transformer["mlp_dim"] = 384
        self.transformer["dropout_rate"] = 0


def main():
    model = network.modeling.__dict__["deeplabv3_resnet50"](num_classes=21, output_stride=16)
    # model = UperNet(num_classes=21)
    # network.convert_to_separable_conv(model.classifier)
    # utils.set_bn_momentum(model.backbone, momentum=0.01)
    model = model.cuda()
    model = model.train()
    # model.classifier = model.classifier.eval()

    model_BR = network.modeling.__dict__["deeplabv3_resnet50"](num_classes=21, output_stride=16)
    # model_BR = UperNet(num_classes=21)
    # network.convert_to_separable_conv(model_BR.classifier)
    # utils.set_bn_momentum(model_BR, momentum=0.01)
    masker = Masker(prune_ratio=0)
    replace_conv2d(model_BR, masker=masker)
    replace_BN(model_BR, masker=masker)
    model_BR = model_BR.cuda()
    model_BR.load_state_dict(model.state_dict(), strict=False)


    input = torch.rand(6, 3, 224, 224).cuda()
    input.requires_grad = True
    input2 = input.clone().detach()
    input2.requires_grad = True

    model_BR.train()
    # model_BR.classifier.eval()
    # model.train()
    # with torch.set_deterministic(True):
    # output = model.backbone(input)["out"]
    output = model.backbone(input)
    output = model.classifier(output)
    output.sum().backward()
    input_grad_ori = input.grad

    # with torch.set_deterministic(True):
    output_BR = model_BR.backbone(input2)
    output_BR = model_BR.classifier(output_BR)
    output_BR.sum().backward()

    input_grad_our = input2.grad

    # the bug may lie in the conv with groups > 0
    print("output shape is {}".format(output.shape))
    print("output_BR  shape is {}".format(output_BR.shape))
    print("output is {}".format(output.mean(-1).mean(-1).mean(-1)))
    print("output_BR is {}".format(output_BR.mean(-1).mean(-1).mean(-1)))

    print("############ prune ratio of 0 #############")
    print("activation dist is {}".format(torch.norm(output_BR[0] - output[0])))
    # it would be good for slightly different, custom softmax layer often introduce some difference
    print("input grad dist is {}".format(torch.norm(input_grad_ori - input_grad_our)))

    set_trace()

    testGradDif(model, model_BR)


if __name__ == "__main__":
    main()
    # testStdConv()
    # testMlpStoreActivationPrune()

# TODO: run python -m torch.distributed.launch --nproc_per_node=2 test_custom_funcs/test_sync_bn.py
