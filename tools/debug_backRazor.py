import network
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


def main():
    model = network.modeling.__dict__["deeplabv3plus_mobilenet"](num_classes=21, output_stride=16)
    # network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    model = model.cuda()

    model_BR = network.modeling.__dict__["deeplabv3plus_mobilenet"](num_classes=21, output_stride=16)
    # network.convert_to_separable_conv(model_BR.classifier)
    utils.set_bn_momentum(model_BR.backbone, momentum=0.01)
    masker = Masker(prune_ratio=0)
    replace_conv2d(model_BR, masker=masker)
    model_BR = model_BR.cuda()
    model_BR.load_state_dict(model.state_dict(), strict=False)

    input = torch.rand(6, 3, 224, 224).cuda()

    # output = model(input)
    # output_BR = model_BR(input)

    # # low level feature is the same
    # output = model.backbone.low_level_features(input)
    # output_BR = model_BR.backbone.low_level_features(input)
    #
    # # low level feature 0 is the same
    # # high level feature 0-7 is the same
    # # high level feature is the same
    # output = model.backbone.high_level_features[0:](output)
    # output_BR = model_BR.backbone.high_level_features[0:](output_BR)

    # backbone is the same
    output = model.backbone(input)
    output_BR = model_BR.backbone(input)

    # for name, item in output.items():
    #     print("name is {}".format(name))
    #     print("item shape is {}".format(output[name].shape))
    #     print("item_BR  shape is {}".format(output_BR[name].shape))
    #     print("item is {}".format(output[name].mean(-1).mean(-1).mean(-1)))
    #     print("item_BR is {}".format(output_BR[name].mean(-1).mean(-1).mean(-1)))

    # proj is the same
    # output = model.classifier.project(output['low_level'])
    # output_BR = model.classifier.project(output_BR['low_level'])

    # aspp is different
    # output = model.classifier.aspp(output['out'])
    # output_BR = model.classifier.aspp(output_BR['out'])

    # aspp is different
    output = model.classifier.aspp(output['out'])
    output_BR = model.classifier.aspp(output_BR['out'])

    # the bug may lie in the conv with groups > 0
    print("output shape is {}".format(output.shape))
    print("output_BR  shape is {}".format(output_BR.shape))
    print("output is {}".format(output.mean(-1).mean(-1).mean(-1)))
    print("output_BR is {}".format(output_BR.mean(-1).mean(-1).mean(-1)))

    set_trace()


if __name__ == "__main__":
    main()
