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
from network.custom_functions.custom_sync_bn import SparseBatchNorm2d

from torch.nn import BatchNorm2d


def testGradDif(model1, model2):
    max_dif = -1
    max_dif_name = None

    grad_dict_model1 = {k: v.grad for k, v in zip(model1.state_dict(), model1.parameters()) if v.requires_grad}
    grad_dict_model2 = {k: v.grad for k, v in zip(model2.state_dict(), model2.parameters()) if v.requires_grad}

    for name1, grad1 in grad_dict_model1.items():
        if grad1 is not None:
            assert grad_dict_model2[name1] is not None
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


def testSyncBN():
    # masker = None
    masker = Masker(prune_ratio=0.9)

    # 665.1 M
    print("test mesa memory")
    model = nn.Sequential(*[SparseConvTranspose2d(in_channels=256,
                                                  out_channels=256,
                                                  kernel_size=3,
                                                  stride=2,
                                                  masker=masker) for _ in range(2)])
    # mesa.policy.deploy_on_init(model, 'model_mesa/policy_tiny-8bit.txt', verbose=print, override_verbose=False)

    # print("test std memory")
    # model = nn.Sequential(*[nn.Conv2d(in_channels=256,
    #                                   out_channels=256,
    #                                   kernel_size=3,
    #                                   stride=1) for _ in range(10)])

    model.cuda()

    # remove gelu
    for module in model:
        module.act_fn = nn.Identity()

    model = model.cuda()
    input = torch.rand(128, 256, 64, 64).cuda()
    MB = 1024.0 * 1024.0
    print("input usage is {:.1f} MB".format(input.element_size() * input.nelement() / MB))

    mlp_origin_out = model(input)
    mlp_origin_out.sum().backward()

    print("############ mesa mlp #############")
    print("max memory is {:.1f} MB".format(torch.cuda.max_memory_allocated() / MB))

    # activation_bits = 32
    # memory_cost, memory_cost_dict = profile_memory_cost(model, input_size=(1, 196, 384), require_backward=True,
    #                                                     activation_bits=activation_bits, trainable_param_bits=32,
    #                                                     frozen_param_bits=8, batch_size=64)
    # MB = 1024 * 1024
    # print("memory_cost is {:.1f} MB, param size is {:.1f} MB, act_size each sample is {:.1f} MB".
    #       format(memory_cost / MB, memory_cost_dict["param_size"] / MB, memory_cost_dict["act_size"] / MB))


def testBNActivationPrune():
    # configMemTest = ConfigMemoryTest()
    # mlp_origin = SparseConvTranspose2d(96, 96, (3, 3), stride=2, padding=1, masker=None).cuda()
    mlp_origin = BatchNorm2d(96).cuda()
    mlp_origin.train()

    prune_ratio = 0.0
    masker = Masker(prune_ratio=prune_ratio)
    # mlp_our = SparseConvTranspose2d(96, 96, (3, 3), stride=2, padding=1, masker=masker).cuda()
    mlp_our = SparseBatchNorm2d(96, masker=masker).cuda()
    mlp_our.train()
    mlp_our.load_state_dict(mlp_origin.state_dict(), strict=True)
    # set_trace()

    input = torch.rand(64, 96, 64, 64).cuda()
    input.requires_grad = True
    input2 = input.clone().detach()
    input2.requires_grad = True

    mlp_origin_out = mlp_origin(input)
    mlp_origin_out.sum().backward()
    input_grad_ori = input.grad

    # our softmax
    input.grad = torch.zeros_like(input.grad)
    input.requires_grad = True

    # when prune ratio is 0, the two should be equal
    mlp_our_out = mlp_our(input2)
    mlp_our_out.sum().backward()

    input_grad_our = input2.grad

    print("############ prune ratio of {} #############".format(prune_ratio))
    print("activation dist is {}".format(torch.norm(mlp_our_out[0] - mlp_origin_out[0])))
    # it would be good for slightly different, custom softmax layer often introduce some difference
    print("input grad dist is {}".format(torch.norm(input_grad_ori - input_grad_our)))
    testGradDif(mlp_origin, mlp_our)


if __name__ == "__main__":
    testBNActivationPrune()
    # testStdConv()
    # testMlpStoreActivationPrune()

# TODO: run python -m torch.distributed.launch --nproc_per_node=2 test_custom_funcs/test_sync_bn.py
