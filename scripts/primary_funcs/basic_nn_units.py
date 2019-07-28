#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import operator
from functools import reduce
import numpy as np
import torch

import torch.nn as nn
from torch.autograd import Variable

__author__ = 'Lou Zehua'
__time__ = '2019/7/25 17:10'


class Step(nn.Module):
    def __init__(self, in_features, out_features=1):
        super(Step, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return (x > 0).float()

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class Add(nn.Module):
    def __init__(self, in_features, out_features=1):
        super(Add, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x_numpy = x.numpy().transpose()
        x = Variable(torch.from_numpy(x_numpy)).float()
        return sum(x).float()

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class Multiply(nn.Module):
    def __init__(self, in_features, out_features=1):
        super(Multiply, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x_numpy = x.numpy().transpose()
        x = Variable(torch.from_numpy(x_numpy)).float()
        return reduce(operator.mul, x, 1)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


if __name__ == '__main__':
    # input
    N = 100
    x_input_array = np.array(torch.rand(N, 3))
    x_input = Variable(torch.from_numpy(x_input_array)).float()
    # output
    net = Multiply(in_features=3, out_features=1)
    output = net(x_input)
    # target
    label = []
    for x in x_input:
        label.append((reduce(operator.mul, x, 1)).numpy())
    y_target = Variable(torch.Tensor(np.array(label))).float()

    # predict test
    y_pred = net.forward(x_input)
    y_pred_array = np.array(y_pred.detach().numpy())
    y_target_array = np.array(y_target.numpy())
    print(sum(y_pred_array == y_target_array))
    print(net.state_dict())
    print(net.__dict__)
