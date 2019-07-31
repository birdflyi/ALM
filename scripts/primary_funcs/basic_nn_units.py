#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import operator
import warnings
from functools import reduce
import numpy as np
import torch

import torch.nn as nn
from torch.autograd import Variable

__author__ = 'Lou Zehua'
__time__ = '2019/7/25 17:10'


class Pass(nn.Module):
    def __init__(self, in_features, out_features):
        super(Pass, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.validate_shape()

    def forward(self, x):
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

    def validate_shape(self):
        if not self.in_features == self.out_features:
            self.out_features = self.in_features
            warnings.warn('"in_features" is different from "out_features". The value of "out_features" will be '
                          'reset to "in_features" value.')


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
        x_shape = x.shape
        x = torch.transpose(x, -1, 0)
        out = sum(x)
        out = Variable(out).float()
        out = torch.transpose(out, -1, 0).view(x_shape[0], -1)
        return out

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
        x_shape = x.shape
        x = torch.transpose(x, -1, 0)
        out = reduce(operator.mul, x, 1)
        out = Variable(out).float()
        out = torch.transpose(out, -1, 0).view(x_shape[0], -1)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


if __name__ == '__main__':
    # input
    N = 100
    x_input_array = np.array(torch.rand(N, 3) - 0.5)
    x_input = Variable(torch.from_numpy(x_input_array)).float()
    # output
    net = Multiply(in_features=3, out_features=1)
    output = net(x_input)
    # target
    label = []
    for x in x_input:
        # label.append((x > 0).numpy())
        # label.append((sum(x)).numpy())
        label.append((reduce(operator.mul, x, 1)).numpy())
    y_target = Variable(torch.Tensor(np.array(label))).float()
    y_target = y_target.view(x_input.shape[0], -1)

    # predict test
    y_pred = net.forward(x_input)
    y_pred_array = np.array(y_pred.detach().numpy())
    y_target_array = np.array(y_target.numpy())
    print(sum(y_pred_array == y_target_array))
    print(net.state_dict())
    print(net.__dict__)
