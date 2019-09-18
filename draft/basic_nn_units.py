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

class Polynomial_ElemWise(nn.Module):
    def __init__(self, in_features, out_features=None, steps=None):
        super(Polynomial_ElemWise, self).__init__()
        self.in_features = in_features
        self.out_features = out_features or self.in_features
        self.steps = steps or 1
        self.check_steps()

    def check_steps(self):
        if not isinstance(self.steps, int) or self.steps < 0:
            raise Exception('Parameter of buil_sequence error! Please set it to a nonnegative integer.')

    def forward(self, x):
        out = torch.ones(x.shape).float()
        for i in range(self.steps):
            out = np.multiply(out, x)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, steps={}'.format(
            self.in_features, self.out_features, self.steps
        )


if __name__ == '__main__':
    # input
    N = 100
    x_input_array = np.array(torch.rand(N, 3) - 0.5)
    x_input = Variable(torch.from_numpy(x_input_array)).float()
    # output
    net = Polynomial_ElemWise(in_features=3)
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
