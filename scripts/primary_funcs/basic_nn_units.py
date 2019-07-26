#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import torch.nn as nn

__author__ = 'Lou Zehua'
__time__ = '2019/7/25 17:10'


class Step(nn.Module):
    def __init__(self, in_features, out_features):
        super(Step, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return (x > 0).float()

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
