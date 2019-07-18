#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import torch.nn as nn

__author__ = 'Lou Zehua'
__time__ = '2019/7/18 11:09'


class Step(nn.Module):
    def __init__(self):
        super(Step, self).__init__()

    def forward(self, x):
        return x > 0
