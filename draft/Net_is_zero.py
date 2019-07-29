#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6

import os

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn

from etc import filePathConf
from scripts.digital_layers import training_purpose
from scripts.digital_layers.Net_step import Net_step
from scripts.feature_fitting_layers.Net_and import Net_and
from scripts.structure.Net_template import Net_template

__author__ = 'Lou Zehua'
__time__ = '2019/7/29 1:26'


class Net_is_zero(Net_template):
    def __init__(self, in_features=1, out_features=1, class_alias=None):
        super(Net_is_zero, self).__init__(in_features, out_features, class_alias)
        self.net_sequence = nn.Sequential(
            Net_and(in_features=2, out_features=1),Net_step(in_features=1, out_features=1)
        )
        self.set_caller_pyfile_path(os.path.abspath(__file__))
        self.check_purpose()
        self.summary()


if __name__ == '__main__':
    net = Net_is_zero()
    # save and load model pairs
    net.save_whole_model()
    model = net.load_whole_model()
    net.save_state_dict_model()
    model = net.load_state_dict_model()
    # print messages of net
    print(model.class_alias)
    print(model.__dict__)
