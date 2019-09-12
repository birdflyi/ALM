#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import os

import torch.nn as nn

from etc import extensions

from scripts.feature_fitting_layers.Net_and import Net_and
from scripts.digital_layers.Net_step import Net_step
from scripts.structure.Net_template import Net_template

__author__ = 'ALM'
__time__ = '2019-09-12 16:02:33.576418'

class Net_and_AD(Net_template):
    def __init__(self, in_features=2, out_features=1, class_alias=None):
        super(Net_and_AD, self).__init__(in_features, out_features, class_alias)
        self.net_sequence = nn.Sequential(
            Net_and(in_features=2, out_features=1),
            Net_step(in_features=1, out_features=1)
        )
        self.set_caller_pyfile_path(os.path.abspath(__file__))
        self.check_purpose()
        self.summary()


if __name__ == '__main__':
    net = Net_and_AD()
    # load, save and register model
    net.set_save_mode(extensions.EXT_MODELS__STATE_DICT)
    model = net.load_model()
    model.set_caller_pyfile_path(os.path.abspath(__file__))
    model.check_purpose()
    model.summary()
    for temp_save_mode in extensions.ext_models.keys():
        model.set_save_mode(temp_save_mode)
        model.save_model()
    model.register_net()
