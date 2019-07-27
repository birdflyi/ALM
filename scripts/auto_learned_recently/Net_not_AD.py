#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import torch.nn as nn

from scripts.feature_fitting_layers.Net_not import Net_not
from scripts.digital_layers.Net_step import Net_step
from scripts.structure.Net_template import Net_template

__author__ = 'ALM'
__time__ = '2019-07-28 03:39:24.200417'

class Net_not_AD(Net_template):
    def __init__(self, in_features=1, out_features=1, class_alias=None):
        super(Net_not_AD, self).__init__(in_features, out_features, class_alias)
        self.net_sequence = nn.Sequential(
            Net_not(in_features=1, out_features=1),Net_step(in_features=1, out_features=1)
        )
        self.check_purpose()
        self.summary()


if __name__ == '__main__':
    net = Net_not_AD()
    # save and load model pairs
    net.save_whole_model()
    model = net.load_whole_model()
    net.save_state_dict_model()
    model = net.load_state_dict_model()
    # print messages of net
    print(model.class_alias)
    print(model.__dict__)
