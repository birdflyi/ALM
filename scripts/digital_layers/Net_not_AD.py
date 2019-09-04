#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import os

import torch.nn as nn

from etc import filePathConf, extensions

from scripts.feature_fitting_layers.Net_not import Net_not
from scripts.digital_layers.Net_step import Net_step
from scripts.structure.Net_template import Net_template

__author__ = 'ALM'
__time__ = '2019-08-01 12:10:42.263590'

class Net_not_AD(Net_template):
    def __init__(self, in_features=1, out_features=1, class_alias=None):
        super(Net_not_AD, self).__init__(in_features, out_features, class_alias)
        self.net_sequence = nn.Sequential(
            Net_not(in_features=1, out_features=1),
            Net_step(in_features=1, out_features=1)
        )
        self.set_caller_pyfile_path(os.path.abspath(__file__))
        self.check_purpose()
        self.summary()


if __name__ == '__main__':
    net = Net_not_AD()
    # save and load model pairs
    save_dir = filePathConf.absPathDict[filePathConf.MODELS_STATE_DICT_DIR]
    ext_state_dict = extensions.ext_models[extensions.EXT_MODELS__STATE_DICT]
    net.reset_save_model_path(save_dir=save_dir, save_mode=extensions.EXT_MODELS__STATE_DICT)
    model = net.load_state_dict_model(net._save_model_path)
    model.set_caller_pyfile_path(os.path.abspath(__file__))
    model.check_purpose()
    model.summary()
    model.save_whole_model()
    model.save_state_dict_model()
