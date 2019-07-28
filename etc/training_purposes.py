#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
from etc import filePathConf

__author__ = 'Lou Zehua'
__time__ = '2019/7/19 15:28'

# Define training_purposes
L_DIGITAL = 0
R_ANALOG = 1
training_purposes = {
    L_DIGITAL: 'L_digital',
    R_ANALOG: 'R_analog',
}

# Define relations between purposes and scripts_dir
project_purposes_scriptsDir = {
    training_purposes[L_DIGITAL]: filePathConf.absPathDict[filePathConf.SCRIPTS_DIGITAL_LAYERS_DIR],
    training_purposes[R_ANALOG]: filePathConf.absPathDict[filePathConf.SCRIPTS_FEATURE_FITTING_LAYERS_DIR],
}
project_scriptsDir_purposes = {v: k for k, v in project_purposes_scriptsDir.items()}
