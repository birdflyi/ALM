#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import os

__author__ = 'Lou Zehua'
__time__ = '2018/9/25 18:33'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------------------------------------------------------
#  Define file_indexes
# ----------------------------------------------------------------------------------------------------------------------
# -----------------------models------------------------
MODELS_STATE_DICT_DIR = 0
MODELS_WHOLE_NET_PARAMS_DIR = 1

# Each directory should be named with a suffix '_DIR'!
absPathDict = {
    MODELS_STATE_DICT_DIR: os.path.join(BASE_DIR, 'models/models_state_dict/'),
    MODELS_WHOLE_NET_PARAMS_DIR: os.path.join(BASE_DIR, 'models/models_whole_net_params/'),
}

fileNameDict = {k: v.replace('\\', '/').split('/')[-1] for k, v in absPathDict.items()}

absDirDict = {k: '/'.join(v.replace('\\', '/').split('/')[:-1]) for k, v in absPathDict.items()}
