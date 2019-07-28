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

# ---------------------net template---------------------
PY_NET_TEMPLATE_CODE_DIR = 10
PY_NET_TEMPLATE_PATH = 11

# -----------------scripts related dir------------------
SCRIPTS_ROOT_DIR = 20
SCRIPTS_FEATURE_FITTING_LAYERS_DIR = 21
SCRIPTS_DIGITAL_LAYERS_DIR = 22

# Each directory should be named with a suffix '_DIR'!
absPathDict = {
    MODELS_STATE_DICT_DIR: os.path.join(BASE_DIR, 'models/models_state_dict/'),
    MODELS_WHOLE_NET_PARAMS_DIR: os.path.join(BASE_DIR, 'models/models_whole_net_params/'),
    PY_NET_TEMPLATE_CODE_DIR: os.path.join(BASE_DIR, 'scripts/auto_learned_recently/'),
    PY_NET_TEMPLATE_PATH: os.path.join(BASE_DIR, 'etc/py_net_template.pytmpl'),
    SCRIPTS_ROOT_DIR: os.path.join(BASE_DIR, 'scripts/'),
    SCRIPTS_FEATURE_FITTING_LAYERS_DIR: os.path.join(BASE_DIR, 'scripts/feature_fitting_layers/'),
    SCRIPTS_DIGITAL_LAYERS_DIR: os.path.join(BASE_DIR, 'scripts/digital_layers/'),
}

fileNameDict = {k: v.replace('\\', '/').split('/')[-1] for k, v in absPathDict.items()}

absDirDict = {k: '/'.join(v.replace('\\', '/').split('/')[:-1]) for k, v in absPathDict.items()}
