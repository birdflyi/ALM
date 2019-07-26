#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6

__author__ = 'Lou Zehua'
__time__ = '2019/7/26 11:16'


# Define extensions of files
# -------------------------------------------------extensions of models-------------------------------------------------
EXT_MODELS__STATE_DICT = 0
EXT_MODELS__WHOLE_NET_PARAMS = 1
ext_models = {
    EXT_MODELS__STATE_DICT: '.state_dict',
    EXT_MODELS__WHOLE_NET_PARAMS: '.model',
}
