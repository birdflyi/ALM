#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import pandas as pd

from etc import filePathConf
from etc.model_registry_settings import df_registry_col_names
from etc.profiles import encoding

__author__ = 'Lou Zehua'
__time__ = '2019/7/18 11:09'

model_registry_path = filePathConf.absPathDict[filePathConf.MODEL_REGISTRY_PATH]
df_registry = pd.read_csv(model_registry_path, delimiter=',', names=df_registry_col_names, encoding=encoding)
