#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import pandas as pd

__author__ = 'Lou Zehua'
__time__ = '2018/9/25 19:27'


def getParamDict(filepath, sep=':', header='infer'):  # get parameter dictionary from filepath separated by sep
    paramDict = {}
    try:
        locParam = pd.read_csv(filepath, header=header, prefix='', sep=sep)
        for i in range(len(locParam)):
            paramDict[locParam.loc[i][0].strip()] = locParam.loc[i][1].strip()
    except BaseException:
        pass
    return paramDict
