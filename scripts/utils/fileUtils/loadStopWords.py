#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
from etc import filePathConf as fpc
from etc.profiles import encoding

__author__ = 'Lou Zehua'
__time__ = '2018/10/19 16:17'


def loadStopWords(stop_words_path=None):
    stopwords = []
    stop_words_path = stop_words_path or fpc.absPathDict[fpc.MANUAL_STOP_WORDS_PATH]
    with open(stop_words_path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    f.close()
    return stopwords
