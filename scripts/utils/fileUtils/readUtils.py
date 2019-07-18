#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
from etc.profiles import encoding

__author__ = 'Lou Zehua'
__time__ = '2018/10/19 13:50'


def readLines(path):
    with open(path, "r", encoding=encoding) as f:
        lines = f.readlines()
    f.close()
    lines = [line.strip() for line in lines]
    return lines
