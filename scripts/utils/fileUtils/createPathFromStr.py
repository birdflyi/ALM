#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import os

__author__ = 'Lou Zehua'
__time__ = '2018/8/11 15:05'


def createDirFromPathStr(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return None


def createFileFromPathStr(path):
    if not os.path.exists(path):
        createDirFromPathStr('/'.join(path.replace('\\', '/').split('/')[0:-1]))
        file = open(path,'a',encoding='utf-8')
        file.close()
    return None