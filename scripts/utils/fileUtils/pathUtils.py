#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import os

from etc.profiles import BASE_DIR

__author__ = 'Lou Zehua'
__time__ = '2018/10/19 13:48'


def createDirFromPathStr(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return None


def createFileFromPathStr(path):
    if not os.path.exists(path):
        createDirFromPathStr('/'.join(path.replace('\\', '/').split('/')[0:-1]))
        file = open(path, 'a', encoding='utf-8')
        file.close()
    return None


def getAbsPath(relative_path, base_dir=None):
    base_dir = base_dir or BASE_DIR
    return os.path.join(base_dir, relative_path)
