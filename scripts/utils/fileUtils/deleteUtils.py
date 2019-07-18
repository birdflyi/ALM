#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import os
import shutil

from etc import filePathConf
from scripts.utils.fileUtils.pathUtils import createDirFromPathStr

__author__ = 'Lou Zehua'
__time__ = '2018/10/19 13:50'


def clearFilesFromDir(dir):
    if not (os.path.isdir(dir) and os.path.exists(dir)):
        return False
    shutil.rmtree(dir)
    createDirFromPathStr(dir)
    return True


if __name__ == '__main__':
    dir = filePathConf.absPathDict[filePathConf.SENTSEG_DIR]
    print(clearFilesFromDir(dir))
