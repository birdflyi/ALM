#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import shutil
import os

from scripts.utils.fileUtils.pathUtils import createDirFromPathStr

__author__ = 'Lou Zehua'
__time__ = '2018/10/22 18:47'


# copy
def copyfile(src_path, dest_path):
    if not os.path.isfile(src_path):
        print("%s not exist!" % src_path)
    else:
        fdir, fname = os.path.split(dest_path)
        createDirFromPathStr(fdir)
        shutil.copyfile(src_path, dest_path)


def copyAllFilesInDir(src_dir, dest_dir):
    if not os.path.isdir(src_dir):
        print("%s not exist!" % src_dir)
    else:
        createDirFromPathStr(dest_dir)
        fileNames = os.listdir(src_dir)
        for fileName in fileNames:
            src_path = os.path.join(src_dir, fileName)
            dest_path = os.path.join(dest_dir, fileName)
            shutil.copyfile(src_path, dest_path)
