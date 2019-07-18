#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import shutil
import os

from scripts.utils.fileUtils.createPathFromStr import createDirFromPathStr

__author__ = 'Lou Zehua'
__time__ = '2018/10/22 18:41'


# move or rename
def movefile(src_path, dest_path):
    if not os.path.isfile(src_path):
        print("%s not exist!" % src_path)
    else:
        fdir, fname = os.path.split(dest_path)
        createDirFromPathStr(fdir)
        shutil.move(src_path, dest_path)


# move or rename files from src_dir: directories of src_dir will be ignored!
def movefiles(src_dir, dest_dir):
    if not os.path.isdir(src_dir):
        print("%s is not a directory!" % src_dir)
    else:
        createDirFromPathStr(dest_dir)
        fileNames = os.listdir(src_dir)
        for fileName in fileNames:
            src_path = os.path.join(src_dir, fileName)
            dest_path = os.path.join(dest_dir, fileName)
            movefile(src_path, dest_path)


# move or rename files from src_dir recursively: dest_dir should not be existed!
def movefiles_recursive(src_dir, dest_dir):
    if not os.path.isdir(src_dir):
        print("%s is not a directory!" % src_dir)
    else:
        shutil.copytree(src_dir, dest_dir, symlinks=False, ignore=None)
        shutil.rmtree(src_dir)
