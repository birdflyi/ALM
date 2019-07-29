#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import os
import re

from etc import extensions
from etc.profiles import BASE_DIR

__author__ = 'Lou Zehua'
__time__ = '2019/7/29 14:12'


def path_Module2File(module_path, file_ext=None, absfilepath=False):
    file_ext = file_ext or extensions.ext_codes[extensions.EXT_CODES__PY]
    file_path = module_path.replace('.', '/') + file_ext
    if absfilepath:
        file_path = os.path.join(BASE_DIR, file_path)
    return file_path


def path_File2Module(file_path, absfilepath=False):
    file_path = file_path.replace('\\', '/')
    if absfilepath:
        base_dir_path = BASE_DIR.replace('\\', '/')
        if not re.match(base_dir_path, file_path):
            raise Warning('Cannot find "{}" with root path "{}", which will cause a fatal error '
                          'in the future.'.format(file_path, base_dir_path))
        module_path = file_path.replace(base_dir_path, '')
    else:
        module_path = file_path
    module_path = module_path.replace(extensions.ext_codes[extensions.EXT_CODES__PY], '').replace('/',  '.').strip('.')
    return module_path
