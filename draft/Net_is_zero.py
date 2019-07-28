#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6

import os

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn

from etc import filePathConf
from scripts.digital_layers import training_purpose
from scripts.digital_layers.Net_step import Net_step
from scripts.structure.Net_template import Net_template

__author__ = 'Lou Zehua'
__time__ = '2019/7/29 1:26'
