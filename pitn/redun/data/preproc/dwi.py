# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import redun
from redun import File, task

import pitn

if __package__ is not None:
    redun_namespace = str(__package__)


@task()
def bvec_flip_correct():
    pass


@task()
def eddy_apply_params():
    pass
