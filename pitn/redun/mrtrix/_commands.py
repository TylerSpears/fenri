# -*- coding: utf-8 -*-
from pathlib import Path

import redun
from redun import File, task

import pitn

if __package__ is not None:
    redun_namespace = str(__package__)


@task()
def dwi_bias_correct():
    pass
