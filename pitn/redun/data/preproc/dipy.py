# -*- coding: utf-8 -*-
from pathlib import Path

import dipy
import redun
from redun import File, task

import pitn

if __package__ is not None:
    redun_namespace = str(__package__)


@task()
def mppca():
    pass


@task()
def gibbs_removal():
    pass
