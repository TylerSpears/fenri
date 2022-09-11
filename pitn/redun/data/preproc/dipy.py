# -*- coding: utf-8 -*-
import dipy
import dipy.denoise
import dipy.denoise.gibbs
import dipy.denoise.localpca
import numpy as np
from redun import task

import pitn

if __package__ is not None:
    redun_namespace = str(__package__)


@task(hash_includes=[dipy.denoise.localpca.mppca])
def mppca(**kwargs):
    return_sigma = kwargs["return_sigma"]
    nout = 2 if return_sigma else None
    f = task(dipy.denoise.localpca.mppca, nout=nout)

    return f(**kwargs)


@task(hash_includes=[dipy.denoise.gibbs.gibbs_removal], config_args=["num_processes"])
def gibbs_removal(
    vol: np.ndarray, slice_axis: int = 2, n_points: int = 3, num_processes: int = 1
) -> np.ndarray:

    # Never make a task operate in-place!
    return dipy.denoise.gibbs.gibbs_removal(
        vol=vol,
        slice_axis=slice_axis,
        n_points=n_points,
        num_processes=num_processes,
        inplace=False,
    )
