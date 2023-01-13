# -*- coding: utf-8 -*-
__version__ = "0.2.0"
from ._lazy_loader import LazyLoader

# io = LazyLoader("io", globals(), "pitn.io")
# samplers = LazyLoader("samplers", globals(), "pitn.samplers")
# transforms = LazyLoader("transforms", globals(), "pitn.transforms")
# viz = LazyLoader("viz", globals(), "pitn.viz")
nn = LazyLoader("nn", globals(), "pitn.nn")
data = LazyLoader("data", globals(), "pitn.data")
# utils = LazyLoader("utils", globals(), "pitn.utils")
coords = LazyLoader("coords", globals(), "pitn.coords")
# metrics = LazyLoader("metrics", globals(), "pitn.metrics")
# eig = LazyLoader("eig", globals(), "pitn.eig")
riemann = LazyLoader("riemann", globals(), "pitn.riemann")
from . import (  # coords,; data,; nn,; riemann,
    affine,
    dsi_studio,
    eig,
    fsl,
    io,
    metrics,
    mrtrix,
    odf,
    samplers,
    tract,
    transforms,
    utils,
    viz,
)

# fsl = LazyLoader("fsl", globals(), "pitn.fsl")
# dsi_studio = LazyLoader("dsi_studio", globals(), "pitn.dsi_studio")
# mrtrix = LazyLoader("mrtrix", globals(), "pitn.mrtrix")
