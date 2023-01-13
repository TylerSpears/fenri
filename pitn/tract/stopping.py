# -*- coding: utf-8 -*-
import collections
from typing import Tuple

import dipy
import einops
import numpy as np
import torch
import torch.nn.functional as F

import pitn

CONTINUE = 1
STOP = 0
VALID = 2
INVALID = -1


def gfa_threshold(
    sample_coords_mm_zyx: torch.Tensor,
    gfa_min_threshold: float,
    gfa_vol: torch.Tensor,
    affine_vox2mm: torch.Tensor,
) -> torch.Tensor:
    samples = pitn.affine.sample_3d(
        gfa_vol,
        sample_coords_mm_zyx,
        affine_vox2mm=affine_vox2mm,
        mode="nearest",
        align_corners=True,
    )

    return torch.where(samples >= gfa_min_threshold, CONTINUE, STOP).to(torch.int8)


def angular_threshold(
    coords_mm_tm1: torch.Tensor, coords_mm_t: torch.Tensor, max_radians: float
) -> torch.Tensor:
    cos_theta = F.cosine_similarity(coords_mm_tm1, coords_mm_t, dim=-1)

    angle = torch.arccos(cos_theta)
    return torch.where(angle <= max_radians, CONTINUE, STOP).to(torch.int8)
