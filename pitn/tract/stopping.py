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


def to_continue_mask(streamline_status: torch.Tensor) -> torch.Tensor:
    return streamline_status == CONTINUE


def streamline_len_mm(
    streamline_status: torch.Tensor,
    streamline_len: torch.Tensor,
    min_len: torch.Tensor = 0,
    max_len: torch.Tensor = torch.inf,
):
    tmp_status = torch.clone(streamline_status)
    # Streamlines that have been stopped should be within the allowed range of lengths.
    tmp_status = torch.where(
        (streamline_status == STOP)
        & ((streamline_len < min_len) | (streamline_len > max_len)),
        INVALID,
        tmp_status,
    )

    # Continuing streamlines must have total length < the threshold, but may be below
    # the minimum while they are still being estimated.
    tmp_status = torch.where(
        (streamline_status == CONTINUE) & (streamline_len > max_len), STOP, tmp_status
    )

    return tmp_status


def gfa_threshold(
    streamline_status: torch.Tensor,
    sample_coords_mm_zyx: torch.Tensor,
    gfa_min_threshold: float,
    gfa_vol: torch.Tensor,
    affine_vox2mm: torch.Tensor,
) -> torch.Tensor:
    # Filter coordinates by only the valid streamlines, to save computation.
    coords = sample_coords_mm_zyx[streamline_status == CONTINUE]
    if coords.numel() == 0:
        samples = torch.tensor(-1)
    else:
        samples = pitn.affine.sample_3d(
            vol=gfa_vol,
            coords_mm_zyx=coords,
            affine_vox2mm=affine_vox2mm,
            mode="nearest",
            align_corners=True,
        )
        samples.squeeze_(-1)
    # Re-expand samples and set stopped/invalid streamlines to a gfa value of -1
    # (always an invalid GFA).
    samples = torch.where(streamline_status == CONTINUE, samples, -1)

    return torch.where(
        (streamline_status == CONTINUE) & (samples < gfa_min_threshold),
        STOP,
        streamline_status,
    ).to(torch.int8)


def angular_threshold(
    coords_mm_tm1: torch.Tensor, coords_mm_t: torch.Tensor, max_radians: float
) -> torch.Tensor:
    cos_theta = F.cosine_similarity(coords_mm_tm1, coords_mm_t, dim=-1)

    angle = torch.arccos(cos_theta)
    return torch.where(angle <= max_radians, CONTINUE, STOP).to(torch.int8)
