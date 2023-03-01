# -*- coding: utf-8 -*-
import collections
from typing import Optional, Sequence, Tuple

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


def merge_status(
    base_status: torch.Tensor, *streamline_statuses: Sequence[torch.Tensor]
) -> torch.Tensor:
    statuses = torch.stack(tuple(streamline_statuses), dim=0)
    tmp_status = torch.clone(base_status)
    tmp_status[(statuses == STOP).any(0)] = STOP
    tmp_status[(statuses == INVALID).any(0)] = INVALID

    return tmp_status


def streamline_len_mm(
    streamline_status: torch.Tensor,
    streamline_len: torch.Tensor,
    min_len: torch.Tensor = 0,
    max_len: torch.Tensor = torch.inf,
):
    tmp_status = torch.clone(streamline_status)
    # Streamlines that have been stopped should be within the allowed range of lengths.
    tmp_status = torch.where(
        (streamline_status == STOP) & (streamline_len < min_len),
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
        samples = -gfa_vol.new_ones(1)
    else:
        samples = pitn.affine.sample_3d(
            vol=gfa_vol,
            coords_mm_zyx=coords,
            affine_vox2mm=affine_vox2mm,
            mode="bilinear",
            align_corners=True,
            override_out_of_bounds_val=-1.0,
        )
        samples.squeeze_(-1)
    # Re-expand samples and set stopped/invalid streamlines to a gfa value of -1
    # (always an invalid GFA).
    # samples = torch.masked_scatter()
    null_samples = -torch.ones_like(streamline_status, dtype=samples.dtype)
    null_samples = torch.where(streamline_status == CONTINUE, samples, null_samples)
    # null_samples.masked_scatter_(streamline_status == CONTINUE, samples)
    samples = null_samples
    # samples.masked_fill_(streamline_status != CONTINUE, -1)
    st_new = streamline_status.masked_fill(
        (streamline_status == CONTINUE) & (samples < gfa_min_threshold), STOP
    )
    return st_new


def scalar_vec_threshold(
    streamline_status: torch.Tensor,
    v: torch.Tensor,
    scalar_min_threshold: Optional[float] = None,
    scalar_max_threshold: Optional[float] = None,
) -> torch.Tensor:

    to_stop_mask = torch.zeros_like(streamline_status).bool()
    if scalar_min_threshold is not None:
        to_stop_mask[v < scalar_min_threshold] = True
    if scalar_max_threshold is not None:
        to_stop_mask[v > scalar_max_threshold] = True

    new_status = torch.where(
        (streamline_status == CONTINUE) & to_stop_mask, STOP, streamline_status
    )
    return new_status


def scalar_vol_threshold(
    streamline_status: torch.Tensor,
    sample_coords_mm_zyx: torch.Tensor,
    scalar_min_threshold: float,
    vol: torch.Tensor,
    affine_vox2mm: torch.Tensor,
) -> torch.Tensor:
    # Filter coordinates by only the valid streamlines, to save computation.
    coords = sample_coords_mm_zyx[streamline_status == CONTINUE]
    if coords.numel() == 0:
        samples = -vol.new_ones(1)
    else:
        samples = pitn.affine.sample_3d(
            vol=vol,
            coords_mm_zyx=coords,
            affine_vox2mm=affine_vox2mm,
            mode="bilinear",
            align_corners=True,
            override_out_of_bounds_val=scalar_min_threshold - 1.0,
        )
        samples.squeeze_(-1)
    null_samples = -torch.ones_like(streamline_status, dtype=samples.dtype)
    null_samples = torch.where(streamline_status == CONTINUE, samples, null_samples)
    samples = null_samples
    st_new = streamline_status.masked_fill(
        (streamline_status == CONTINUE) & (samples < scalar_min_threshold), STOP
    )
    return st_new


def angular_threshold(
    streamline_status: torch.Tensor,
    coords_mm_tm1: torch.Tensor,
    coords_mm_t: torch.Tensor,
    max_radians: float,
) -> torch.Tensor:
    cos_theta = F.cosine_similarity(coords_mm_tm1, coords_mm_t, dim=-1)

    theta = torch.arccos(cos_theta)
    new_status = torch.where(
        (streamline_status == CONTINUE) & (theta > max_radians), STOP, streamline_status
    )
    return new_status
