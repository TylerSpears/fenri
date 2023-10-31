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
    gfa_min_threshold: float,
    sh_coeff: torch.Tensor,
) -> torch.Tensor:
    gen_fa = pitn.odf.gfa(sh_coeff)
    st_new = streamline_status.masked_fill(
        (streamline_status == CONTINUE) & (gen_fa < gfa_min_threshold), STOP
    )
    return st_new


def vol_sample_threshold(
    streamline_status: torch.Tensor,
    vol: torch.Tensor,
    affine_vox2real: torch.Tensor,
    sample_coords: torch.Tensor,
    sample_min: float = -torch.inf,
    sample_max: float = torch.inf,
    **sample_vol_kwargs,
) -> torch.Tensor:

    c = einops.rearrange(sample_coords, "b coord -> 1 b 1 1 coord")
    affine = affine_vox2real.unsqueeze(0)
    v = vol
    if v.ndim == 3:
        v = v.unsqueeze(0)
    if v.ndim == 4:
        v = v.unsqueeze(0)

    samples = pitn.affine.sample_vol(
        v, coords_mm_xyz=c, affine_vox2mm=affine, **sample_vol_kwargs
    )
    samples = samples.flatten()
    st_new = torch.where(
        (streamline_status == CONTINUE)
        & ((samples < sample_min) | (samples > sample_max)),
        streamline_status.new_ones(1) * STOP,
        streamline_status,
    )
    return st_new


def val_threshold(
    streamline_status: torch.Tensor,
    val: torch.Tensor,
    val_min_thresh: float = -torch.inf,
    val_max_thresh: float = torch.inf,
) -> torch.Tensor:
    val_out_of_bounds = (val < val_min_thresh) | (val > val_max_thresh) | val.isnan()
    st_new = streamline_status.masked_fill(
        (streamline_status == CONTINUE) & val_out_of_bounds, STOP
    )
    return st_new


def angular_threshold(
    streamline_status: torch.Tensor,
    angle_x: torch.Tensor,
    angle_y: torch.Tensor,
    max_radians: float,
) -> torch.Tensor:
    cos_sim = F.cosine_similarity(angle_x, angle_y, dim=-1)
    cos_sim.clamp_(min=pitn.tract.MIN_COS_SIM, max=pitn.tract.MAX_COS_SIM)
    arc_len = torch.arccos(cos_sim)
    arc_len.masked_fill_(
        torch.isclose(cos_sim, cos_sim.new_tensor([pitn.tract.MAX_COS_SIM])), 0.0
    )

    new_status = torch.where(
        (streamline_status == CONTINUE) & (arc_len > max_radians),
        STOP,
        streamline_status,
    )
    return new_status
