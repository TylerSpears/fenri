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
    coords_real_tm1: torch.Tensor,
    coords_real_t: torch.Tensor,
    max_radians: float,
) -> torch.Tensor:
    cos_sim = F.cosine_similarity(coords_real_tm1, coords_real_t, dim=-1)
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
