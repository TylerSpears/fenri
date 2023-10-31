# -*- coding: utf-8 -*-
from typing import Callable, Optional, Tuple

import numpy as np
import torch

import pitn


def gen_tract_step_rk4(
    start_point_xyz: torch.Tensor,
    init_direction: torch.Tensor,
    step_size: float,
    fn_xyz_seed_direction2out_direction: Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ],
) -> torch.Tensor:

    # Technically, the seed direction comes *into* the coordinate's ODF, then *exits* on
    # the opposite side of the sphere. However, this is just two antipodal flips, which
    # cancel out. So, the "in direction" is just (based on) the "out direction".
    k1_xyz_tangent = fn_xyz_seed_direction2out_direction(
        start_point_xyz, init_direction
    )

    k2_xyz_tangent = fn_xyz_seed_direction2out_direction(
        start_point_xyz + (step_size / 2 * k1_xyz_tangent),
        k1_xyz_tangent,
    )

    k3_xyz_tangent = fn_xyz_seed_direction2out_direction(
        start_point_xyz + (step_size / 2 * k2_xyz_tangent),
        k2_xyz_tangent,
    )

    k4_xyz_tangent = fn_xyz_seed_direction2out_direction(
        start_point_xyz + (step_size * k3_xyz_tangent),
        k3_xyz_tangent,
    )

    weighted_summed_tangent = (
        k1_xyz_tangent + (2 * k2_xyz_tangent) + (2 * k3_xyz_tangent) + k4_xyz_tangent
    )
    # Scaling the vectors to unit norm substitutes as the division by 6.
    tangent_tp1 = (
        step_size
        * weighted_summed_tangent
        / torch.linalg.vector_norm(weighted_summed_tangent, ord=2, dim=-1, keepdim=True)
    )

    # Return the tangent vector rather than point_{t+1}, for ease of passing this
    # tangent as a new tangent_{t}.
    return tangent_tp1
