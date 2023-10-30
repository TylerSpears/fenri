# -*- coding: utf-8 -*-
from typing import Callable, Optional, Tuple

import numpy as np
import torch

import pitn


def gen_tract_step_rk4(
    start_point_xyz: torch.Tensor,
    step_size: float,
    fn_xyz_direction_t2theta_phi: Callable[
        [torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]
    ],
    init_direction_theta_phi: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    k1 = fn_xyz_direction_t2theta_phi(start_point_xyz, init_direction_theta_phi)
    k1_theta, k1_phi = k1
    k1_xyz_tangent = pitn.tract.unit_sphere2xyz(k1_theta, k1_phi)

    k2 = fn_xyz_direction_t2theta_phi(
        start_point_xyz + (step_size / 2 * k1_xyz_tangent),
        torch.stack([k1_theta, k1_phi], -1),
    )
    k2_theta, k2_phi = k2
    k2_xyz_tangent = pitn.tract.unit_sphere2xyz(k2_theta, k2_phi)

    k3 = fn_xyz_direction_t2theta_phi(
        start_point_xyz + (step_size / 2 * k2_xyz_tangent),
        torch.stack([k2_theta, k2_phi], -1),
    )
    k3_theta, k3_phi = k3
    k3_xyz_tangent = pitn.tract.unit_sphere2xyz(k3_theta, k3_phi)

    k4 = fn_xyz_direction_t2theta_phi(
        start_point_xyz + (step_size * k3_xyz_tangent),
        torch.stack([k3_theta, k3_phi], -1),
    )
    k4_theta, k4_phi = k4
    k4_xyz_tangent = pitn.tract.unit_sphere2xyz(k4_theta, k4_phi)

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
