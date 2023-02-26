# -*- coding: utf-8 -*-
from typing import Callable, Optional, Tuple

import torch


def __unit_sphere2zyx(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    #! Inputs to this function should generally be 64-bit floats! Precision is poor for
    # 32-bit floats.
    # r = 1 on the unit sphere.
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    return torch.stack([z, y, x], dim=-1)


_unit_sphere2zyx = torch.jit.trace(
    __unit_sphere2zyx,
    example_inputs=(
        torch.linspace(0, torch.pi, 10, dtype=torch.float64),
        torch.linspace(-torch.pi + 1e-6, torch.pi, 10, dtype=torch.float64),
    ),
)

unit_sphere2zyx = _unit_sphere2zyx


def _zyx2unit_sphere_theta_phi(
    coords_zyx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    #! Inputs to this function should generally be 64-bit floats! Precision is poor for
    # 32-bit floats.
    z = coords_zyx[..., 0]
    y = coords_zyx[..., 1]
    x = coords_zyx[..., 2]
    theta = torch.arccos(z)
    phi = torch.arctan2(y, x)

    return (theta, phi)


zyx2unit_sphere_theta_phi = torch.jit.trace(
    _zyx2unit_sphere_theta_phi,
    example_inputs=_unit_sphere2zyx(
        torch.linspace(0, torch.pi, 10, dtype=torch.float64),
        torch.linspace(-torch.pi + 1e-6, torch.pi, 10, dtype=torch.float64),
    ),
)


def gen_tract_step_rk4(
    start_point_zyx: torch.Tensor,
    step_size: float,
    fn_zyx_direction_t2theta_phi: Callable[
        [torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]
    ],
    init_direction_theta_phi: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    # Allow for a different initial direction vector than that found by the derivative
    # function.
    if (init_direction_theta_phi is None) or (
        torch.as_tensor(init_direction_theta_phi) == 0
    ).all():
        k1 = fn_zyx_direction_t2theta_phi(start_point_zyx, init_direction_theta_phi)
    else:
        k1 = (init_direction_theta_phi[..., 0], init_direction_theta_phi[..., 1])
    k1_theta, k1_phi = k1
    k1_zyx_tangent = _unit_sphere2zyx(k1_theta, k1_phi)

    k2 = fn_zyx_direction_t2theta_phi(
        start_point_zyx + step_size / 2 * k1_zyx_tangent,
        torch.stack([k1_theta, k1_phi], -1),
    )
    k2_theta, k2_phi = k2
    k2_zyx_tangent = _unit_sphere2zyx(k2_theta, k2_phi)

    k3 = fn_zyx_direction_t2theta_phi(
        start_point_zyx + step_size / 2 * k2_zyx_tangent,
        torch.stack([k2_theta, k2_phi], -1),
    )
    k3_theta, k3_phi = k3
    k3_zyx_tangent = _unit_sphere2zyx(k3_theta, k3_phi)

    k4 = fn_zyx_direction_t2theta_phi(
        start_point_zyx + step_size * k3_zyx_tangent,
        torch.stack([k3_theta, k3_phi], -1),
    )
    k4_theta, k4_phi = k4
    k4_zyx_tangent = _unit_sphere2zyx(k4_theta, k4_phi)

    weighted_summed_tangent = (
        k1_zyx_tangent + 2 * k2_zyx_tangent + 2 * k3_zyx_tangent + k4_zyx_tangent
    )
    # Scaling the vectors to unit norm substitutes as the division by 6.
    tangent_tp1 = (
        step_size
        * weighted_summed_tangent
        / torch.linalg.vector_norm(weighted_summed_tangent, ord=2, dim=-1, keepdim=True)
    )

    # tangent_tp1 = (step_size / 6) * (
    #     k1_zyx_tangent + 2 * k2_zyx_tangent + 2 * k3_zyx_tangent + k4_zyx_tangent
    # )
    # Return the tangent vector rather than point_{t+1}, for ease of passing this
    # tangent as a new tangent_{t}.
    return tangent_tp1
