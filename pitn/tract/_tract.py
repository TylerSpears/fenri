# -*- coding: utf-8 -*-
import collections
from typing import Callable, Tuple

import einops
import numpy as np
import torch

import pitn
import pitn.tract.peak

_PointContainer = collections.namedtuple("_PointContainer", ("origin", "theta", "phi"))


def seeds_from_peaks(
    max_peaks_per_voxel: int,
    seed_coords_mm: torch.Tensor,
    peaks: torch.Tensor,
    theta_peak: torch.Tensor,
    phi_peak: torch.Tensor,
    valid_peak_mask: torch.Tensor,
) -> _PointContainer:

    topk = pitn.tract.peak.topk_peaks(
        max_peaks_per_voxel,
        peaks,
        theta_peak=theta_peak,
        phi_peak=phi_peak,
        valid_peak_mask=valid_peak_mask,
    )
    topk_valid = topk.valid_peak_mask
    seed_coord = einops.rearrange(seed_coords_mm, "... 3 -> (...) 3")
    k = topk.peaks.shape[-1]

    # Only keep the directions with valid peaks.
    seed_coord = einops.repeat(seed_coord, "b 3 -> b k 3", k=k)
    seed_coord = torch.masked_select(seed_coord, topk_valid[..., None])
    seed_theta = topk.theta[topk_valid]
    seed_phi = topk.phi[topk_valid]

    # Additionally, invert each peak to account for bi-polar symmetry.
    seed_theta = torch.cat([seed_theta, (seed_theta + torch.pi / 2) % torch.pi], dim=0)
    # Phi's range does not include -pi, so if phi is exactly pi, it must be rounded
    # the the next lowest value.
    seed_phi = torch.cat(
        [
            seed_phi,
            torch.clamp_min(-seed_phi, -torch.pi + torch.finfo(seed_phi.dtype).eps),
        ],
        dim=0,
    )
    seed_coord = torch.cat([seed_coord, seed_coord], dim=0)

    return _PointContainer(origin=seed_coord, theta=seed_theta, phi=seed_phi)


def __unit_sphere2zyx(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    # Rescale phi from (-pi, pi] to (0, 2*pi]
    phi = phi + torch.pi
    # r = 1 on the unit sphere.
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    return torch.stack([z, y, x], dim=-1)


_unit_sphere2zyx = torch.jit.trace(
    __unit_sphere2zyx,
    example_inputs=(
        torch.linspace(0, torch.pi, 10),
        torch.linspace(-torch.pi + 1e-6, torch.pi, 10),
    ),
)


def gen_tract_step_rk4(
    start_point_zyx: torch.Tensor,
    step_size: float,
    fn_zyx2theta_phi: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:

    k1 = fn_zyx2theta_phi(start_point_zyx)
    k1_theta, k1_phi = k1
    k1_zyx = _unit_sphere2zyx(k1_theta, k1_phi)

    k2 = fn_zyx2theta_phi(start_point_zyx + step_size / 2 * k1_zyx)
    k2_theta, k2_phi = k2
    k2_zyx = _unit_sphere2zyx(k2_theta, k2_phi)

    k3 = fn_zyx2theta_phi(start_point_zyx + step_size / 2 * k2_zyx)
    k3_theta, k3_phi = k3
    k3_zyx = _unit_sphere2zyx(k3_theta, k3_phi)

    k4 = fn_zyx2theta_phi(start_point_zyx + step_size * k3_zyx)
    k4_theta, k4_phi = k4
    k4_zyx = _unit_sphere2zyx(k4_theta, k4_phi)

    p_1 = step_size / 6 * (k1_zyx + 2 * k2_zyx + 2 * k3_zyx + k4_zyx)

    return p_1
