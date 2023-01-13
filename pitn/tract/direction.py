# -*- coding: utf-8 -*-
import collections
from typing import Tuple

import dipy
import einops
import numpy as np
import torch

import pitn

# Note for theta and phi:
# theta - torch.Tensor of spherical polar (or inclinication) coordinate theta in range
#   [0, $\pi$].
# phi - torch.Tensor of spherical azimuth coordinate phi in range (-\pi$, $\pi$].

_SphereSampleResult = collections.namedtuple("_FodfResult", ("vals", "theta", "phi"))


def fodf_duplicate_hemisphere2sphere(
    fodf: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor
) -> _SphereSampleResult:
    sphere_theta = torch.cat([theta, (theta + torch.pi / 2) % torch.pi], dim=0)
    sphere_phi = torch.cat(
        [
            phi,
            torch.clamp_min(-phi, -torch.pi + torch.finfo(phi.dtype).eps),
        ],
        dim=0,
    )

    # Values are the same, so just duplicate them.
    sphere_sample = torch.cat([fodf, fodf], dim=0)

    return _SphereSampleResult(vals=sphere_sample, theta=sphere_theta, phi=sphere_phi)


def __euclid_dist_spherical_coords(
    r_1, theta_1, phi_1, r_2, theta_2, phi_2
) -> torch.Tensor:
    angle_coeff = torch.sin(theta_1) * torch.sin(theta_2) * torch.cos(
        phi_1 - phi_2
    ) + torch.cos(theta_1) * torch.cos(theta_2)
    d_squared = r_1**2 + r_2**2 - 2 * r_1 * r_2 * angle_coeff
    d = torch.sqrt(d_squared)
    return d


_euclid_dist_spherical_coords = torch.jit.trace(
    __euclid_dist_spherical_coords,
    (
        torch.rand(4),
        torch.linspace(0, torch.pi, 4),
        torch.linspace(-torch.pi + 1e-6, torch.pi, 4),
        torch.rand(4),
        torch.linspace(torch.pi, 0, 4),
        torch.linspace(torch.pi, -torch.pi + 1e-6, 4),
    ),
)


def closest_opposing_direction(
    entry_vec_theta_phi: torch.Tensor,
    fodf_peaks: torch.Tensor,
    peak_coords_theta_phi: torch.Tensor,
    peaks_valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    v = einops.rearrange(entry_vec_theta_phi, "... c -> (...) 1 c", c=2)
    batch_size = v.shape[0]
    theta_entry = v[..., 0]
    phi_entry = v[..., 1]

    p = einops.rearrange(
        peak_coords_theta_phi, "b ... c -> b (...) c", b=batch_size, c=2
    )
    theta_peak = p[..., 0]
    phi_peak = p[..., 1]
    peaks_p = einops.rearrange(fodf_peaks, "b ... -> b (...)")
    peaks_p_mask = einops.rearrange(peaks_valid_mask, "b ... -> b (...)")

    # Radius is always 1, but we must pass a Tensor object instead of an int for the jit
    # tracing.
    peak_dists = _euclid_dist_spherical_coords(
        peaks_p.new_ones(1),
        theta_entry,
        phi_entry,
        peaks_p.new_ones(1),
        theta_peak,
        phi_peak,
    )
    # Only take distances of valid peaks, ignore the "dummy" peaks that are given for
    # the purposes of processing non-jagged tensors.
    peak_dists = peaks_p_mask * peak_dists
    # Maximizing the distance between the entry "south pole" and the outgoing is equal
    # to minimizing the distance when mapping both to the same hemisphere; the great
    # circle from entry to outgoing should be as close to an "equator" as possible.
    closest_dir_idx = torch.argmax(peak_dists, dim=1, keepdim=True)
    closest_peaks = torch.take_along_dim(peaks_p, closest_dir_idx, dim=1)
    closest_dirs = torch.take_along_dim(p, closest_dir_idx[..., None], dim=1)

    closest_peaks = torch.reshape(closest_peaks, entry_vec_theta_phi.shape[:-1])
    closest_dirs = torch.reshape(closest_dirs, entry_vec_theta_phi.shape)

    return closest_dirs, closest_peaks
