# -*- coding: utf-8 -*-
import collections
from typing import Optional, Tuple

import dipy
import einops
import numpy as np
import torch

import pitn

# Note for theta and phi:
# theta - torch.Tensor of spherical polar (or inclinication) coordinate theta in range
#   [0, $\pi$].
# phi - torch.Tensor of spherical azimuth coordinate phi in range (-\pi$, $\pi$].

_SphereSampleResult = collections.namedtuple("_FodfResult", ("theta", "phi", "vals"))


def wrap_bound_modulo(
    x: torch.Tensor, low: torch.Tensor, high: torch.Tensor
) -> torch.Tensor:
    # Totally stolen from <https://stackoverflow.com/a/22367889>
    d = high - low
    return ((x - low) % d) + low


def project_sph_coord_opposite_hemisphere(
    theta: torch.Tensor, phi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    theta_prime = torch.pi - theta
    open_interval_phi_low = -torch.pi + torch.finfo(phi.dtype).eps
    phi_prime = wrap_bound_modulo(phi + torch.pi, open_interval_phi_low, torch.pi)

    return theta_prime, phi_prime


def fodf_duplicate_hemisphere2sphere(
    theta: torch.Tensor,
    phi: torch.Tensor,
    sphere_fn_vals: Tuple[torch.Tensor, ...] = tuple(),
    fn_vals_concat_dim: Tuple[int, ...] = tuple(),
) -> _SphereSampleResult:
    theta_prime, phi_prime = project_sph_coord_opposite_hemisphere(theta, phi)
    sphere_theta = torch.cat([theta, theta_prime], dim=-1)
    sphere_phi = torch.cat(
        [
            phi,
            phi_prime,
        ],
        dim=-1,
    )

    if isinstance(sphere_fn_vals, (tuple, list)) and (len(sphere_fn_vals) > 0):
        if isinstance(fn_vals_concat_dim, int):
            fn_vals_concat_dim = (fn_vals_concat_dim,) * len(sphere_fn_vals)
        sphere_sample = list()
        for fn, concat_dim in zip(sphere_fn_vals, fn_vals_concat_dim):
            sphere_sample.append(torch.cat([fn, fn], dim=concat_dim))
        sphere_sample = tuple(sphere_sample)
    else:
        sphere_sample = None

    return _SphereSampleResult(theta=sphere_theta, phi=sphere_phi, vals=sphere_sample)


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

    sph_coord = einops.rearrange(
        peak_coords_theta_phi, "b ... c -> b (...) c", b=batch_size, c=2
    )
    theta_peak = sph_coord[..., 0]
    phi_peak = sph_coord[..., 1]

    peaks_p = einops.rearrange(fodf_peaks, "b ... -> b (...)")
    peaks_p_mask = einops.rearrange(peaks_valid_mask, "b ... -> b (...)")
    full_sphere = fodf_duplicate_hemisphere2sphere(
        theta_peak, phi_peak, (peaks_p, peaks_p_mask), fn_vals_concat_dim=(1, 1)
    )
    theta_peak = full_sphere.theta
    phi_peak = full_sphere.phi
    peaks_p = full_sphere.vals[0]
    peaks_p_mask = full_sphere.vals[1]

    theta_entry = theta_entry.expand_as(theta_peak)
    phi_entry = phi_entry.expand_as(phi_peak)
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
    peak_dists.masked_fill_(~peaks_p_mask, -torch.inf)
    # peak_dists = peak_dists * peaks_p_mask
    # peak_dists.masked_fill_(~peak_dists.nonzero(), -torch.inf)
    # Maximizing the distance between the entry "south pole" and the outgoing is equal
    # to minimizing the distance when mapping both to the same hemisphere; the great
    # circle from entry to outgoing should be as close to an "equator" as possible.
    closest_dir_idx = torch.argmax(peak_dists, dim=1, keepdim=True)

    opposing_peaks = torch.take_along_dim(peak_dists, closest_dir_idx, dim=1)
    opposing_theta = torch.take_along_dim(theta_peak, closest_dir_idx, dim=1)
    opposing_phi = torch.take_along_dim(phi_peak, closest_dir_idx, dim=1)

    # If there are no peaks to be found, replace the coordinates with NaNs.
    opposing_theta.masked_fill_(opposing_peaks.isneginf(), torch.nan)
    opposing_phi.masked_fill_(opposing_peaks.isneginf(), torch.nan)

    opposing_theta = torch.reshape(opposing_theta, entry_vec_theta_phi.shape[:-1])
    opposing_phi = torch.reshape(opposing_phi, entry_vec_theta_phi.shape[:-1])

    return opposing_theta, opposing_phi
