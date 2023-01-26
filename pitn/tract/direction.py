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


def fodf_duplicate_hemisphere2sphere(
    theta: torch.Tensor,
    phi: torch.Tensor,
    sphere_fn_vals: Tuple[torch.Tensor, ...] = tuple(),
    fn_vals_concat_dim: Tuple[int, ...] = tuple(),
) -> _SphereSampleResult:
    sphere_theta = torch.cat([theta, torch.pi - theta], dim=-1)
    open_interval_phi_low = -torch.pi + torch.finfo(phi.dtype).eps
    sphere_phi = torch.cat(
        [
            phi,
            wrap_bound_modulo(phi + torch.pi, open_interval_phi_low, torch.pi),
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

    p = einops.rearrange(
        peak_coords_theta_phi, "b ... c -> b (...) c", b=batch_size, c=2
    )
    theta_peak = p[..., 0]
    phi_peak = p[..., 1]
    # Change the spherical "south pole" such that the entry vector is theta=pi, phi=0,
    # then map each peak coordinate to the opposite "north pole."
    theta_peak_v = torch.pi - ((theta_peak + (torch.pi - theta_entry)) % torch.pi)
    open_interval_phi_low = -torch.pi + torch.finfo(phi_peak.dtype).eps
    phi_peak_v = wrap_bound_modulo(
        phi_peak - phi_entry, open_interval_phi_low, torch.pi
    )
    phi_peak_v = wrap_bound_modulo(
        phi_peak_v + torch.pi, open_interval_phi_low, torch.pi
    )

    peaks_p = einops.rearrange(fodf_peaks, "b ... -> b (...)")
    peaks_p_mask = einops.rearrange(peaks_valid_mask, "b ... -> b (...)")

    # Set all entry vectors to theta=pi and phi=0
    theta_entry_prime = (0 * theta_entry.expand_as(theta_peak_v)) + torch.pi
    phi_entry_prime = theta_entry.expand_as(phi_peak_v) * 0
    # Radius is always 1, but we must pass a Tensor object instead of an int for the jit
    # tracing.
    peak_dists = _euclid_dist_spherical_coords(
        peaks_p.new_ones(1),
        theta_entry_prime,
        phi_entry_prime,
        peaks_p.new_ones(1),
        theta_peak_v,
        phi_peak_v,
    )
    # Only take distances of valid peaks, ignore the "dummy" peaks that are given for
    # the purposes of processing non-jagged tensors.
    peak_dists = peaks_p_mask * peak_dists
    # Maximizing the distance between the entry "south pole" and the outgoing is equal
    # to minimizing the distance when mapping both to the same hemisphere; the great
    # circle from entry to outgoing should be as close to an "equator" as possible.
    closest_dir_idx = torch.argmax(peak_dists, dim=1, keepdim=True)
    closest_peaks = torch.take_along_dim(peaks_p, closest_dir_idx, dim=1)

    # Every peak is actually two peaks, due to polar symmetry. So, we only want the peak
    # direction that is on the opposite hemisphere from the entry vector.
    # Start from spherical peak coordinate with south pole v, then invert only the first
    # mapping (from south pole (pi, 0) -> (v_theta, v_phi)), but keep the direction in
    # the same hemisphere it's currently at.
    theta_peak_prime = (theta_peak_v - (torch.pi + theta_entry) % torch.pi) % torch.pi
    phi_peak_prime = wrap_bound_modulo(
        phi_peak_v - phi_entry, open_interval_phi_low, torch.pi
    )
    phi_peak_prime = wrap_bound_modulo(
        phi_peak_prime + torch.pi, open_interval_phi_low, torch.pi
    )
    # All directions in p_prime should be found in either the original peak coordinates,
    # or the coordinates opposite those original coordinates.
    p_prime = torch.stack([theta_peak_prime, phi_peak_prime], -1)
    closest_dirs = torch.take_along_dim(p_prime, closest_dir_idx[..., None], dim=1)

    closest_peaks = torch.reshape(closest_peaks, entry_vec_theta_phi.shape[:-1])
    closest_dirs = torch.reshape(closest_dirs, entry_vec_theta_phi.shape)

    return closest_dirs, closest_peaks
