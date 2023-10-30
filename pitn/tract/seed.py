# -*- coding: utf-8 -*-
import collections
import math
from functools import partial
from typing import Callable, Optional, Tuple

import dipy
import einops
import numpy as np
import torch

import pitn

_SeedPoint = collections.namedtuple("_SeedPoint", ("theta", "phi", "amplitude"))


def get_topk_starting_peaks(
    sh_coeffs: torch.Tensor,
    seed_theta: torch.Tensor,
    seed_phi: torch.Tensor,
    fn_peak_finding__sh_theta_phi2theta_phi: Callable[
        [
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        Tuple[torch.Tensor, torch.Tensor],
    ],
    fn_sh_basis: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    max_n_peaks: int,
    min_odf_height: float,
    min_peak_arc_len: float,
) -> _SeedPoint:
    """Find largest peaks of an ODF through exhaustive searching.

    Parameters
    ----------
    sh_coeffs : torch.Tensor
        Spherical harmonic coefficients, shape [B x N_basis]
    seed_theta : torch.Tensor
        Seed direction polar angles in spherical coordinates, shape [N_seeds]
    seed_phi : torch.Tensor
        Seed direction azimuthal angles in spherical coordinates, shape [N_seeds]
    fn_peak_finding__sh_theta_phi2theta_phi : Callable[ [ torch.Tensor, torch.Tensor, torch.Tensor, ], Tuple[torch.Tensor, torch.Tensor], ]
        Peak finding function.

        Accepts a batch of SH coefficients, thetas, and phis. Returns a theta, phi
        tuple of peaks found in the ODF.
    fn_sh_basis: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Function that returns the orthnormal basis for the SH coefficients.

        Accepts a batch of theta, phi angles of shape [B_angles] and returns a batch of
        basis vectors of shape [B_angles x N_basis]
    max_n_peaks : int
        Max number of peaks to return for each SH coefficient.
    min_odf_height : float
        Minimum required height of the found peaks.
    min_peak_arc_len: float
        Minimum arc length between peaks in the same ODF such that they can be
        considered as distinct peaks.

    Returns
    -------
    _SeedPoint
        Tuple of (theta, phi, amplitude) peak angles/intensity values

        Each has the shape [B x 2 x max_n_peaks], with the size 2 dimension representing
        the antipodal angles for each peak.
    """

    batch_size = sh_coeffs.shape[0]
    n_seeds = seed_theta.numel()
    bn_sh_coeffs = einops.repeat(sh_coeffs, "b sh -> (b n_seeds) sh", n_seeds=n_seeds)
    bn_seed_theta = einops.repeat(seed_theta, "n_seeds -> (b n_seeds)", b=batch_size)
    bn_seed_phi = einops.repeat(seed_phi, "n_seeds -> (b n_seeds)", b=batch_size)

    bn_peak_theta, bn_peak_phi = fn_peak_finding__sh_theta_phi2theta_phi(
        bn_sh_coeffs, bn_seed_theta, bn_seed_phi
    )
    peak_theta = einops.rearrange(
        bn_peak_theta, "(b n_seeds) -> b n_seeds", b=batch_size
    )
    del bn_peak_theta
    peak_phi = einops.rearrange(bn_peak_phi, "(b n_seeds) -> b n_seeds", b=batch_size)
    del bn_peak_phi

    # Find min arc lengths before projecting to a hemisphere,
    within_batch_pair_arc_len = pitn.tract.antipodal_arc_len_spherical(
        theta_1=peak_theta.unsqueeze(1),
        phi_1=peak_phi.unsqueeze(1),
        theta_2=peak_theta.unsqueeze(2),
        phi_2=peak_phi.unsqueeze(2),
    )

    # Project all peaks onto northern hemisphere.
    proj_theta_phi = pitn.tract.antipodal_sphere_coords(theta=peak_theta, phi=peak_phi)
    proj_peak_theta = torch.where(
        peak_theta > (torch.pi / 2), proj_theta_phi[0], peak_theta
    )
    proj_peak_phi = torch.where(
        peak_theta > (torch.pi / 2), proj_theta_phi[1], peak_phi
    )
    del proj_theta_phi, peak_phi, peak_theta
    b_sh_coeffs = einops.rearrange(
        bn_sh_coeffs, "(b n_seeds) sh -> b n_seeds sh", n_seeds=n_seeds
    )
    del bn_sh_coeffs
    amps = b_sh_coeffs * einops.rearrange(
        fn_sh_basis(
            einops.rearrange(proj_peak_theta, "b n_seeds -> (b n_seeds)"),
            einops.rearrange(proj_peak_phi, "b n_seeds -> (b n_seeds)"),
        ),
        "(b n_seeds) sh -> b n_seeds sh",
        b=batch_size,
    )
    amps = amps.sum(-1)
    AMP_SENTINAL = min_odf_height - 100.0
    amps[amps < min_odf_height] = AMP_SENTINAL

    topk_theta = torch.zeros((batch_size, max_n_peaks)).to(proj_peak_theta)
    topk_phi = topk_theta.clone().to(proj_peak_phi)
    topk_amps = topk_theta.clone().to(amps)

    tmp_amps = amps.clone()
    amps_sort_idx = torch.argsort(tmp_amps, dim=-1, stable=True, descending=True)
    batch_idx = torch.arange(batch_size).to(amps_sort_idx)
    for i in range(max_n_peaks):
        # The first index should point to the i'th largest, distinct peak on the ODF.
        ki_idx = amps_sort_idx[:, 0]
        # Determine if other calculated peaks are within this same lobe.
        ki_same_lobe_mask = (
            within_batch_pair_arc_len[(batch_idx, ki_idx)] < min_peak_arc_len
        )
        # "Pop" off the k_i index from the amp matrix by setting all within-lobe values
        # to some sentinal value, and store the amplitude, theta, and phi values at the
        # k_i index.
        # Grab the amplitude before erasing it with the sentinal value.
        ki_amp = tmp_amps[(batch_idx, ki_idx)]
        tmp_amps[ki_same_lobe_mask] = AMP_SENTINAL
        ki_theta = proj_peak_theta[(batch_idx, ki_idx)]
        ki_phi = proj_peak_phi[(batch_idx, ki_idx)]
        # Mask to handle batches that do not contain k unique peaks.
        topk_theta[:, i] = ki_theta
        topk_phi[:, i] = ki_phi
        topk_amps[:, i] = ki_amp

        # Re-sort to find the next largest set of peaks.
        amps_sort_idx = torch.argsort(tmp_amps, dim=-1, stable=True, descending=True)
    # If a peak's amplitude equals the sentinal value, then that set of sh coefficients
    # did not have a distinct peak at that i'th position.
    invalid_peak_mask = topk_amps == AMP_SENTINAL
    topk_amps[invalid_peak_mask] = 0.0
    topk_theta[invalid_peak_mask] = 0.0
    topk_phi[invalid_peak_mask] = 0.0

    topk_amps_two_poles = einops.repeat(
        topk_amps, "b n_peak -> b num_poles n_peak", num_poles=2
    )
    antipodal_topk_theta, antipodal_topk_phi = pitn.tract.antipodal_sphere_coords(
        topk_theta, topk_phi
    )
    antipodal_topk_phi[invalid_peak_mask] = 0.0

    topk_theta_two_poles = torch.stack([topk_theta, antipodal_topk_theta], dim=-2)
    topk_phi_two_poles = torch.stack([topk_phi, antipodal_topk_phi], dim=-2)

    return _SeedPoint(
        theta=topk_theta_two_poles,
        phi=topk_phi_two_poles,
        amplitude=topk_amps_two_poles,
    )
