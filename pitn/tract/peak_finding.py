# -*- coding: utf-8 -*-
import functools
from functools import partial
from typing import Union

import einops
import numpy as np
import scipy
import torch

import pitn


@functools.lru_cache(maxsize=10)
def _get_degree_order_vecs(l_max: int, batch_size: int, device="cpu"):
    unique_l = np.arange(0, l_max + 2, 2).astype(int)

    l_degrees = list()
    m_orders = list()
    for l in unique_l:
        for m in np.arange(-l, l + 1):
            l_degrees.append(l)
            m_orders.append(m)
    l_degrees = einops.repeat(
        torch.Tensor(l_degrees).to(torch.int32), "d -> b d", b=batch_size
    )
    m_orders = einops.repeat(
        torch.Tensor(m_orders).to(torch.int32), "o -> b o", b=batch_size
    )
    return l_degrees.to(device), m_orders.to(device)


def _broad_sh_mask(l_all: torch.Tensor, m_all: torch.Tensor, l=None, m=None):
    # Assume that for l_all and m_all, the first N-1 dimensions are "batch" dimensions
    # that store "batch" number of repeats of the same array, with unique values in dim
    # -1.
    l_all = l_all.view(-1, l_all.shape[-1])[0]
    m_all = m_all.view(-1, m_all.shape[-1])[0]

    if l is None:
        l = torch.unique(l_all)
    elif np.isscalar(l):
        l = torch.as_tensor([l]).reshape(-1)
    elif torch.is_tensor(l):
        l = l.reshape(-1)
    else:
        # General iterable
        l = torch.as_tensor(list(l)).reshape(-1)
    l = l.to(l_all)
    if m is None:
        m = torch.unique(m_all)
    elif np.isscalar(m):
        m = torch.as_tensor([m]).reshape(-1)
    elif torch.is_tensor(m):
        m = m.reshape(-1)
    else:
        # General iterable
        m = torch.as_tensor(list(m)).reshape(-1)
    m = m.to(m_all)

    return torch.isin(l_all, l) & torch.isin(m_all, m)


def _sh_idx(l: torch.Tensor, m: torch.Tensor, invalid_idx_replace=None):
    # Assume l and m are copied across their respective n-1 first dimensions.
    l_ = l.view(-1, l.shape[-1])[0]
    m_ = m.view(-1, m.shape[-1])[0]
    idx = ((l_ // 2) * (l_ + 1) + m_).to(torch.long)

    if invalid_idx_replace is not None:
        l_max = l.max().cpu().item()
        max_idx = (((l_max + 1) * (l_max + 2)) // 2) - 1
        idx.masked_fill_(idx > max_idx, int(invalid_idx_replace))
    return idx


@torch.no_grad()
def _lagrange_poly(
    degree: torch.Tensor, order: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    order_ = order.detach().cpu().numpy()
    degree_ = degree.detach().cpu().numpy()
    x_ = x.detach().cpu().numpy()
    associate_lagr_poly = scipy.special.lpmv(order_, degree_, x_[..., np.newaxis])
    return torch.from_numpy(associate_lagr_poly).to(x)


@torch.no_grad()
def _spherical_harmonic(
    degree: torch.Tensor,
    order: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
):
    theta_scipy = phi
    theta_scipy = theta_scipy.detach().cpu().numpy()
    phi_scipy = theta
    phi_scipy = phi_scipy.detach().cpu().numpy()
    order_ = order.detach().cpu().numpy()
    degree_ = degree.detach().cpu().numpy()

    Y_m_l = scipy.special.sph_harm(
        order_, degree_, theta_scipy[..., np.newaxis], phi_scipy[..., np.newaxis]
    )

    return torch.from_numpy(Y_m_l).to(theta.device)


def _sh_basis_mrtrix3(
    theta: torch.Tensor,
    phi: torch.Tensor,
    degree: torch.Tensor,
    order: torch.Tensor,
):
    Y_m_abs_l = _spherical_harmonic(
        order=torch.abs(order), degree=degree, theta=theta, phi=phi
    )
    Y_m_abs_l = torch.where(order < 0, np.sqrt(2) * Y_m_abs_l.imag, Y_m_abs_l)
    Y_m_abs_l = torch.where(order > 0, np.sqrt(2) * Y_m_abs_l.real, Y_m_abs_l)

    return Y_m_abs_l.real


def _batch_basis_mrtrix3(
    theta: torch.Tensor,
    phi: torch.Tensor,
    l_max: int,
    degree: torch.Tensor,
    order: torch.Tensor,
):
    Y_m_abs_l = _spherical_harmonic(
        order=torch.abs(order), degree=degree, theta=theta, phi=phi
    )
    Y_m_abs_l = torch.where(order < 0, np.sqrt(2) * Y_m_abs_l.imag, Y_m_abs_l)
    Y_m_abs_l = torch.where(order > 0, np.sqrt(2) * Y_m_abs_l.real, Y_m_abs_l)

    return Y_m_abs_l.real


@torch.no_grad()
@functools.lru_cache(maxsize=10)
def _gen_sh_norm_coeff(l_max: int, batch_size: int, device, dtype):

    l, m = _get_degree_order_vecs(l_max=l_max, batch_size=batch_size)
    l_ = l[0].numpy()
    m_ = m[0].numpy()
    f1 = (2 * l_ + 1) / (4 * np.pi)
    f2 = scipy.special.factorial(l_ - m_) / scipy.special.factorial(l_ + m_)

    N_l_m_ = np.sqrt(f1 * f2)
    N_l_m = (
        torch.from_numpy(N_l_m_).to(dtype=dtype, device=device).repeat(batch_size, 1)
    )
    return N_l_m


def sh_first_derivative(
    sh_coeffs: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
    pole_angle_tol: float = 1e-5,
) -> torch.Tensor:
    batch_size = theta.shape[0]
    l_max = int(l.max().cpu().item())

    cart_coords = pitn.tract.unit_sphere2xyz(theta=theta, phi=phi)
    x = cart_coords[..., 0]
    y = cart_coords[..., 1]
    z = cart_coords[..., 2]

    elev_mrtrix = torch.arccos(z)
    azim_mrtrix = torch.arctan2(y, x)
    at_pole = torch.sin(elev_mrtrix) < pole_angle_tol

    # Convenience function for indexing into l/m indexable matrices.
    bmask_l_m = lambda l_, m_: (Ellipsis, _broad_sh_mask(l_all=l, m_all=m, l=l_, m=m_))

    P_l_m = _lagrange_poly(order=m, degree=l, x=torch.cos(elev_mrtrix))
    N_l_m = _gen_sh_norm_coeff(
        l_max=l_max, batch_size=batch_size, dtype=theta.dtype, device=theta.device
    )
    norm_P_l_m = N_l_m * P_l_m

    nonzero_degrees = list(range(2, l_max + 1, 2))
    ds_delev_zonal = (
        torch.sqrt(l * (l + 1))[bmask_l_m(nonzero_degrees, 0)]
        * norm_P_l_m[bmask_l_m(nonzero_degrees, 1)]
        * sh_coeffs[bmask_l_m(nonzero_degrees, 0)]
    ).sum(-1)

    m_pos_mask = bmask_l_m(None, range(1, l_max + 1))
    # Shape [batch_size, (n_sh_coeffs - ((l_max / 2) + 1))/2], for both pos m and neg m.
    m_pos = m[m_pos_mask]
    m_neg = -m_pos
    l_pos_m = l[m_pos_mask]
    cos_azim = np.sqrt(2) * torch.cos(m_pos * azim_mrtrix.unsqueeze(-1))
    sin_azim = np.sqrt(2) * torch.sin(m_pos * azim_mrtrix.unsqueeze(-1))
    coeff_m_pos = sh_coeffs[m_pos_mask]
    coeff_m_neg = sh_coeffs[..., _sh_idx(l_pos_m, m_neg)]

    k = (
        torch.sqrt((l_pos_m + m_pos) * (l_pos_m - m_pos + 1))
        * norm_P_l_m[..., _sh_idx(l_pos_m, m_pos - 1)]
    )
    # Only add the second term when m + 1 exists, aka not greater than m's respective
    # degree.
    k -= torch.where(
        l_pos_m > m_pos,
        torch.sqrt((l_pos_m - m_pos) * (l_pos_m + m_pos + 1))
        * norm_P_l_m[..., _sh_idx(l_pos_m, m_pos + 1, invalid_idx_replace=0)],
        0,
    )
    k *= -0.5

    ds_delev_nonzone = k * (cos_azim * coeff_m_pos + sin_azim * coeff_m_neg)
    ds_delev_nonzone = ds_delev_nonzone.sum(-1)

    # Numerically, derivative of the azimuth depends on whether or not we're at a pole.
    ds_dazim = torch.where(
        at_pole.unsqueeze(-1),
        k * (cos_azim * coeff_m_neg - sin_azim * coeff_m_pos),
        m_pos
        * norm_P_l_m[m_pos_mask]
        * (cos_azim * coeff_m_neg - sin_azim * coeff_m_pos),
    )
    ds_dazim = ds_dazim.sum(-1)

    # This division by sin(theta) makes the result derivative differ from the finite-
    # difference estimate. It's in the mrtrix derivative, but why?
    # <https://github.com/MRtrix3/mrtrix3/blob/5e95b5a498da0358984d3dbc2d8c05e37610fb9d/core/math/SH.h#L643>
    # ds_dazim /= torch.where(~at_pole, torch.sin(elev_mrtrix), 1.0)

    ds_delev = ds_delev_zonal + ds_delev_nonzone

    jac = torch.stack([ds_delev, ds_dazim], dim=-1)

    return jac


@torch.no_grad()
def find_peak_grad_ascent(
    sh_coeffs: torch.Tensor,
    seed_theta: torch.Tensor,
    seed_phi: torch.Tensor,
    lr: float,
    momentum: float,
    max_epochs: int,
    l_max: int,
    tol_angular: float = 0.01 * (torch.pi / 180),
    return_all_steps: bool = False,
) -> Union[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]],
]:
    batch_size = sh_coeffs.shape[0]
    l, m = _get_degree_order_vecs(
        l_max=l_max, batch_size=batch_size, device=sh_coeffs.device
    )
    params = torch.stack([seed_theta, seed_phi], -1)
    param_steps = list() if return_all_steps else None

    POLE_ANGLE_TOL = 1e-6
    grad_fn = partial(
        sh_first_derivative,
        sh_coeffs=sh_coeffs,
        l=l,
        m=m,
        pole_angle_tol=POLE_ANGLE_TOL,
    )

    converged = torch.zeros_like(seed_theta).bool()
    nu_t = torch.zeros_like(params)
    epoch = 0
    if return_all_steps:
        param_steps.append(params)

    # Main gradient ascent loop.
    while (not converged.all()) and (epoch < max_epochs):

        nu_tm1 = nu_t
        grad_t = grad_fn(theta=params[..., 0], phi=params[..., 1])
        nu_t = momentum * nu_tm1 + lr * grad_t
        # Batches that have converged should no longer be updated.
        # Add the gradient to perform gradient ascent, rather than gradient descent.
        params_tp1 = params + (nu_t * (~converged.unsqueeze(-1)))
        # Never thought I'd use the %= operator...
        params_tp1[..., 0] %= torch.pi
        params_tp1[..., 1] %= 2 * torch.pi
        arc_len_t_to_tp1 = pitn.tract.arc_len_spherical(
            theta_1=params[..., 0],
            phi_1=params[..., 1],
            theta_2=params_tp1[..., 0],
            phi_2=params_tp1[..., 1],
        )

        converged = converged | (arc_len_t_to_tp1 < tol_angular)

        epoch += 1
        params = params_tp1
        if return_all_steps:
            param_steps.append(params)

    if not return_all_steps:
        ret = (params[..., 0], params[..., 1])
    else:
        ret = (params[..., 0], params[..., 1], param_steps)

    return ret
