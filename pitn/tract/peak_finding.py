# -*- coding: utf-8 -*-
import functools
from functools import partial
from typing import Union

import einops
import jax
import jax.numpy as jnp
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


def __jax_spherical_harmonic(
    degree: jax.Array,
    order: jax.Array,
    theta: jax.Array,
    phi: jax.Array,
    l_max: int,
) -> jax.Array:
    scipy_theta = phi
    scipy_phi = theta
    return jax.scipy.special.sph_harm(
        m=order, n=degree, theta=scipy_theta, phi=scipy_phi, n_max=l_max
    )


def __jax_norm_lagrange_poly(
    degree: jax.Array,
    order: jax.Array,
    theta: jax.Array,
    l_max: int,
) -> jax.Array:
    # We don't care about phi, so long as it doesn't 0-out the sh value.
    dummy_phi = jnp.zeros_like(theta)
    sh = __jax_spherical_harmonic(
        degree=degree, order=order, theta=theta, phi=dummy_phi, l_max=l_max
    )
    exp_theta = jnp.exp(1j * order * dummy_phi)
    # Remove the e^(i m phi) to just get N_lm * Y_lm(cos theta)
    # Only return real component, imaginary component should be 0 anyway.
    return (sh / exp_theta).real


def _norm_lagrange_poly(
    degree: torch.Tensor,
    order: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:

    theta = theta.squeeze(-1)
    degree = degree.squeeze(0)
    order = order.squeeze(0)
    l_max = int(degree.max().cpu().item())

    assert tuple(degree.shape) == tuple(order.shape)
    if (theta.ndim > 2) or (degree.ndim > 2):
        raise RuntimeError(
            "ERROR: Will only accept 1D or 2D shapes, "
            + f"got {tuple(theta.shape)}, {tuple(degree.shape)}"
        )
    elif (theta.ndim == 2) and (degree.ndim == 2):
        assert tuple(theta.shape) == tuple(degree.shape)
    elif theta.ndim == 2:
        target_shape = tuple(theta.shape)
    elif degree.ndim == 2:
        target_shape = tuple(degree.shape)
    else:
        target_shape = (theta.shape[0], degree.shape[0])
    batch_size_angle = target_shape[0]
    batch_size_lm = target_shape[1]
    l_max = int(degree.max().cpu().item())

    if tuple(degree.shape) != target_shape:
        broad_degree = einops.repeat(
            degree, "b_lm -> b_angle b_lm", b_angle=batch_size_angle
        )
        broad_order = einops.repeat(
            order, "b_lm -> b_angle b_lm", b_angle=batch_size_angle
        )
    else:
        broad_degree = degree
        broad_order = order
    if tuple(theta.shape) != target_shape:
        broad_theta = einops.repeat(
            theta, "b_angle -> b_angle b_lm", b_lm=batch_size_lm
        )
    else:
        broad_theta = theta

    broad_degree = einops.rearrange(broad_degree, "b_angle b_lm -> (b_angle b_lm)")
    broad_order = einops.rearrange(broad_order, "b_angle b_lm -> (b_angle b_lm)")
    broad_theta = einops.rearrange(broad_theta, "b_angle b_lm -> (b_angle b_lm)")

    jax_norm_P_lm = __jax_norm_lagrange_poly(
        degree=pitn.tract.t2j(broad_degree),
        order=pitn.tract.t2j(broad_order),
        theta=pitn.tract.t2j(broad_theta),
        l_max=l_max,
    )
    norm_P_lm = einops.rearrange(
        pitn.tract.j2t(jax_norm_P_lm),
        "(b_angle b_lm) -> b_angle b_lm",
        b_angle=batch_size_angle,
    )

    return norm_P_lm


def _sph_harm(
    degree: torch.Tensor,
    order: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:

    theta = theta.squeeze(-1)
    phi = phi.squeeze(-1)
    degree = degree.squeeze(0)
    order = order.squeeze(0)

    assert tuple(degree.shape) == tuple(order.shape)
    assert tuple(theta.shape) == tuple(phi.shape)
    if (theta.ndim > 2) or (degree.ndim > 2):
        raise RuntimeError(
            "ERROR: Will only accept 1D or 2D shapes, "
            + f"got {tuple(theta.shape)}, {tuple(degree.shape)}"
        )
    elif (theta.ndim == 2) and (degree.ndim == 2):
        assert tuple(theta.shape) == tuple(degree.shape)
    elif theta.ndim == 2:
        target_shape = tuple(theta.shape)
    elif degree.ndim == 2:
        target_shape = tuple(degree.shape)
    else:
        target_shape = (theta.shape[0], degree.shape[0])
    batch_size_angle = target_shape[0]
    batch_size_lm = target_shape[1]
    l_max = int(degree.max().cpu().item())

    if tuple(degree.shape) != target_shape:
        broad_degree = einops.repeat(
            degree, "b_lm -> b_angle b_lm", b_angle=batch_size_angle
        )
        broad_order = einops.repeat(
            order, "b_lm -> b_angle b_lm", b_angle=batch_size_angle
        )
    else:
        broad_degree = degree
        broad_order = order
    if tuple(theta.shape) != target_shape:
        broad_theta = einops.repeat(
            theta, "b_angle -> b_angle b_lm", b_lm=batch_size_lm
        )
        broad_phi = einops.repeat(phi, "b_angle -> b_angle b_lm", b_lm=batch_size_lm)
    else:
        broad_theta = theta
        broad_phi = phi

    broad_degree = einops.rearrange(broad_degree, "b_angle b_lm -> (b_angle b_lm)")
    broad_order = einops.rearrange(broad_order, "b_angle b_lm -> (b_angle b_lm)")
    broad_theta = einops.rearrange(broad_theta, "b_angle b_lm -> (b_angle b_lm)")
    broad_phi = einops.rearrange(broad_phi, "b_angle b_lm -> (b_angle b_lm)")

    jax_sph_harm = __jax_spherical_harmonic(
        degree=pitn.tract.t2j(broad_degree),
        order=pitn.tract.t2j(broad_order),
        theta=pitn.tract.t2j(broad_theta),
        phi=pitn.tract.t2j(broad_phi),
        l_max=l_max,
    )
    sph_harm_vals = einops.rearrange(
        pitn.tract.j2t(jax_sph_harm),
        "(b_angle b_lm) -> b_angle b_lm",
        b_angle=batch_size_angle,
    )

    return sph_harm_vals


def sh_grad(
    sh_coeffs: torch.Tensor,
    l: torch.Tensor,
    m: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
    pole_angle_tol: float = 1e-5,
) -> torch.Tensor:
    l_max = int(l.max().cpu().item())

    cart_coords = pitn.tract.unit_sphere2xyz(theta=theta, phi=phi)
    x = cart_coords[..., 0]
    y = cart_coords[..., 1]

    elev_mrtrix = theta
    azim_mrtrix = torch.arctan2(y, x)
    at_pole = torch.sin(elev_mrtrix) < pole_angle_tol

    # Convenience function for indexing into l/m indexable matrices.
    bmask_l_m = lambda l_, m_: (Ellipsis, _broad_sh_mask(l_all=l, m_all=m, l=l_, m=m_))

    # P_l_m = _lagrange_poly(order=m, degree=l, x=torch.cos(elev_mrtrix))
    # N_l_m = _gen_sh_norm_coeff(
    #     l_max=l_max, batch_size=batch_size, dtype=theta.dtype, device=theta.device
    # )
    # norm_P_l_m = N_l_m * P_l_m
    norm_P_l_m = _norm_lagrange_poly(degree=l, order=m, theta=elev_mrtrix)

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

    # Scale ds/d azimuth to make this derivative into a gradient.
    # <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Integration_and_differentiation_in_spherical_coordinates>
    ds_dazim /= torch.where(~at_pole, torch.sin(elev_mrtrix), 1.0)

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
        sh_grad,
        sh_coeffs=sh_coeffs,
        l=l,
        m=m,
        pole_angle_tol=POLE_ANGLE_TOL,
    )
    # If a batch has all zero phi angles or all zero sh coefficients, then that
    # batch should not be optimized over, and should already be considered "converged"
    converged = (seed_theta == 0.0) | (sh_coeffs == 0.0).all(-1)
    # converged = torch.zeros_like(seed_theta).bool()
    nu_t = torch.zeros_like(params)
    epoch = 0
    if return_all_steps:
        param_steps.append(params)

    # Main gradient ascent loop.
    if converged.all():
        max_epochs = 0
    for epoch in range(max_epochs):
        nu_tm1 = nu_t
        grad_t = grad_fn(theta=params[..., 0], phi=params[..., 1])
        nu_t = momentum * nu_tm1 + lr * grad_t
        # Batches that have converged should no longer be updated.
        # Add the gradient to perform gradient ascent, rather than gradient descent.
        params_tp1 = params + (nu_t * (~converged.unsqueeze(-1)))
        # Theta does not "cycle back" between 0 and pi, it "bounces back" such as in
        # a sequence 0.01 -> 0.001 -> 0.0 -> 0.001 -> 0.01. This is unlike phi which
        # does cycle back: 2pi - 2eps -> 2pi - eps -> 0 -> 0 + eps ...
        theta_tp1 = params_tp1[..., 0]
        # The where() handles theta > pi, and the abs() handles theta < pi.
        params_tp1[..., 0] = torch.where(
            theta_tp1 > torch.pi,
            torch.pi - (theta_tp1 % torch.pi),
            torch.abs(theta_tp1),
        )
        # Phi just cycles back.
        params_tp1[..., 1] %= 2 * torch.pi
        arc_len_t_to_tp1 = pitn.tract.arc_len_spherical(
            theta_1=params[..., 0],
            phi_1=params[..., 1],
            theta_2=params_tp1[..., 0],
            phi_2=params_tp1[..., 1],
        )

        converged = converged | (arc_len_t_to_tp1 < tol_angular)
        if converged.all():
            break

        params = params_tp1
        if return_all_steps:
            param_steps.append(params)

    if not return_all_steps:
        ret = (params[..., 0], params[..., 1])
    else:
        ret = (params[..., 0], params[..., 1], param_steps)

    return ret
