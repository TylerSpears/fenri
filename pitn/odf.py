# -*- coding: utf-8 -*-
import collections
import functools
from typing import Tuple

import dipy
import dipy.reconst.csdeconv
import dipy.reconst.shm
import einops
import numpy as np
import torch
import torch.nn.functional as F

import pitn
import pitn.affine
import pitn.tract.direction


def gfa(fodf_samples: torch.Tensor, sphere_samples_idx=1) -> torch.Tensor:
    s = fodf_samples.movedim(sphere_samples_idx, -1)
    s = einops.rearrange(s, "... samples -> (...) samples", samples=s.shape[-1])
    s_var = torch.var(s, dim=-1, keepdim=True, unbiased=True)
    s_ms = torch.mean(s**2, dim=-1, keepdim=True)
    gfa_ = torch.where(s_ms > 0, torch.sqrt(s_var / s_ms), 0)

    result_shape = list(fodf_samples.shape)
    result_shape[sphere_samples_idx] = 1
    return gfa_.reshape(result_shape)


def sample_odf_coeffs_lin_interp(
    coords_mm_zyx: torch.Tensor,
    fodf_coeff_vol: torch.Tensor,
    affine_vox2mm: torch.Tensor,
    padding_mode="zeros",
    align_corners=True,
) -> torch.Tensor:
    interp_coeffs = pitn.affine.sample_3d(
        fodf_coeff_vol,
        coords_mm_zyx,
        affine_vox2mm=affine_vox2mm,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    return interp_coeffs


_ThetaPhiResult = collections.namedtuple("_ThetaPhiResult", ("theta", "phi"))


@functools.lru_cache(maxsize=10)
def get_torch_sample_sphere_coords(
    sphere, device: torch.DeviceObjType, dtype: torch.dtype
) -> _ThetaPhiResult:
    theta = sphere.theta
    phi = sphere.phi
    theta = torch.from_numpy(theta).to(device=device, dtype=dtype, copy=True)
    phi = torch.from_numpy(phi).to(device=device, dtype=dtype, copy=True)

    return _ThetaPhiResult(theta, phi)


_SHOrderDegreeResult = collections.namedtuple(
    "_SHOrderDegreeResult", ("sh", "order", "degree")
)


@functools.lru_cache(maxsize=10)
def get_torch_sh_transform(
    sh_order: int, polar_coord: torch.Tensor, azimuth_coord: torch.Tensor
) -> _SHOrderDegreeResult:

    polar = polar_coord.detach().flatten().cpu().numpy()
    azimuth = azimuth_coord.detach().flatten().cpu().numpy()
    # The azimuth in dipy/scipy is set to be between (0, 2*pi], rather than (-pi, pi],
    # so the azimuth coordinate must be set to that compatible range.
    if azimuth.min() < 0 and azimuth.min() >= -torch.pi and azimuth.max() <= torch.pi:
        azimuth = pitn.tract.direction.wrap_bound_modulo(azimuth, 0, 2 * torch.pi)
    # The Tournier basis is what Mrtrix uses, so we'll default to that for now.
    # <https://mrtrix.readthedocs.io/en/3.0.4/concepts/spherical_harmonics.html#formulation-used-in-mrtrix3>
    # <https://github.com/dipy/dipy/blob/13af40fec09fb23a3692cb0bfdcb91d08acfd766/dipy/reconst/shm.py#L363>
    sh, order, degree = dipy.reconst.shm.real_sh_tournier(
        sh_order=sh_order, theta=polar, phi=azimuth, full_basis=False, legacy=False
    )
    # sh, order, degree = dipy.reconst.csdeconv.real_sh_descoteaux(
    #     sh_order=sh_order, theta=azimuth, phi=polar, full_basis=False, legacy=False
    # )
    sh = torch.from_numpy(sh).to(polar_coord)
    order = torch.from_numpy(order).to(polar_coord.device)
    degree = torch.from_numpy(degree).to(polar_coord.device)

    return _SHOrderDegreeResult(sh, order, degree)


def sample_sphere_coords(
    odf_coeffs: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
    sh_order: int,
    sh_order_dim=1,
    mask: torch.Tensor = None,
    force_nonnegative: bool = True,
) -> torch.Tensor:
    """Sample a spherical function on the sphere with coefficients in the SH domain.

    Parameters
    ----------
    odf_coeffs : torch.Tensor
        Spherical function coefficients in the SH domain that define the fn to sample.
    theta : torch.Tensor
        A flattened Tensor of spherical polar coordinate theta in range [0, $\pi$].
    phi : torch.Tensor
        A flattened Tensor of spherical azimuth coordinate phi in range ($-\pi$, $\pi$].
    sh_order : int
        Even-valued spherical harmonics order, must match number of orders in odf_coeffs
    mask : torch.Tensor, optional
        Mask Tensor for masking "voxels" in the odf coeffs, by default None.

        Shape is assumed to be b x c x space_dim_0 [x space_dim_1 x ...], with c=1.

    Returns
    -------
    torch.Tensor
    """
    orig_spatial_shape = list(odf_coeffs.shape)
    orig_spatial_shape.pop(sh_order_dim)
    orig_spatial_shape = tuple(orig_spatial_shape)
    if odf_coeffs.ndim < 5:
        if odf_coeffs.ndim == 1:
            odf_coeffs = odf_coeffs[None]
        if odf_coeffs.ndim == 2:
            odf_coeffs = odf_coeffs[..., None, None, None]

    fn_coeffs = odf_coeffs.movedim(sh_order_dim, -1)
    fn_coeffs = fn_coeffs.reshape(-1, fn_coeffs.shape[-1])
    if mask is not None:
        fn_mask = mask.squeeze(sh_order_dim)
        # fn_mask = mask.movedim(1, -1)[..., 0]
        fn_mask = fn_mask.broadcast_to(orig_spatial_shape).reshape(-1)
        fn_coeffs = fn_coeffs[fn_mask]
    else:
        fn_mask = None
    phi = pitn.tract.direction.wrap_bound_modulo(phi, 0, 2 * torch.pi)
    sh_transform, _, _ = get_torch_sh_transform(
        sh_order=sh_order, polar_coord=theta, azimuth_coord=phi
    )
    # Expand to have batch dim of 1.
    sh_transform = sh_transform.T[None]

    if fn_mask is not None:
        fn_samples = torch.zeros(
            fn_mask.shape[0],
            sh_transform.shape[2],
            dtype=sh_transform.dtype,
            device=sh_transform.device,
        )
        fn_samples[fn_mask] = torch.matmul(fn_coeffs, sh_transform)
    else:
        fn_samples = torch.matmul(fn_coeffs, sh_transform)

    if force_nonnegative:
        # Enforce non-negativity constraint.
        fn_samples.clamp_min_(0)
    fn_samples = fn_samples.reshape(*orig_spatial_shape, -1)
    fn_samples = fn_samples.movedim(-1, sh_order_dim)

    return fn_samples.contiguous()


def thresh_fodf_samples_by_pdf(
    sphere_samples: torch.Tensor, pdf_thresh_min: float
) -> torch.Tensor:
    s_pdf = sphere_samples - sphere_samples.min(1, keepdim=True).values
    s_pdf = s_pdf / s_pdf.sum(1, keepdim=True)
    s = torch.where(s_pdf < pdf_thresh_min, 0, sphere_samples)

    return s


def thresh_fodf_samples_by_value(
    sphere_samples: torch.Tensor, value_min: float
) -> torch.Tensor:
    s = torch.where(sphere_samples < value_min, 0, sphere_samples)
    return s


def thresh_fodf_samples_by_quantile(
    sphere_samples: torch.Tensor, q_low: float
) -> torch.Tensor:
    q_low_val = torch.nanquantile(sphere_samples, q=q_low, dim=1, keepdim=True)
    s = torch.where(sphere_samples < q_low_val, 0, sphere_samples)

    return s


def _unit_sphere_arc_length(
    theta_1: torch.Tensor,
    phi_1: torch.Tensor,
    theta_2: torch.Tensor,
    phi_2: torch.Tensor,
) -> torch.Tensor:
    # Assume these are all arcs on the unit sphere.
    r = 1
    dist_squared = (
        r**2
        + r**2
        - 2
        * r
        * r
        * (
            torch.sin(theta_1) * torch.sin(theta_2) * torch.cos(phi_1 - phi_2)
            + torch.cos(theta_1) * torch.cos(theta_2)
        )
    )
    dist = torch.sqrt(dist_squared)

    return dist


@functools.lru_cache(maxsize=10)
def adjacent_sphere_points_idx(
    theta: torch.Tensor, phi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert theta.shape == phi.shape
    assert theta.ndim == 1 and phi.ndim == 1
    # Get the full pairwise arc length between all points on the sphere. This matrix is
    # symmetric.
    pairwise_arc_len = _unit_sphere_arc_length(
        theta[:, None], phi[:, None], theta[None], phi[None]
    )

    # Find the arc length that will determine adjacency to each point.
    # Find the closest pole idx.
    pole_idx = torch.sort(theta).indices[0]
    pole_adjacent = pairwise_arc_len[pole_idx, :]
    pole_adjacent_sorted = torch.sort(pole_adjacent)
    # Grab the arc length halfway between the length of the "closest 6" and the next
    # closest set.
    sphere_surface_point_radius = (
        pole_adjacent_sorted.values[6]
        + (pole_adjacent_sorted.values[7] - pole_adjacent_sorted.values[6]) / 2
    )

    arc_len_sorted = pairwise_arc_len.sort(1)
    # Grab indices 1-7 because we don't care about index 0 (same point, arc len ~= 0.0), and
    # we only want *up to* the closest 6 points.
    nearest_point_idx = arc_len_sorted.indices[:, 1:7]
    # Now we want only those points within the pre-determined radius. Points near the bottom
    # of the hemisphere will have fewer than 6 adjacent points.
    # !Make sure to use the mask to avoid silent indexing errors in the future!
    # We could provide invalid indices at the non-adjacent points, but that makes
    # function-writing more difficult down the line.
    nearest_point_idx_mask = arc_len_sorted.values[:, 1:7] < sphere_surface_point_radius

    return nearest_point_idx, nearest_point_idx_mask
