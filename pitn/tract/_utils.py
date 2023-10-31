# -*- coding: utf-8 -*-
import collections
import functools
from functools import partial
from typing import Optional, Tuple

import jax
import jax.dlpack
import jax.numpy as jnp
import numpy as np
import torch
from jax import lax

MIN_COS_SIM = -1.0 + torch.finfo(torch.float32).eps
MAX_COS_SIM = 1.0 - torch.finfo(torch.float32).eps
MIN_THETA = 0.0
MAX_THETA = torch.pi
MIN_PHI = 0.0
MAX_PHI = (2 * torch.pi) - torch.finfo(torch.float32).eps
AT_POLE_EPS = torch.finfo(torch.float32).eps


def wrap_bound_modulo(
    x: torch.Tensor, low: torch.Tensor, high: torch.Tensor
) -> torch.Tensor:
    # Totally stolen from <https://stackoverflow.com/a/22367889>
    d = high - low
    return ((x - low) % d) + low


def __unit_sphere2xyz(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    #! Inputs to this function should generally be 64-bit floats! Precision is poor for
    #! 32-bit floats.
    # r = 1 on the unit sphere.
    # Theta does not "cycle back" between 0 and pi, it "bounces back" such as in
    # a sequence 0.01 -> 0.001 -> 0.0 -> 0.001 -> 0.01. This is unlike phi which
    # does cycle back: 2pi - 2eps -> 2pi - eps -> 0 -> 0 + eps ...
    # The where() handles theta > pi, and the abs() handles theta < pi.
    theta = torch.where(
        theta > torch.pi,
        torch.pi - (theta % torch.pi),
        torch.abs(theta),
    )
    # Phi just cycles back.
    phi = phi % (2 * torch.pi)
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    return torch.stack([x, y, z], dim=-1)


unit_sphere2xyz = torch.jit.trace(
    __unit_sphere2xyz,
    example_inputs=(
        torch.linspace(0, torch.pi, 10, dtype=torch.float64),
        torch.linspace(1e-6, (2 * torch.pi) - 1e-6, 10, dtype=torch.float64),
    ),
)
_unit_sphere2xyz = unit_sphere2xyz


def _xyz2unit_sphere_theta_phi(
    coords_xyz: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    #! Inputs to this function should generally be 64-bit floats! Precision is poor for
    #! 32-bit floats.

    x = coords_xyz[..., 0]
    y = coords_xyz[..., 1]
    z = coords_xyz[..., 2]
    r = torch.linalg.norm(coords_xyz, ord=2, axis=-1)
    r[r == 0] = 1.0
    theta = torch.arccos(z / r)
    # The discontinuities of atan2 mean we have to shift and cycle some values.
    phi = torch.arctan2(y, x) % (2 * torch.pi)
    at_pole = torch.sin(theta) < AT_POLE_EPS
    # At N and S poles, y = x = 0, which would make phi undefined. However, phi is
    # arbitrary at poles in spherical coordinates, so just set to a small non-zero value
    # for avoiding potential numerical issues.
    phi = torch.where(at_pole, AT_POLE_EPS, phi)
    return theta, phi


xyz2unit_sphere_theta_phi = torch.jit.trace(
    _xyz2unit_sphere_theta_phi,
    example_inputs=_unit_sphere2xyz(
        torch.linspace(0, torch.pi, 10, dtype=torch.float64),
        torch.linspace(1e-6, (2 * torch.pi) - 1e-6, 10, dtype=torch.float64),
    ),
)


def arc_len_spherical(
    theta_1: torch.Tensor,
    phi_1: torch.Tensor,
    theta_2: torch.Tensor,
    phi_2: torch.Tensor,
):
    coords_1 = unit_sphere2xyz(theta_1, phi_1)
    coords_2 = unit_sphere2xyz(theta_2, phi_2)
    cos_sim = torch.nn.functional.cosine_similarity(coords_1, coords_2, dim=-1)
    cos_sim.clamp_(min=MIN_COS_SIM, max=MAX_COS_SIM)
    arc_len = torch.arccos(cos_sim)
    arc_len.masked_fill_(torch.isclose(cos_sim, cos_sim.new_tensor([MAX_COS_SIM])), 0.0)
    return arc_len


def antipodal_xyz_coords(coords_xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return -coords_xyz


def antipodal_sphere_coords(
    theta: torch.Tensor, phi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.pi - theta, (phi + torch.pi) % (2 * torch.pi)


def antipodal_arc_len_spherical(
    theta_1: torch.Tensor,
    phi_1: torch.Tensor,
    theta_2: torch.Tensor,
    phi_2: torch.Tensor,
) -> torch.Tensor:
    theta_1p, phi_1p = antipodal_sphere_coords(theta_1, phi_1)
    arc_len_1_2 = arc_len_spherical(
        theta_1=theta_1, phi_1=phi_1, theta_2=theta_2, phi_2=phi_2
    )
    arc_len_1p_2 = arc_len_spherical(
        theta_1=theta_1p, phi_1=phi_1p, theta_2=theta_2, phi_2=phi_2
    )

    return torch.minimum(arc_len_1_2, arc_len_1p_2)


def t2j(t_tensor: torch.Tensor) -> jax.Array:
    t = t_tensor.contiguous()
    # Dlpack does not handle boolean arrays.
    if t.dtype == torch.bool:
        t = t.to(torch.uint8)
        to_bool = True
    else:
        to_bool = False
    if not jax.config.x64_enabled and t.dtype == torch.float64:
        # Unsafe casting, but it's necessary if jax can only handle 32-bit floats. In
        # some edge cases, like if any dimension size is 1, the conversion will error
        # out.
        t = t.to(torch.float32)

    # 1-dims cause all sorts of problems, so just remove them before conversion, then
    # add them back afterwards.
    if 1 in tuple(t.shape):
        orig_shape = tuple(t.shape)
        t = t.squeeze()
        to_expand = tuple(
            filter(lambda i_d: orig_shape[i_d] == 1, range(len(orig_shape)))
        )
    else:
        to_expand = None

    if t.device.type.casefold() == "cuda":
        target_dev_idx = t.device.index
        jax_dev = list(filter(lambda d: d.id == target_dev_idx, jax.devices()))[0]
    else:
        jax_dev = None

    j = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(t))
    if jax_dev is not None:
        j = jax.device_put(j, jax_dev)
    j = j.astype(bool) if to_bool else j

    if to_expand is not None:
        j = lax.expand_dims(j, to_expand)

    return j


def j2t(j_tensor: jax.Array, delete_from_jax: bool = False) -> torch.Tensor:
    j = j_tensor.block_until_ready()
    if j.dtype == bool:
        j = j.astype(jnp.uint8)
        to_bool = True
    else:
        to_bool = False

    if j.device().platform.casefold() == "gpu":
        target_dev_idx = j.device().id
        torch_dev = f"cuda:{target_dev_idx}"
        if target_dev_idx > (torch.cuda.device_count() - 1):
            torch_dev = None
    else:
        torch_dev = None

    t = torch.utils.dlpack.from_dlpack(
        jax.dlpack.to_dlpack(j, take_ownership=delete_from_jax)
    )
    if torch_dev is not None:
        t = t.to(torch_dev)
    t = t.bool() if to_bool else t

    return t


# def sh_basis_mrtrix3(theta: torch.Tensor, phi: torch.Tensor, l_max: int):
#     pass
