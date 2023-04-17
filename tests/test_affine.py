# -*- coding: utf-8 -*-
import einops
import numpy as np
import pytest
import pytransform3d
import pytransform3d.rotations as rots
import pytransform3d.transformations as tfs
import torch

import pitn


def test_transform_coords_simple_affines():
    # Unbatched
    s = 10
    coords = torch.stack(
        torch.meshgrid(
            [
                torch.arange(s),
            ]
            * 3,
            indexing="ij",
        ),
        dim=-1,
    ).float()
    affine = torch.eye(4).float()

    # Identity
    c_t = pitn.affine.transform_coords(coords, affine)
    assert c_t.shape == coords.shape
    assert torch.isclose(c_t, coords).all()

    # Isotropic scaling
    # Decrease unit size from 1 to 0.5.
    a = affine.clone()
    a[:3, :3] = a[:3, :3] / 2
    c_t = pitn.affine.transform_coords(coords, a)
    assert torch.isclose(c_t, coords / 2).all()
    # Increase unit size to a downsample factor.
    downsample_factor = 4.79
    a = affine.clone()
    a[:3, :3] = a[:3, :3] * downsample_factor
    c_t = pitn.affine.transform_coords(coords, a)
    assert torch.isclose(c_t, coords * downsample_factor).all()

    # Translation
    a = affine.clone()
    t = a.new_tensor([2.794, -9.033, 0.442])
    a[:3, -1] = a[:3, -1] + t
    c_t = pitn.affine.transform_coords(coords, a)
    assert torch.isclose(c_t, coords + t[None, None, None]).all()
    # Scaling + translation
    a = affine.clone()
    scale = 1.55
    t = a.new_tensor([-0.97, 12.3, 0.0])
    a[:3, -1] = a[:3, -1] + t
    a[:3, :3] = a[:3, :3] * scale
    c_t = pitn.affine.transform_coords(coords, a)
    assert torch.isclose(c_t, (coords * scale) + t[None, None, None]).all()


def test_transform_coords_vs_pytransform3d_outputs():
    p_a = np.stack(
        [
            [0, 0, 0],
            [1, 1, 1],
            [0.5, 0.5, 0.5],
            [12.756, 0.54108, 4.3926],
            [3.068, -7.3837, 5.7512],
            [23.1878, 10.2116, -16.45599],
        ],
        axis=0,
    )

    # Add batch dimension of 1.
    torch_p_a = torch.from_numpy(p_a)[None]

    # p3d accepts homogenious coordinates, so make the fourth coordinate equal to 1.
    p_a = np.concatenate([p_a, np.ones((p_a.shape[0], 1))], axis=-1)

    n_translations = 5
    translations = np.stack(
        np.meshgrid(
            np.linspace(0, 4, n_translations),
            np.linspace(-2, 2, n_translations),
            np.linspace(40, 79, n_translations),
            indexing="ij",
        ),
        axis=-1,
    )
    translations = translations.reshape(-1, 3)

    n_rots = 5
    rotations = np.stack(
        np.meshgrid(
            np.linspace(-np.pi, np.pi, n_rots),
            np.linspace(-np.pi / 2, np.pi / 2, n_rots),
            np.linspace(-np.pi, np.pi, n_rots),
            indexing="ij",
        ),
        axis=-1,
    )
    rotations = rotations.reshape(-1, 3)

    for t in translations:
        t = np.squeeze(t)
        for r in rotations:
            r = np.squeeze(r)
            A_r = rots.matrix_from_euler(r, 0, 1, 2, extrinsic=True)
            A = tfs.transform_from(A_r, t)

            torch_A = torch.from_numpy(A)
            p3d_p_b = tfs.transform(A, p_a)
            p3d_p_b = p3d_p_b[..., :-1]

            pitn_p_b = pitn.affine.transform_coords(torch_p_a, torch_A)
            assert torch.isclose(torch.from_numpy(p3d_p_b), pitn_p_b).all()


def test_transform_coords_batch_and_position_invariance():
    b = 3
    s = 10
    coords = torch.stack(
        torch.meshgrid(
            [
                torch.arange(s),
            ]
            * 3,
            indexing="ij",
        ),
        dim=-1,
    ).float()
    coords_b = torch.stack([coords, -coords, coords * 0.988 + 1.79], dim=0)

    affine = torch.eye(4).float()
    affine_b = list()
    scale = 1.55
    t = affine.new_tensor([-0.97, 12.3, 1.0])
    for coeff_t, coeff_s in zip((0, 28.7, -3.23), (1, 0.42, 2.55)):
        a = affine.clone()
        a[:3, -1] = a[:3, -1] + (coeff_t * t)
        a[:3, :3] = a[:3, :3] * (coeff_s * scale)
        affine_b.append(a)
    affine_b = torch.stack(affine_b, dim=0)

    c_res_orig = pitn.affine.transform_coords(coords_b, affine_b)

    # Reverse batch ordering
    c = torch.flip(coords_b, dims=(0,))
    a = torch.flip(affine_b, dims=(0,))
    c_res_reverse = pitn.affine.transform_coords(c, a)
    assert torch.isclose(c_res_orig, torch.flip(c_res_reverse, (0,))).all()

    # Broadcast coordinates over batch dim
    c = coords_b[0:1]
    a = affine_b
    c_res_broad_over_c = pitn.affine.transform_coords(c, a)
    assert torch.isclose(c_res_orig[0], c_res_broad_over_c[0]).all()


def test_canonicalize_coords_3d_affine_valid_inputs():
    # Both batched, same batch size.
    b = 5
    affine = torch.eye(4)
    affine_b = einops.repeat(affine, "a_1 a_2 -> b a_1 a_2", b=b)

    s = 10
    coords = torch.stack(
        torch.meshgrid(
            [
                torch.arange(s),
            ]
            * 3,
            indexing="ij",
        ),
        dim=-1,
    )
    coords_b = einops.repeat(coords, "s1 s2 s3 coord -> b s1 s2 s3 coord", b=b)

    c_res, a_res = pitn.affine._canonicalize_coords_3d_affine(coords_b, affine_b)
    assert tuple(c_res.shape) == (b, s, s, s, 3)
    assert tuple(a_res.shape) == (b, 4, 4)
    assert c_res.dtype == a_res.dtype

    # Only coords is batched.
    c_res, a_res = pitn.affine._canonicalize_coords_3d_affine(coords_b, affine)
    assert tuple(c_res.shape) == (b, s, s, s, 3)
    assert tuple(a_res.shape) == (b, 4, 4)

    # Only affine is batched.
    c_res, a_res = pitn.affine._canonicalize_coords_3d_affine(coords, affine_b)
    assert tuple(c_res.shape) == (b, s, s, s, 3)
    assert tuple(a_res.shape) == (b, 4, 4)

    # Neither is batched, so batch size is 1.
    c_res, a_res = pitn.affine._canonicalize_coords_3d_affine(coords, affine)
    assert tuple(c_res.shape) == (1, s, s, s, 3)
    assert tuple(a_res.shape) == (1, 4, 4)

    # Variants on the coordinates shape.
    # 1D
    c = coords_b[0, 0, 0, 0]
    c_res, a_res = pitn.affine._canonicalize_coords_3d_affine(c, affine_b)
    assert tuple(c_res.shape) == (b, 1, 1, 1, 3)
    assert tuple(a_res.shape) == (b, 4, 4)
    c_res, a_res = pitn.affine._canonicalize_coords_3d_affine(c, affine)
    assert tuple(c_res.shape) == (1, 1, 1, 1, 3)
    assert tuple(a_res.shape) == (1, 4, 4)

    # 2D
    c = coords_b[:, 0, 0, 0, :]
    c_res, a_res = pitn.affine._canonicalize_coords_3d_affine(c, affine_b)
    assert tuple(c_res.shape) == (b, 1, 1, 1, 3)
    assert tuple(a_res.shape) == (b, 4, 4)
    c_res, a_res = pitn.affine._canonicalize_coords_3d_affine(c, affine)
    assert tuple(c_res.shape) == (b, 1, 1, 1, 3)
    assert tuple(a_res.shape) == (b, 4, 4)

    # 3D
    c = coords_b[:, 0, :, 0, :]
    c_res, a_res = pitn.affine._canonicalize_coords_3d_affine(c, affine_b)
    assert tuple(c_res.shape) == (b, s, 1, 1, 3)
    assert tuple(a_res.shape) == (b, 4, 4)
    c_res, a_res = pitn.affine._canonicalize_coords_3d_affine(c, affine)
    assert tuple(c_res.shape) == (b, s, 1, 1, 3)
    assert tuple(a_res.shape) == (b, 4, 4)

    # 4D
    c = coords_b[0]
    c_res, a_res = pitn.affine._canonicalize_coords_3d_affine(c, affine_b)
    assert tuple(c_res.shape) == (b, s, s, s, 3)
    assert tuple(a_res.shape) == (b, 4, 4)
    c_res, a_res = pitn.affine._canonicalize_coords_3d_affine(c, affine)
    assert tuple(c_res.shape) == (1, s, s, s, 3)
    assert tuple(a_res.shape) == (1, 4, 4)

    # Test datatype differences.
    c = coords_b.to(torch.float32)
    a = affine_b.to(torch.float64)
    c_res, a_res = pitn.affine._canonicalize_coords_3d_affine(c, a)
    assert c_res.dtype == torch.float64
    assert a_res.dtype == torch.float64


def test_canonicalize_coords_3d_affine_invalid_inputs():
    # Both batched, different batch sizes.
    b = 13
    affine = torch.eye(4)
    affine_b = einops.repeat(affine, "a_1 a_2 -> b a_1 a_2", b=b)

    s = 10
    coords = torch.stack(
        torch.meshgrid(
            [
                torch.arange(s),
            ]
            * 3,
            indexing="ij",
        ),
        dim=-1,
    )
    coords_b = einops.repeat(coords, "s1 s2 s3 coord -> b s1 s2 s3 coord", b=b)

    c = coords_b[2:-2]
    with pytest.raises(RuntimeError):
        c_res, a_res = pitn.affine._canonicalize_coords_3d_affine(c, affine_b)
    a = affine_b[:-4]
    with pytest.raises(RuntimeError):
        c_res, a_res = pitn.affine._canonicalize_coords_3d_affine(coords_b, a)

    # Test invalid coordinate and affine shapes.
    # 6D coordinates
    c = torch.stack([coords_b, coords_b], dim=0)
    with pytest.raises(RuntimeError):
        c_res, a_res = pitn.affine._canonicalize_coords_3d_affine(c, affine_b)
    # 4D affine
    a = torch.stack([affine_b, affine_b], dim=0)
    with pytest.raises(RuntimeError):
        c_res, a_res = pitn.affine._canonicalize_coords_3d_affine(coords_b, a)
    # 1D affine
    a = affine_b[0, 0]
    with pytest.raises(RuntimeError):
        c_res, a_res = pitn.affine._canonicalize_coords_3d_affine(coords_b, a)
