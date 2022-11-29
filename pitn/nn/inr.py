# -*- coding: utf-8 -*-
import fractions
from typing import Callable, List, Optional

import einops
import monai
import numpy as np
import torch
import torch.nn.functional as F

import pitn


class ResMLP(torch.nn.Module):
    def __init__(self, in_size, out_size, internal_size, n_layers, activate_fn):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.internal_size = internal_size
        self.n_layers = n_layers
        assert self.n_layers >= 3

        if isinstance(activate_fn, str):
            activate_fn = pitn.utils.torch_lookups.activation[activate_fn]
        self.activate_fn = activate_fn

        n_pre = (self.n_layers - 1) // 2
        pre_res_mlp = list()
        pre_res_mlp.append(torch.nn.Linear(self.in_size, self.internal_size))
        pre_res_mlp.append(self.activate_fn())

        for _ in range(1, n_pre):
            pre_res_mlp.append(torch.nn.Linear(self.internal_size, self.internal_size))
            pre_res_mlp.append(self.activate_fn())

        self.pre_res = torch.nn.Sequential(*pre_res_mlp)
        self.res = torch.nn.Linear(self.internal_size, self.internal_size)
        self.activate_callable = self.activate_fn()

        n_post = self.n_layers - n_pre - 1
        post_res_mlp = list()
        for _ in range(1, n_post):
            post_res_mlp.append(torch.nn.Linear(self.internal_size, self.internal_size))
            post_res_mlp.append(self.activate_fn())
        post_res_mlp.append(torch.nn.Linear(self.internal_size, self.out_size))
        self.post_res = torch.nn.Sequential(*post_res_mlp)

    def forward(self, x):

        y = self.pre_res(x)
        y_res = self.res(y)
        y_res = self.activate_callable(y_res)
        y = y + y_res
        y = self.post_res(y)

        return y


class SkipMLPBlock(torch.nn.Module):
    def __init__(
        self,
        n_context_features: int,
        n_coord_features: int,
        n_dense_layers: int,
        activate_fn,
        activate_fn_kwargs=dict(),
        norm_cls=None,
        norm_kwargs=dict(),
    ):
        super().__init__()
        self.n_context_features = n_context_features
        self.n_coord_features = n_coord_features
        if isinstance(activate_fn, str):
            activate_fn = pitn.utils.torch_lookups.activation[activate_fn]

        lins = list()
        for _ in range(1, n_dense_layers + 1):
            lin = torch.nn.Linear(
                in_features=self.n_context_features + self.n_coord_features,
                out_features=self.n_context_features,
            )
            activ = activate_fn(**activate_fn_kwargs)
            norm = norm_cls(**norm_kwargs) if norm_cls is not None else None
            lins.append(
                torch.nn.ModuleDict({"linear": lin, "activate_fn": activ, "norm": norm})
            )

        self.layers = torch.nn.ModuleList(lins)

    def forward(self, x_context, x_coord):
        # Perform forward pass for first linear layer to store skip connection.
        x = torch.cat([x_context, x_coord], dim=1)
        y = self.layers[0]["linear"](x)
        y_res = torch.clone(y)
        y = self.layers[0]["activate_fn"](y)
        if self.layers[0]["norm"] is not None:
            y = self.layers[0]["norm"](y)
        # Iterate through remaining linear layers.
        for i, layer in enumerate(self.layers[1:]):
            i = i + 1
            x_i = torch.cat([y, x_coord], dim=1)
            y = layer["linear"](x_i)
            if i == len(self.layers):
                y = y + y_res
            y = layer["activate_fn"](y)
            if layer["norm"] is not None:
                y = layer["norm"](y)
        # Return both to allow chaining in a Sequential container.
        return y, x_coord


# Decoder models
class SimpleResMLPINR(torch.nn.Module):
    def __init__(
        self,
        n_coord_features: int,
        n_context_features: int,
        # n_context_groups: int,
        out_features: int,
        internal_features: int,
        n_layers: int,
        activate_fn,
        n_vox_size_features: int = 3,
    ):
        super().__init__()

        self.n_coord_features = n_coord_features
        self.n_context_features = n_context_features
        # self.n_context_groups = n_context_groups
        self.n_vox_size_features = n_vox_size_features
        self.out_features = out_features

        # self.in_group_norm = torch.nn.GroupNorm(
        #     self.n_context_groups, self.n_context_features, affine=True
        # )

        self.dense_repr = ResMLP(
            self.n_coord_features + self.n_context_features + self.n_vox_size_features,
            self.out_features,
            internal_size=internal_features,
            n_layers=n_layers,
            activate_fn=activate_fn,
        )

    def forward(
        self,
        query_coord,
        context_v,
        vox_size,
    ):
        # norm_v = self.in_group_norm(context_v)
        norm_v = context_v
        y = torch.cat((norm_v, query_coord, vox_size), dim=1)
        y = self.dense_repr(y)
        return y


def linear_weighted_ctx_v(
    encoded_feat_vol: torch.Tensor,
    input_space_extent: torch.Tensor,
    target_space_extent: torch.Tensor,
    reindex_spatial_extents: bool,
):
    # Affine sampling is given in reverse order (*not* the same as 'xy' indexing, nor
    # the same as 'ij' indexing). If the input spatial extents are given in left->right
    # dimension order, i.e. (vol_dim[0], vol_dim[1], vol_dim[2]), then the spatial
    # extents must be put in right->left order, i.e.,
    # (vol_dim[2], vol_dim[1], vol_dim[0]). This is not documented anywhere, so far as
    # I can tell.
    if reindex_spatial_extents:
        input_space_extent = einops.rearrange(
            input_space_extent, "b c x y z -> b c z y x"
        )
        target_space_extent = einops.rearrange(
            target_space_extent, "b c x y z -> b c z y x"
        )

    # Normalize the input space grid to [-1, 1]
    input_space_extent = input_space_extent.movedim(1, -1)
    target_space_extent = target_space_extent.movedim(1, -1)
    l_bound = torch.amin(input_space_extent, dim=(1, 2, 3), keepdim=True)
    u_bound = torch.amax(input_space_extent, dim=(1, 2, 3), keepdim=True)

    scale = (1 - -1) / (u_bound - l_bound)
    # norm_in_space = (scale * input_space_extent) + -1 - (l_bound * scale)

    # norm_in_space = (input_space_extent - l_bound) / (0.5 * (u_bound - l_bound))
    # norm_in_space = norm_in_space - 1
    norm_target_space_aligned = (
        (scale * target_space_extent)
        + -1
        - (torch.amin(target_space_extent, dim=(1, 2, 3), keepdim=True) * scale)
    )

    # norm_target_space_aligned = (target_space_extent - l_bound) / (
    #     0.5 * (u_bound - l_bound)
    # )
    # norm_target_space_aligned = norm_target_space_aligned - 1
    # Assert/assume that all target coordinates are bounded by the input spatial extent.
    assert (norm_target_space_aligned >= -1).all() and (
        norm_target_space_aligned <= 1
    ).all()

    # Resample the encoded volumetric features according to the normalized to
    # the within-patch coordinates in [-1, 1].
    weighted_feat_vol = F.grid_sample(
        encoded_feat_vol,
        grid=norm_target_space_aligned,
        # The interpolation is labelled as bilinear, but this is actually trilinear.
        mode="bilinear",
        align_corners=True,
    )
    return weighted_feat_vol


def weighted_ctx_v(
    encoded_feat_vol: torch.Tensor,
    input_space_extent: torch.Tensor,
    target_space_extent: torch.Tensor,
    reindex_spatial_extents: bool,
    sample_mode="bilinear",
    align_corners=True,
):
    # See
    # <https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html>
    # for sample_mode options.

    # Normalize the input space grid to [-1, 1]
    input_space_extent = input_space_extent.movedim(1, -1)
    target_space_extent = target_space_extent.movedim(1, -1)
    l_bound = torch.amin(input_space_extent, dim=(1, 2, 3), keepdim=True)
    u_bound = torch.amax(input_space_extent, dim=(1, 2, 3), keepdim=True)

    scale = (1 - -1) / (u_bound - l_bound)
    norm_target_space_aligned = (scale * target_space_extent) + -1 - (scale * l_bound)

    # norm_in_space = (input_space_extent - l_bound) / (0.5 * (u_bound - l_bound))
    # norm_in_space = norm_in_space - 1

    # norm_target_space_aligned = (target_space_extent - l_bound) / (
    #     0.5 * (u_bound - l_bound)
    # )
    # norm_target_space_aligned = norm_target_space_aligned - 1
    # Assert/assume that all target coordinates are bounded by the input spatial extent.
    assert (norm_target_space_aligned >= -1).all() and (
        norm_target_space_aligned <= 1
    ).all()

    # Affine sampling is given in reverse order (*not* the same as 'xy' indexing, nor
    # the same as 'ij' indexing). If the input spatial extents are given in left->right
    # dimension order, i.e. (vol_dim[0], vol_dim[1], vol_dim[2]), then the spatial
    # extents must be put in right->left order, i.e.,
    # (vol_dim[2], vol_dim[1], vol_dim[0]). This is not documented anywhere, so far as
    # I can tell.
    if reindex_spatial_extents:
        # norm_target_space_aligned = einops.rearrange(
        #     norm_target_space_aligned, "b i j k dim -> b k j i dim"
        # )
        encoded_feat_vol = einops.rearrange(encoded_feat_vol, "b c i j k -> b c k j i")

    # Resample the encoded volumetric features according to the normalized to
    # the within-patch coordinates in [-1, 1].
    weighted_feat_vol = F.grid_sample(
        encoded_feat_vol,
        grid=norm_target_space_aligned,
        mode=sample_mode,
        align_corners=align_corners,
    )
    return weighted_feat_vol


def fourier_position_encoding(
    v: torch.Tensor, sigma_scale: torch.Tensor, m_num_freqs: int
) -> torch.Tensor:
    """From the "Positional Encoding" method found in section 6.1 [1].

    [1] M. Tancik et al., “Fourier Features Let Networks Learn High Frequency Functions
    in Low Dimensional Domains.” arXiv, Jun. 18, 2020. doi: 10.48550/arXiv.2006.10739.

    Parameters
    ----------
    v : torch.Tensor
    sigma_scale : torch.Tensor
    m_num_freqs : int

    Returns
    -------
    torch.Tensor
    """

    v = torch.atleast_3d(v)
    sigma_scale = torch.atleast_3d(sigma_scale)
    coeffs = (
        2 * torch.pi * sigma_scale ** (torch.arange(0, m_num_freqs).to(v) / m_num_freqs)
    )
    theta = coeffs * v
    theta = theta.reshape(v.shape[0], -1)
    y = torch.empty(v.shape[0], 2 * theta.shape[1]).to(v)
    y[:, ::2] = torch.cos(theta)
    y[:, 1::2] = torch.sin(theta)

    return y
