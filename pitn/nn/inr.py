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
    norm_in_space = (input_space_extent - l_bound) / (0.5 * (u_bound - l_bound))
    norm_in_space = norm_in_space - 1

    norm_target_space_aligned = (target_space_extent - l_bound) / (
        0.5 * (u_bound - l_bound)
    )
    norm_target_space_aligned = norm_target_space_aligned - 1
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
