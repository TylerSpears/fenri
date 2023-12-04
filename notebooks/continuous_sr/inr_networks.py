# -*- coding: utf-8 -*-
import itertools

import einops
import numpy as np
import torch
import torch.nn.functional as F

import pitn


# Encoding model
class INREncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_coord_channels: bool,
        interior_channels: int,
        out_channels: int,
        n_res_units: int,
        n_dense_units: int,
        activate_fn,
        post_batch_norm: bool = False,
    ):
        super().__init__()

        self.init_kwargs = dict(
            in_channels=in_channels,
            input_coord_channels=input_coord_channels,
            interior_channels=interior_channels,
            out_channels=out_channels,
            n_res_units=n_res_units,
            n_dense_units=n_dense_units,
            activate_fn=activate_fn,
            post_batch_norm=post_batch_norm,
        )

        self.in_channels = in_channels
        # This is just a convenience flag, for now.
        self.input_coord_channels = input_coord_channels
        self.interior_channels = interior_channels
        self.out_channels = out_channels
        self.post_batch_norm = post_batch_norm

        if isinstance(activate_fn, str):
            activate_fn = pitn.utils.torch_lookups.activation[activate_fn]

        self._activation_fn_init = activate_fn
        self.activate_fn = activate_fn()

        # Pad to maintain the same input shape.
        self.pre_conv = torch.nn.Sequential(
            torch.nn.Conv3d(
                self.in_channels,
                self.in_channels,
                kernel_size=1,
                padding="same",
                padding_mode="reflect",
            ),
            self.activate_fn,
            torch.nn.Conv3d(
                self.in_channels,
                self.interior_channels,
                kernel_size=3,
                padding="same",
                padding_mode="reflect",
            ),
        )

        # Construct the densely-connected cascading layers.
        # Create n_dense_units number of dense units.
        top_level_units = list()
        for _ in range(n_dense_units):
            # Create n_res_units number of residual units for every dense unit.
            res_layers = list()
            for _ in range(n_res_units):
                res_layers.append(
                    pitn.nn.layers.ResBlock3dNoBN(
                        self.interior_channels,
                        kernel_size=3,
                        activate_fn=activate_fn,
                        padding="same",
                        padding_mode="reflect",
                    )
                )
            top_level_units.append(
                pitn.nn.layers.DenseCascadeBlock3d(self.interior_channels, *res_layers)
            )

        # Wrap everything into a densely-connected cascade.
        self.cascade = pitn.nn.layers.DenseCascadeBlock3d(
            self.interior_channels, *top_level_units
        )

        self.post_conv = torch.nn.Sequential(
            torch.nn.Conv3d(
                self.interior_channels,
                self.interior_channels,
                kernel_size=5,
                padding="same",
                padding_mode="reflect",
            ),
            self.activate_fn,
            torch.nn.Conv3d(
                self.interior_channels,
                self.out_channels,
                kernel_size=3,
                padding="same",
                padding_mode="reflect",
            ),
            self.activate_fn,
            torch.nn.ReplicationPad3d((1, 0, 1, 0, 1, 0)),
            torch.nn.AvgPool3d(kernel_size=2, stride=1),
            torch.nn.Conv3d(
                self.out_channels,
                self.out_channels,
                kernel_size=1,
                padding="same",
                padding_mode="reflect",
            ),
        )
        if self.post_batch_norm:
            self.output_batch_norm = torch.nn.BatchNorm3d(self.out_channels)
        else:
            self.output_batch_norm = torch.nn.Identity()

    def forward(self, x: torch.Tensor):
        y = self.pre_conv(x)
        y = self.activate_fn(y)
        y = self.cascade(y)
        y = self.activate_fn(y)
        y = self.post_conv(y)
        if self.post_batch_norm:
            y = self.output_batch_norm(y)

        return y


# Encoding model
class BvecEncoder(torch.nn.Module):
    def __init__(
        self,
        in_dwi_channels: int,
        spatial_coord_channels: int,
        interior_channels: int,
        out_channels: int,
        n_res_units: int,
        n_dense_units: int,
        activate_fn,
        post_batch_norm: bool = False,
    ):
        super().__init__()

        self.init_kwargs = dict(
            in_dwi_channels=in_dwi_channels,
            spatial_coord_channels=spatial_coord_channels,
            interior_channels=interior_channels,
            out_channels=out_channels,
            n_res_units=n_res_units,
            n_dense_units=n_dense_units,
            activate_fn=activate_fn,
            post_batch_norm=post_batch_norm,
        )

        self.in_dwi_channels = in_dwi_channels
        self.in_bvec_channels = 3 * self.in_dwi_channels
        self.pre_carn_in_channels = self.in_dwi_channels + self.in_bvec_channels
        self.pre_carn_out_channels = self.in_dwi_channels

        self.spatial_coord_channels = spatial_coord_channels
        self.carn_in_channels = self.pre_carn_out_channels + self.spatial_coord_channels

        self.interior_channels = interior_channels
        self.out_channels = out_channels
        self.post_batch_norm = post_batch_norm

        if isinstance(activate_fn, str):
            activate_fn = pitn.utils.torch_lookups.activation[activate_fn]

        self._activation_fn_init = activate_fn
        self.activate_fn = activate_fn()

        self.dwi_bvec_rearrange = einops.layers.torch.Rearrange(
            "b (m n_dwis) x y z -> b (n_dwis m) x y z", m=4
        )
        self.pre_carn_conv = torch.nn.Sequential(
            torch.nn.Conv3d(
                self.pre_carn_in_channels,
                self.in_dwi_channels * 3,
                kernel_size=1,
                groups=self.in_dwi_channels,
                padding="same",
                padding_mode="reflect",
            ),
            self.activate_fn,
            # torch.nn.Conv3d(
            #     self.in_dwi_channels * 3,
            #     self.in_dwi_channels * 2,
            #     kernel_size=1,
            #     groups=self.in_dwi_channels,
            #     padding="same",
            #     padding_mode="reflect",
            # ),
            # self.activate_fn,
            torch.nn.Conv3d(
                self.in_dwi_channels * 3,
                self.pre_carn_out_channels,
                kernel_size=1,
                groups=self.in_dwi_channels,
                padding="same",
                padding_mode="reflect",
            ),
            self.activate_fn,
        )

        # Pad to maintain the same input shape.
        self.pre_conv = torch.nn.Sequential(
            torch.nn.Conv3d(
                self.carn_in_channels,
                self.carn_in_channels,
                kernel_size=1,
                padding="same",
                padding_mode="reflect",
            ),
            self.activate_fn,
            torch.nn.Conv3d(
                self.carn_in_channels,
                self.interior_channels,
                kernel_size=3,
                padding="same",
                padding_mode="reflect",
            ),
        )

        # Construct the densely-connected cascading layers.
        # Create n_dense_units number of dense units.
        top_level_units = list()
        for _ in range(n_dense_units):
            # Create n_res_units number of residual units for every dense unit.
            res_layers = list()
            for _ in range(n_res_units):
                res_layers.append(
                    pitn.nn.layers.ResBlock3dNoBN(
                        self.interior_channels,
                        kernel_size=3,
                        activate_fn=activate_fn,
                        padding="same",
                        padding_mode="reflect",
                    )
                )
            top_level_units.append(
                pitn.nn.layers.DenseCascadeBlock3d(self.interior_channels, *res_layers)
            )

        # Wrap everything into a densely-connected cascade.
        self.cascade = pitn.nn.layers.DenseCascadeBlock3d(
            self.interior_channels, *top_level_units
        )

        self.post_conv = torch.nn.Sequential(
            torch.nn.Conv3d(
                self.interior_channels,
                self.interior_channels,
                kernel_size=5,
                padding="same",
                padding_mode="reflect",
            ),
            self.activate_fn,
            torch.nn.Conv3d(
                self.interior_channels,
                self.out_channels,
                kernel_size=3,
                padding="same",
                padding_mode="reflect",
            ),
            self.activate_fn,
            torch.nn.ReplicationPad3d((1, 0, 1, 0, 1, 0)),
            torch.nn.AvgPool3d(kernel_size=2, stride=1),
            torch.nn.Conv3d(
                self.out_channels,
                self.out_channels,
                kernel_size=1,
                padding="same",
                padding_mode="reflect",
            ),
        )
        if self.post_batch_norm:
            self.output_batch_norm = torch.nn.BatchNorm3d(self.out_channels)
        else:
            self.output_batch_norm = torch.nn.Identity()

    def forward(
        self,
        x_dwi: torch.Tensor,
        x_bvec: torch.Tensor,
        x_spatial_coord: torch.Tensor,
        x_mask=None,
    ):
        # Expand bvec to match the spatial shape of the dwi input.
        x_bvec = einops.repeat(
            x_bvec,
            "b coord n_dwi -> b (coord n_dwi) x y z",
            coord=3,
            x=x_dwi.shape[2],
            y=x_dwi.shape[3],
            z=x_dwi.shape[4],
        )
        # Concat dwi and b-vectors, reduce down to the same number of channels as the
        # original dwis.
        x_dwi_reduce = self.dwi_bvec_rearrange(torch.cat([x_dwi, x_bvec], dim=1))
        if x_mask is not None:
            x_dwi_reduce *= x_mask
        y = self.pre_carn_conv(x_dwi_reduce)

        x_spatial_coord = einops.rearrange(
            x_spatial_coord, "b x y z coord -> b coord x y z"
        )
        y = torch.cat([y, x_spatial_coord], dim=1)
        if x_mask is not None:
            y *= x_mask

        y = self.pre_conv(y)
        y = self.activate_fn(y)
        y = self.cascade(y)
        y = self.activate_fn(y)
        y = self.post_conv(y)
        if self.post_batch_norm:
            y = self.output_batch_norm(y)

        return y


class Decoder(torch.nn.Module):
    def __init__(
        self,
        context_v_features: int,
        out_features: int,
        m_encode_num_freqs: int,
        sigma_encode_scale: float,
        in_features=None,
    ):
        super().__init__()
        self.init_kwargs = dict(
            context_v_features=context_v_features,
            out_features=out_features,
            m_encode_num_freqs=m_encode_num_freqs,
            sigma_encode_scale=sigma_encode_scale,
            in_features=in_features,
        )

        # Determine the number of input features needed for the MLP.
        # The order for concatenation is
        # 1) ctx feats over the low-res input space, unfolded over a 3x3x3 window
        # ~~2) target voxel shape~~
        # 3) absolute coords of this forward pass' prediction target
        # 4) absolute coords of the high-res target voxel
        # ~~5) relative coords between high-res target coords and this forward pass'
        #    prediction target, normalized by low-res voxel shape~~
        # 6) encoding of relative coords
        self.context_v_features = context_v_features
        self.ndim = 3
        self.m_encode_num_freqs = m_encode_num_freqs
        self.sigma_encode_scale = torch.as_tensor(sigma_encode_scale)
        self.n_encode_features = self.ndim * 2 * self.m_encode_num_freqs
        self.n_coord_features = 2 * self.ndim + self.n_encode_features
        self.internal_features = self.context_v_features + self.n_coord_features

        self.in_features = in_features
        self.out_features = out_features

        # "Swish" function, recommended in MeshFreeFlowNet
        activate_cls = torch.nn.SiLU
        self.activate_fn = activate_cls(inplace=True)
        # Optional resizing linear layer, if the input size should be different than
        # the hidden layer size.
        if self.in_features is not None:
            self.lin_pre = torch.nn.Linear(self.in_features, self.context_v_features)
            self.norm_pre = None
        else:
            self.lin_pre = None
            self.norm_pre = None
        self.norm_pre = None

        # Internal hidden layers are two res MLPs.
        self.internal_res_repr = torch.nn.ModuleList(
            [
                pitn.nn.inr.SkipMLPBlock(
                    n_context_features=self.context_v_features,
                    n_coord_features=self.n_coord_features,
                    n_dense_layers=3,
                    activate_fn=activate_cls,
                )
                for _ in range(2)
            ]
        )
        self.lin_post = torch.nn.Linear(self.context_v_features, self.out_features)

    def encode_relative_coord(self, coords):
        c = einops.rearrange(coords, "b n coord -> (b n) coord")
        sigma = self.sigma_encode_scale.expand_as(c).to(c)[..., None]
        encode_pos = pitn.nn.inr.fourier_position_encoding(
            c, sigma_scale=sigma, m_num_freqs=self.m_encode_num_freqs
        )
        encode_pos = einops.rearrange(
            encode_pos, "(b n) freqs -> b n freqs", n=coords.shape[1]
        )
        return encode_pos

    def sub_grid_forward(
        self,
        context_v,
        context_world_coord,
        query_world_coord,
        query_world_coord_mask,
        query_sub_grid_coord,
        context_sub_grid_coord,
    ):
        # Take relative coordinate difference between the current context
        # coord and the query coord, given in normalized ([0, 1]) sub-grid coordinates.
        batch_size = context_v.shape[0]
        rel_q2ctx_norm_sub_grid_coord = (
            query_sub_grid_coord - context_sub_grid_coord + 1
        ) / 2

        # assert (rel_q2ctx_norm_sub_grid_coord >= 0).all() and (
        #     rel_q2ctx_norm_sub_grid_coord <= 1.0
        # ).all()
        encoded_rel_q2ctx_coord = self.encode_relative_coord(
            rel_q2ctx_norm_sub_grid_coord
        )
        # b n 1 -> b 1 n
        context_v = context_v * query_world_coord_mask[:, None, :, 0]
        # Perform forward pass of the MLP.
        if self.norm_pre is not None:
            context_v = self.norm_pre(context_v)
        # Group batches and queries-per-batch into just batches, keep context channels
        # as the feature vector.
        context_feats = einops.rearrange(context_v, "b channels n -> (b n) channels")
        feat_mask = einops.rearrange(query_world_coord_mask, "b n 1 -> (b n) 1")
        coord_feats = (
            context_world_coord,
            query_world_coord,
            encoded_rel_q2ctx_coord,
        )
        coord_feats = torch.cat(coord_feats, dim=-1)

        coord_feats = einops.rearrange(coord_feats, "b n coord -> (b n) coord")
        x_coord = coord_feats * feat_mask
        y_sub_grid_pred = context_feats * feat_mask

        if self.lin_pre is not None:
            y_sub_grid_pred = self.lin_pre(y_sub_grid_pred)
            y_sub_grid_pred = self.activate_fn(y_sub_grid_pred)

        for l in self.internal_res_repr:
            y_sub_grid_pred, x_coord = l(y_sub_grid_pred, x_coord)
        # The SkipMLPBlock contains the residual addition, so no need to add here.
        y_sub_grid_pred = self.lin_post(y_sub_grid_pred)
        y_sub_grid_pred = einops.rearrange(
            y_sub_grid_pred, "(b n) channels -> b channels n", b=batch_size
        )

        return y_sub_grid_pred

    def forward(
        self,
        context_v: torch.Tensor,
        context_world_coord_grid: torch.Tensor,
        query_world_coord: torch.Tensor,
        query_world_coord_mask: torch.Tensor,
        affine_context_vox2world: torch.Tensor,
        affine_query_vox2world: torch.Tensor,
        context_vox_size_world: torch.Tensor,
        query_vox_size_world: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = context_v.shape[0]
        query_orig_shape = tuple(query_world_coord.shape)
        # All coords and coord grids must be *coordinate-last* format (similar to
        # channel-last).
        q_world = einops.rearrange(
            query_world_coord, "b ... c -> b (...) c", b=batch_size, c=3
        )
        n_q_per_batch = q_world.shape[1]
        # Query mask should be b x y z 1
        q_mask = einops.rearrange(query_world_coord_mask, "b ... 1 -> b (...) 1")
        # Replace each unmasked (i.e., invalid) query point by a dummy point, in this
        # case a point from roughly the middle of each batch of query points, which
        # should be as safe as possible from being out of bounds wrt the context vox
        # indices.
        dummy_q_world = q_world[:, (n_q_per_batch // 2)]
        dummy_q_world.unsqueeze_(1)
        # Replace all unmasked q world coords with the dummy coord.
        q_world = torch.where(q_mask, q_world, dummy_q_world)

        affine_world2ctx_vox = torch.linalg.inv(affine_context_vox2world)
        q_ctx_vox = pitn.affine.transform_coords(q_world, affine_world2ctx_vox)
        q_ctx_vox_bottom = q_ctx_vox.floor().long()
        # The vox coordinates are not broadcast over every batch (like they are over
        # every channel), so we need a batch idx to associate each sub-grid voxel with
        # the appropriate batch index.
        batch_vox_idx = einops.repeat(
            torch.arange(
                batch_size,
                dtype=q_ctx_vox_bottom.dtype,
                device=q_ctx_vox_bottom.device,
            ),
            "batch -> batch n",
            n=n_q_per_batch,
        )
        q_sub_grid_coord = q_ctx_vox - q_ctx_vox_bottom.to(q_ctx_vox)
        # q_bottom_in_world_coord = pitn.affine.transform_coords(
        #     q_ctx_vox_bottom.to(affine_context_vox2world), affine_context_vox2world
        # )

        y_weighted_accumulate = None
        # Build the low-res representation one sub-grid voxel index at a time.
        # Each sub-grid is a [0, 1] voxel coordinate system local to the query point,
        # where the origin is the context voxel that is "lower" in all dimensions
        # than the query coordinate.
        # The indicators specify if the current voxel index that surrounds the
        # query coordinate should be "off  or not. If not, then
        # the center voxel (read: no voxel offset from the center) is selected
        # (for that dimension).
        sub_grid_offset_ijk = q_ctx_vox_bottom.new_zeros(1, 1, 3)
        for (
            corner_offset_i,
            corner_offset_j,
            corner_offset_k,
        ) in itertools.product((0, 1), (0, 1), (0, 1)):
            # Rebuild indexing tuple for each element of the sub-window
            sub_grid_offset_ijk[..., 0] = corner_offset_i
            sub_grid_offset_ijk[..., 1] = corner_offset_j
            sub_grid_offset_ijk[..., 2] = corner_offset_k
            sub_grid_index_ijk = q_ctx_vox_bottom + sub_grid_offset_ijk

            sub_grid_context_v = context_v[
                batch_vox_idx.flatten(),
                :,
                sub_grid_index_ijk[..., 0].flatten(),
                sub_grid_index_ijk[..., 1].flatten(),
                sub_grid_index_ijk[..., 2].flatten(),
            ]
            sub_grid_context_v = einops.rearrange(
                sub_grid_context_v,
                "(b n) channels -> b channels n",
                b=batch_size,
                n=n_q_per_batch,
            )
            sub_grid_context_world_coord = context_world_coord_grid[
                batch_vox_idx.flatten(),
                sub_grid_index_ijk[..., 0].flatten(),
                sub_grid_index_ijk[..., 1].flatten(),
                sub_grid_index_ijk[..., 2].flatten(),
                :,
            ]
            sub_grid_context_world_coord = einops.rearrange(
                sub_grid_context_world_coord,
                "(b n) coords -> b n coords",
                b=batch_size,
                n=n_q_per_batch,
            )

            sub_grid_pred_ijk = self.sub_grid_forward(
                context_v=sub_grid_context_v,
                context_world_coord=sub_grid_context_world_coord,
                query_world_coord=q_world,
                query_sub_grid_coord=q_sub_grid_coord,
                query_world_coord_mask=q_mask,
                context_sub_grid_coord=sub_grid_offset_ijk,
            )
            # Initialize the accumulated prediction after finding the
            # output size; easier than trying to pre-compute the shape.
            if y_weighted_accumulate is None:
                y_weighted_accumulate = torch.zeros_like(sub_grid_pred_ijk)

            sub_grid_offset_ijk_compliment = torch.abs(1 - sub_grid_offset_ijk)
            sub_grid_context_vox_coord_compliment = (
                q_ctx_vox_bottom + sub_grid_offset_ijk_compliment
            )
            w_sub_grid_cube_ijk = torch.abs(
                sub_grid_context_vox_coord_compliment - q_ctx_vox
            )
            # Each coordinate difference is a side of the cube, so find the volume.
            w_ijk = einops.reduce(
                w_sub_grid_cube_ijk, "b n coord -> b 1 n", reduction="prod"
            )

            # Accumulate weighted cell predictions to eventually create
            # the final prediction.
            y_weighted_accumulate += w_ijk * sub_grid_pred_ijk

        y = y_weighted_accumulate

        out_channels = y.shape[1]
        # Reshape prediction to match the input query coordinates spatial shape.
        q_in_within_batch_samples_shape = query_orig_shape[1:-1]
        y = y.reshape(*((batch_size, out_channels) + q_in_within_batch_samples_shape))

        return y


class SimplifiedDecoder(torch.nn.Module):
    def __init__(
        self,
        context_v_features: int,
        out_features: int,
        m_encode_num_freqs: int,
        n_internal_features: int,
        n_internal_layers: int,
        sigma_encode_scale: float,
    ):
        super().__init__()
        self.init_kwargs = dict(
            context_v_features=context_v_features,
            out_features=out_features,
            m_encode_num_freqs=m_encode_num_freqs,
            sigma_encode_scale=sigma_encode_scale,
            n_internal_layers=n_internal_layers,
            n_internal_features=n_internal_features,
        )

        # Determine the number of input features needed for the MLP.
        # The order for concatenation is
        # 1) ctx feats over the low-res input space, unfolded over a 3x3x3 window
        # ~~2) target voxel shape~~
        # 3) absolute coords of this forward pass' prediction target
        # 4) absolute coords of the high-res target voxel
        # ~~5) relative coords between high-res target coords and this forward pass'
        #    prediction target, normalized by low-res voxel shape~~
        # 6) encoding of relative coords
        self.context_v_features = context_v_features
        self.ndim = 3
        self.m_encode_num_freqs = m_encode_num_freqs
        self.sigma_encode_scale = torch.as_tensor(sigma_encode_scale)
        self.n_encode_features = self.ndim * 2 * self.m_encode_num_freqs
        self.n_coord_features = 2 * self.ndim + self.n_encode_features
        self.n_internal_layers = n_internal_layers

        self.input_features = self.context_v_features + self.n_coord_features
        self.internal_features = n_internal_features
        self.out_features = out_features

        # "Swish" function, recommended in MeshFreeFlowNet
        activate_cls = torch.nn.SiLU
        self.activate_fn = activate_cls(inplace=True)

        lins = list()
        lins.append(torch.nn.Linear(self.input_features, self.internal_features))
        lins.append(activate_cls(inplace=True))

        for _ in range(self.n_internal_layers):
            lins.append(torch.nn.Linear(self.internal_features, self.internal_features))
            lins.append(activate_cls(inplace=True))
        lins.append(torch.nn.Linear(self.internal_features, self.out_features))

        self.linear_layers = torch.nn.Sequential(*lins)

    def encode_relative_coord(self, coords):
        c = einops.rearrange(coords, "b n coord -> (b n) coord")
        sigma = self.sigma_encode_scale.expand_as(c).to(c)[..., None]
        encode_pos = pitn.nn.inr.fourier_position_encoding(
            c, sigma_scale=sigma, m_num_freqs=self.m_encode_num_freqs
        )
        encode_pos = einops.rearrange(
            encode_pos, "(b n) freqs -> b n freqs", n=coords.shape[1]
        )
        return encode_pos

    def sub_grid_forward(
        self,
        context_v,
        context_world_coord,
        query_world_coord,
        query_world_coord_mask,
        query_sub_grid_coord,
        context_sub_grid_coord,
    ):
        # Take relative coordinate difference between the current context
        # coord and the query coord, given in normalized ([0, 1]) sub-grid coordinates.
        batch_size = context_v.shape[0]
        rel_q2ctx_norm_sub_grid_coord = (
            query_sub_grid_coord - context_sub_grid_coord + 1
        ) / 2

        # assert (rel_q2ctx_norm_sub_grid_coord >= 0).all() and (
        #     rel_q2ctx_norm_sub_grid_coord <= 1.0
        # ).all()
        encoded_rel_q2ctx_coord = self.encode_relative_coord(
            rel_q2ctx_norm_sub_grid_coord
        )
        # b n 1 -> b 1 n
        context_v = context_v * query_world_coord_mask[:, None, :, 0]
        # Perform forward pass of the MLP.
        # Group batches and queries-per-batch into just batches, keep context channels
        # as the feature vector.
        feat_mask = einops.rearrange(query_world_coord_mask, "b n 1 -> (b n) 1")
        context_feats = einops.rearrange(context_v, "b channels n -> (b n) channels")
        context_feats *= feat_mask

        coord_feats = (
            context_world_coord,
            query_world_coord,
            encoded_rel_q2ctx_coord,
        )
        coord_feats = torch.cat(coord_feats, dim=-1)
        coord_feats = einops.rearrange(coord_feats, "b n coord -> (b n) coord")
        coord_feats *= feat_mask

        y_sub_grid_pred = torch.cat([context_feats, coord_feats], dim=1)

        del (
            context_v,
            context_feats,
            feat_mask,
            coord_feats,
            rel_q2ctx_norm_sub_grid_coord,
            encoded_rel_q2ctx_coord,
        )

        y_sub_grid_pred = self.linear_layers(y_sub_grid_pred)
        y_sub_grid_pred = einops.rearrange(
            y_sub_grid_pred, "(b n) channels -> b channels n", b=batch_size
        )

        return y_sub_grid_pred

    def forward(
        self,
        context_v: torch.Tensor,
        context_real_coords: torch.Tensor,
        query_real_coords: torch.Tensor,
        query_coords_mask: torch.Tensor,
        affine_context_vox2real: torch.Tensor,
        context_spacing: torch.Tensor,
        query_spacing: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = context_v.shape[0]
        q_orig_shape = tuple(query_real_coords.shape)
        # All coords and coord grids must be *coordinate-last* format (similar to
        # channel-last).
        q_world = einops.rearrange(
            query_real_coords, "b ... c -> b (...) c", b=batch_size, c=3
        )
        n_q_per_batch = q_world.shape[1]
        # Input query coord mask should be b x y z 1
        q_mask = einops.rearrange(query_coords_mask, "b ... 1 -> b (...) 1").bool()
        # Replace each unmasked (i.e., invalid) query point by a dummy point, in this
        # case a point from roughly the middle of each batch of query points, which
        # should be as safe as possible from being out of bounds wrt the context vox
        # indices.
        dummy_q_world = q_world[:, (n_q_per_batch // 2)]
        dummy_q_world.unsqueeze_(1)
        # Replace all unmasked q world coords with the dummy coord.
        q_world = torch.where(q_mask, q_world, dummy_q_world)
        q_world_homog = torch.cat(
            [q_world, q_world.new_ones(q_world.shape[0], q_world.shape[1], 1)], dim=-1
        )
        q_world_homog = einops.rearrange(q_world_homog, "b v c -> b c v")
        # Cast both coordinates and affines to 64-bit precision, then truncate back to
        # single precision.
        q_ctx_vox = (
            torch.linalg.solve(
                affine_context_vox2real.to(torch.float64),
                q_world_homog.to(torch.float64),
                left=True,
            )
            .round(decimals=6)
            .to(torch.float32)
        )
        q_ctx_vox = einops.rearrange(q_ctx_vox[..., :-1, :], "b c v -> b v c")
        # If a coordinate is on the edge of the input grid, then the affine transform
        # may introduce numerical errors that bring it out of bounds. So, for those
        # points at the edge that are sufficiently close to the edge, just clamp
        # those values. Also, if there are true out-of-bounds coordinates, allow those
        # through.
        min_q_vox = q_ctx_vox.new_zeros(1)
        max_q_vox = torch.as_tensor(context_v.shape[2:]).reshape(1, 1, -1).to(q_ctx_vox)
        q_ctx_vox_clamped = torch.clamp(q_ctx_vox, min=min_q_vox, max=max_q_vox)
        tol_vox_coord = 1e-5
        q_ctx_vox = torch.where(
            (q_ctx_vox != q_ctx_vox_clamped)
            & (torch.abs(q_ctx_vox - q_ctx_vox_clamped) < tol_vox_coord),
            q_ctx_vox_clamped,
            q_ctx_vox,
        )
        del q_world_homog, dummy_q_world, q_ctx_vox_clamped
        # affine_ctx_real2vox = pitn.affine.inv_affine(
        #     affine_context_vox2real, rounding_decimals=6
        # )
        # affine_world2ctx_vox = torch.linalg.inv(affine_context_vox2world)
        # Transform query real-space coordinates into context voxel coordinates.
        # q_ctx_vox = pitn.affine.transform_coords(q_world, affine_ctx_real2vox)
        q_ctx_vox_bottom = q_ctx_vox.floor().long()
        # The vox coordinates are not broadcast over every batch (like they are over
        # every channel), so we need a batch idx to associate each sub-grid voxel with
        # the appropriate batch index.
        batch_vox_idx = einops.repeat(
            torch.arange(
                batch_size,
                dtype=q_ctx_vox_bottom.dtype,
                device=q_ctx_vox_bottom.device,
            ),
            "batch -> batch n",
            n=n_q_per_batch,
        )
        q_sub_grid_coord = q_ctx_vox - q_ctx_vox_bottom.to(q_ctx_vox)
        # q_bottom_in_world_coord = pitn.affine.transform_coords(
        #     q_ctx_vox_bottom.to(affine_context_vox2world), affine_context_vox2world
        # )

        y_weighted_accumulate = None
        # Build the low-res representation one sub-grid voxel index at a time.
        # Each sub-grid is a [0, 1] voxel coordinate system local to the query point,
        # where the origin is the context voxel that is "lower" in all dimensions
        # than the query coordinate.
        # The indicators specify if the current voxel index that surrounds the
        # query coordinate should be "off  or not. If not, then
        # the center voxel (read: no voxel offset from the center) is selected
        # (for that dimension).
        sub_grid_offset_ijk = q_ctx_vox_bottom.new_zeros(1, 1, 3)
        for (
            corner_offset_i,
            corner_offset_j,
            corner_offset_k,
        ) in itertools.product((0, 1), (0, 1), (0, 1)):
            # Rebuild indexing tuple for each element of the sub-window
            sub_grid_offset_ijk[..., 0] = corner_offset_i
            sub_grid_offset_ijk[..., 1] = corner_offset_j
            sub_grid_offset_ijk[..., 2] = corner_offset_k
            sub_grid_index_ijk = q_ctx_vox_bottom + sub_grid_offset_ijk

            sub_grid_context_v = context_v[
                batch_vox_idx.flatten(),
                :,
                sub_grid_index_ijk[..., 0].flatten(),
                sub_grid_index_ijk[..., 1].flatten(),
                sub_grid_index_ijk[..., 2].flatten(),
            ]
            sub_grid_context_v = einops.rearrange(
                sub_grid_context_v,
                "(b n) channels -> b channels n",
                b=batch_size,
                n=n_q_per_batch,
            )
            assert (sub_grid_index_ijk >= 0).all()
            sub_grid_context_real_coords = context_real_coords[
                batch_vox_idx.flatten(),
                sub_grid_index_ijk[..., 0].flatten(),
                sub_grid_index_ijk[..., 1].flatten(),
                sub_grid_index_ijk[..., 2].flatten(),
                :,
            ]
            sub_grid_context_real_coords = einops.rearrange(
                sub_grid_context_real_coords,
                "(b n) coords -> b n coords",
                b=batch_size,
                n=n_q_per_batch,
            )

            sub_grid_pred_ijk = self.sub_grid_forward(
                context_v=sub_grid_context_v,
                context_world_coord=sub_grid_context_real_coords,
                query_world_coord=q_world,
                query_sub_grid_coord=q_sub_grid_coord,
                query_world_coord_mask=q_mask,
                context_sub_grid_coord=sub_grid_offset_ijk,
            )
            # Initialize the accumulated prediction after finding the
            # output size; easier than trying to pre-compute the shape.
            if y_weighted_accumulate is None:
                y_weighted_accumulate = torch.zeros_like(sub_grid_pred_ijk)

            sub_grid_offset_ijk_compliment = torch.abs(1 - sub_grid_offset_ijk)
            sub_grid_context_vox_coord_compliment = (
                q_ctx_vox_bottom + sub_grid_offset_ijk_compliment
            )
            w_sub_grid_cube_ijk = torch.abs(
                sub_grid_context_vox_coord_compliment - q_ctx_vox
            )
            # Each coordinate difference is a side of the cube, so find the volume.
            w_ijk = einops.reduce(
                w_sub_grid_cube_ijk, "b n coord -> b 1 n", reduction="prod"
            )

            # Accumulate weighted cell predictions to eventually create
            # the final prediction.
            y_weighted_accumulate += w_ijk * sub_grid_pred_ijk

        y = y_weighted_accumulate

        out_channels = y.shape[1]
        # Reshape prediction to match the input query coordinates spatial shape.
        q_in_within_batch_samples_shape = q_orig_shape[1:-1]
        y = y.reshape(*((batch_size, out_channels) + q_in_within_batch_samples_shape))

        return y

    def _legacy_forward(
        self,
        context_v: torch.Tensor,
        context_world_coord_grid: torch.Tensor,
        query_world_coord: torch.Tensor,
        query_world_coord_mask: torch.Tensor,
        affine_context_vox2world: torch.Tensor,
        affine_query_vox2world: torch.Tensor,
        context_vox_size_world: torch.Tensor,
        query_vox_size_world: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = context_v.shape[0]
        query_orig_shape = tuple(query_world_coord.shape)
        # All coords and coord grids must be *coordinate-last* format (similar to
        # channel-last).
        q_world = einops.rearrange(
            query_world_coord, "b ... c -> b (...) c", b=batch_size, c=3
        )
        n_q_per_batch = q_world.shape[1]
        # Query mask should be b x y z 1
        q_mask = einops.rearrange(query_world_coord_mask, "b ... 1 -> b (...) 1")
        # Replace each unmasked (i.e., invalid) query point by a dummy point, in this
        # case a point from roughly the middle of each batch of query points, which
        # should be as safe as possible from being out of bounds wrt the context vox
        # indices.
        dummy_q_world = q_world[:, (n_q_per_batch // 2)]
        dummy_q_world.unsqueeze_(1)
        # Replace all unmasked q world coords with the dummy coord.
        q_world = torch.where(q_mask, q_world, dummy_q_world)

        affine_world2ctx_vox = torch.linalg.inv(affine_context_vox2world)
        q_ctx_vox = pitn.affine.transform_coords(q_world, affine_world2ctx_vox)
        q_ctx_vox_bottom = q_ctx_vox.floor().long()
        # The vox coordinates are not broadcast over every batch (like they are over
        # every channel), so we need a batch idx to associate each sub-grid voxel with
        # the appropriate batch index.
        batch_vox_idx = einops.repeat(
            torch.arange(
                batch_size,
                dtype=q_ctx_vox_bottom.dtype,
                device=q_ctx_vox_bottom.device,
            ),
            "batch -> batch n",
            n=n_q_per_batch,
        )
        q_sub_grid_coord = q_ctx_vox - q_ctx_vox_bottom.to(q_ctx_vox)
        # q_bottom_in_world_coord = pitn.affine.transform_coords(
        #     q_ctx_vox_bottom.to(affine_context_vox2world), affine_context_vox2world
        # )

        y_weighted_accumulate = None
        # Build the low-res representation one sub-grid voxel index at a time.
        # Each sub-grid is a [0, 1] voxel coordinate system local to the query point,
        # where the origin is the context voxel that is "lower" in all dimensions
        # than the query coordinate.
        # The indicators specify if the current voxel index that surrounds the
        # query coordinate should be "off  or not. If not, then
        # the center voxel (read: no voxel offset from the center) is selected
        # (for that dimension).
        sub_grid_offset_ijk = q_ctx_vox_bottom.new_zeros(1, 1, 3)
        for (
            corner_offset_i,
            corner_offset_j,
            corner_offset_k,
        ) in itertools.product((0, 1), (0, 1), (0, 1)):
            # Rebuild indexing tuple for each element of the sub-window
            sub_grid_offset_ijk[..., 0] = corner_offset_i
            sub_grid_offset_ijk[..., 1] = corner_offset_j
            sub_grid_offset_ijk[..., 2] = corner_offset_k
            sub_grid_index_ijk = q_ctx_vox_bottom + sub_grid_offset_ijk

            sub_grid_context_v = context_v[
                batch_vox_idx.flatten(),
                :,
                sub_grid_index_ijk[..., 0].flatten(),
                sub_grid_index_ijk[..., 1].flatten(),
                sub_grid_index_ijk[..., 2].flatten(),
            ]
            sub_grid_context_v = einops.rearrange(
                sub_grid_context_v,
                "(b n) channels -> b channels n",
                b=batch_size,
                n=n_q_per_batch,
            )
            sub_grid_context_world_coord = context_world_coord_grid[
                batch_vox_idx.flatten(),
                sub_grid_index_ijk[..., 0].flatten(),
                sub_grid_index_ijk[..., 1].flatten(),
                sub_grid_index_ijk[..., 2].flatten(),
                :,
            ]
            sub_grid_context_world_coord = einops.rearrange(
                sub_grid_context_world_coord,
                "(b n) coords -> b n coords",
                b=batch_size,
                n=n_q_per_batch,
            )

            sub_grid_pred_ijk = self.sub_grid_forward(
                context_v=sub_grid_context_v,
                context_world_coord=sub_grid_context_world_coord,
                query_world_coord=q_world,
                query_sub_grid_coord=q_sub_grid_coord,
                query_world_coord_mask=q_mask,
                context_sub_grid_coord=sub_grid_offset_ijk,
            )
            # Initialize the accumulated prediction after finding the
            # output size; easier than trying to pre-compute the shape.
            if y_weighted_accumulate is None:
                y_weighted_accumulate = torch.zeros_like(sub_grid_pred_ijk)

            sub_grid_offset_ijk_compliment = torch.abs(1 - sub_grid_offset_ijk)
            sub_grid_context_vox_coord_compliment = (
                q_ctx_vox_bottom + sub_grid_offset_ijk_compliment
            )
            w_sub_grid_cube_ijk = torch.abs(
                sub_grid_context_vox_coord_compliment - q_ctx_vox
            )
            # Each coordinate difference is a side of the cube, so find the volume.
            w_ijk = einops.reduce(
                w_sub_grid_cube_ijk, "b n coord -> b 1 n", reduction="prod"
            )

            # Accumulate weighted cell predictions to eventually create
            # the final prediction.
            y_weighted_accumulate += w_ijk * sub_grid_pred_ijk

        y = y_weighted_accumulate

        out_channels = y.shape[1]
        # Reshape prediction to match the input query coordinates spatial shape.
        q_in_within_batch_samples_shape = query_orig_shape[1:-1]
        y = y.reshape(*((batch_size, out_channels) + q_in_within_batch_samples_shape))

        return y


class FixedUpsampleEncoder(torch.nn.Module):
    def __init__(
        self,
        spatial_upscale_factor: float,
        in_channels: int,
        input_coord_channels: bool,
        interior_channels: int,
        out_channels: int,
        n_res_units: int,
        n_dense_units: int,
        activate_fn,
        post_batch_norm: bool = False,
    ):
        super().__init__()

        self.init_kwargs = dict(
            in_channels=in_channels,
            input_coord_channels=input_coord_channels,
            interior_channels=interior_channels,
            out_channels=out_channels,
            n_res_units=n_res_units,
            n_dense_units=n_dense_units,
            activate_fn=activate_fn,
            spatial_upscale_factor=spatial_upscale_factor,
            post_batch_norm=post_batch_norm,
        )

        self.upscale_factor = spatial_upscale_factor
        self.espcn_upscale_factor = int(np.ceil(self.upscale_factor))
        self.post_espcn_downscale = 1 / self.espcn_upscale_factor * self.upscale_factor
        self.in_channels = in_channels
        # This is just a convenience flag, for now.
        self.input_coord_channels = input_coord_channels
        self.interior_channels = interior_channels
        self.out_channels = out_channels
        self.post_batch_norm = post_batch_norm

        if isinstance(activate_fn, str):
            activate_fn = pitn.utils.torch_lookups.activation[activate_fn]

        self._activation_fn_init = activate_fn
        self.activate_fn = activate_fn()

        # Pad to maintain the same input shape.
        self.pre_conv = torch.nn.Sequential(
            torch.nn.Conv3d(
                self.in_channels,
                self.in_channels,
                kernel_size=1,
                padding="same",
                padding_mode="reflect",
            ),
            self.activate_fn,
            torch.nn.Conv3d(
                self.in_channels,
                self.interior_channels,
                kernel_size=3,
                padding="same",
                padding_mode="reflect",
            ),
        )

        # Construct the densely-connected cascading layers.
        # Create n_dense_units number of dense units.
        top_level_units = list()
        for _ in range(n_dense_units):
            # Create n_res_units number of residual units for every dense unit.
            res_layers = list()
            for _ in range(n_res_units):
                res_layers.append(
                    pitn.nn.layers.ResBlock3dNoBN(
                        self.interior_channels,
                        kernel_size=3,
                        activate_fn=activate_fn,
                        padding="same",
                        padding_mode="reflect",
                    )
                )
            top_level_units.append(
                pitn.nn.layers.DenseCascadeBlock3d(self.interior_channels, *res_layers)
            )

        # Wrap everything into a densely-connected cascade.
        self.cascade = pitn.nn.layers.DenseCascadeBlock3d(
            self.interior_channels, *top_level_units
        )
        self.upsampler = torch.nn.Sequential(
            pitn.nn.layers.upsample.ICNRUpsample3d(
                in_channels=self.interior_channels,
                out_channels=self.out_channels,
                upscale_factor=self.espcn_upscale_factor,
                activate_fn=activate_fn,
                blur=True,
                zero_bias=True,
            ),
            torch.nn.Conv3d(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                padding="same",
                padding_mode="reflect",
            ),
        )
        self.post_sampler = torch.nn.Conv3d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            padding="same",
            padding_mode="reflect",
        )
        if self.post_batch_norm:
            self.output_batch_norm = torch.nn.BatchNorm3d(self.interior_channels)
        else:
            self.output_batch_norm = torch.nn.Identity()

    def forward(self, x: torch.Tensor, upsample=True):
        y = self.pre_conv(x)
        y = self.activate_fn(y)
        y = self.cascade(y)
        y = self.activate_fn(y)
        if self.post_batch_norm:
            y = self.output_batch_norm(y)
        if upsample:
            y = self.upsampler(y)
            y = torch.nn.functional.interpolate(
                y,
                scale_factor=self.post_espcn_downscale,
                mode="trilinear",
                align_corners=False,
            )
            y = self.activate_fn(y)
            y = self.post_sampler(y)

        return y


class FixedDecoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        interior_channels: int,
        out_channels: int,
        n_res_units: int,
        n_dense_units: int,
        activate_fn,
    ):
        super().__init__()

        self.init_kwargs = dict(
            in_channels=in_channels,
            interior_channels=interior_channels,
            out_channels=out_channels,
            n_res_units=n_res_units,
            n_dense_units=n_dense_units,
            activate_fn=activate_fn,
        )

        self.in_channels = in_channels
        self.interior_channels = interior_channels
        self.out_channels = out_channels

        if isinstance(activate_fn, str):
            activate_fn = pitn.utils.torch_lookups.activation[activate_fn]

        self._activation_fn_init = activate_fn
        self.activate_fn = activate_fn()

        # Pad to maintain the same input shape.
        self.pre_conv = torch.nn.Sequential(
            torch.nn.Conv3d(
                self.in_channels,
                self.in_channels,
                kernel_size=1,
                padding="same",
                padding_mode="reflect",
            ),
            self.activate_fn,
            torch.nn.Conv3d(
                self.in_channels,
                self.interior_channels,
                kernel_size=3,
                padding="same",
                padding_mode="reflect",
            ),
        )

        # Construct the densely-connected cascading layers.
        # Create n_dense_units number of dense units.
        top_level_units = list()
        for _ in range(n_dense_units):
            # Create n_res_units number of residual units for every dense unit.
            res_layers = list()
            for _ in range(n_res_units):
                res_layers.append(
                    pitn.nn.layers.ResBlock3dNoBN(
                        self.interior_channels,
                        kernel_size=3,
                        activate_fn=activate_fn,
                        padding="same",
                        padding_mode="reflect",
                    )
                )
            top_level_units.append(
                pitn.nn.layers.DenseCascadeBlock3d(self.interior_channels, *res_layers)
            )

        # Wrap everything into a densely-connected cascade.
        self.cascade = pitn.nn.layers.DenseCascadeBlock3d(
            self.interior_channels, *top_level_units
        )
        self.post_conv = torch.nn.Sequential(
            torch.nn.Conv3d(
                self.interior_channels,
                self.interior_channels,
                kernel_size=5,
                padding="same",
                padding_mode="reflect",
            ),
            self.activate_fn,
            torch.nn.Conv3d(
                self.interior_channels,
                self.out_channels,
                kernel_size=3,
                padding="same",
                padding_mode="reflect",
            ),
            self.activate_fn,
            torch.nn.Conv3d(
                self.out_channels,
                self.out_channels,
                kernel_size=1,
                padding="same",
                padding_mode="reflect",
            ),
        )

    def forward(self, x: torch.Tensor):
        y = self.pre_conv(x)
        y = self.activate_fn(y)
        y = self.cascade(y)
        y = self.activate_fn(y)
        y = self.post_conv(y)

        return y

    @staticmethod
    def crop_pad_to_match_gt_shape(
        model_output: torch.Tensor, ground_truth: torch.Tensor, **pad_kwargs
    ) -> torch.Tensor:
        x_shape = torch.as_tensor(model_output.shape[2:])
        y_shape = torch.as_tensor(ground_truth.shape[2:])

        d = y_shape - x_shape

        pad_ijk_low = tuple(torch.floor(d / 2).cpu().int().numpy().tolist())
        pad_ijk_high = tuple(torch.ceil(d / 2).cpu().int().numpy().tolist())

        # Padding goes from inner-most dim -> outermost dim (for a bcijk tensor, it runs
        # kjicb).
        padding_kji = tuple(
            itertools.chain.from_iterable(
                [
                    (pad_ijk_low[dim], pad_ijk_high[dim])
                    for dim in reversed(range(len(pad_ijk_low)))
                ]
            )
        )

        y = torch.nn.functional.pad(model_output, pad=padding_kji, **pad_kwargs)
        return y
