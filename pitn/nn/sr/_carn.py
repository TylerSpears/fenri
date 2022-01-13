# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import einops

import pitn
import pitn.nn.layers as layers


class CascadeUpsampleModeRefine(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        upscale_factor: int,
        n_res_units: int,
        n_dense_units: int,
        activate_fn,
        upsample_activate_fn,
    ):
        super().__init__()

        self.channels = channels
        self.upscale_factor = upscale_factor


        # Pad to maintain the same input shape.
        self.pre_conv = torch.nn.Conv3d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )

        self.activate_fn = activate_fn

        # Construct the densely-connected cascading layers.
        # Create n_dense_units number of dense units.
        top_level_units = list()
        for _ in range(n_dense_units):
            # Create n_res_units number of residual units for every dense unit.
            res_layers = list()
            for _ in range(n_res_units):
                res_layers.append(
                    layers.ResBlock3dNoBN(
                        self.channels,
                        kernel_size=3,
                        activate_fn=activate_fn,
                        padding=1,
                    )
                )
            top_level_units.append(
                layers.DenseCascadeBlock3d(self.channels, *res_layers)
            )

        # Wrap everything into a densely-connected cascade.
        self.cascade = layers.DenseCascadeBlock3d(self.channels, *top_level_units)

        self.upsample = layers.upsample.ICNRUpsample3d(
            self.channels,
            self.channels,
            self.upscale_factor,
            activate_fn=upsample_activate_fn,
            blur=True,
            zero_bias=True,
        )

        # Perform group convolution to refine each channel of the input independently of
        # every other channel.
        self.hr_modality_refinement = torch.nn.LazyConv3d(
            self.channels, kernel_size=5, padding=2, groups=self.channels
        )

        self.post_conv = torch.nn.Conv3d(
            self.channels, self.channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor, x_mode_refine: torch.Tensor):
        y = self.pre_conv(x)
        y = self.activate_fn(y)
        y = self.cascade(y)
        y = self.upsample(y)

        # Integrate the extra HR multi-modal refinement input (i.e. a T2 or T1 patch).
        # Repeat dimensions as necessary to have the same size as the previous layer's
        # output.
        x_mode_refine = x_mode_refine.expand_as(y)
        # Interleave the channels for group convolution, where each channel from the
        # network is convolved with a copy of the extra-modal HR input.
        mode_fused = einops.rearrange(
            [y, x_mode_refine], "mixed_mode b c ... -> b (c mixed_mode) ..."
        )

        y = self.hr_modality_refinement(mode_fused)

        y = self.activate_fn(y)
        y = self.post_conv(y)

        return y