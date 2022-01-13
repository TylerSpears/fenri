# -*- coding: utf-8 -*-
"""General DIQT network models.
Based on work by Stefano B. Blumberg to illustrate methodology utilised in:
Deeper Image Quality Transfer: Training Low-Memory Neural Networks for 3D Images (MICCAI 2018)

Code taken from:
<github.com/sbb-gh/Deeper-Image-Quality-Transfer-Training-Low-Memory-Neural-Networks-for-3D-Images>
"""

import numpy as np
import torch
import torch.nn.functional as F
import einops
import einops.layers
import einops.layers.torch

import pitn


# Basic conv net definition.
class ThreeConv(torch.nn.Module):
    """Basic three-layer 3D conv network for DIQT.

    Assumes input shape is `B x C x H x W x D`."""

    def __init__(self, channels: int, downsample_factor: int, norm_method=None):
        super().__init__()
        self.channels = channels
        self.downsample_factor = downsample_factor
        self.norm_method = norm_method

        # Set up Conv layers.
        self.conv1 = torch.nn.Conv3d(self.channels, 50, kernel_size=(3, 3, 3))
        self.conv2 = torch.nn.Conv3d(50, 100, kernel_size=(1, 1, 1))
        self.conv3 = torch.nn.Conv3d(
            100, self.channels * (self.downsample_factor ** 3), kernel_size=(3, 3, 3)
        )
        self.output_shuffle = pitn.nn.layers.upsample.ESPCNShuffle(
            self.channels, self.downsample_factor
        )

        if self.norm_method is not None:
            if "instance" in self.norm_method.casefold():
                self.norm = torch.nn.InstanceNorm3d(self.channels, eps=1e-10)
                self.norm_method = "instance"
            elif "batch" in self.norm_method.casefold():
                self.norm = torch.nn.BatchNorm3d(self.channels, eps=1e-10)
                self.norm_method = "batch"
            else:
                raise RuntimeError(
                    f"ERROR: Invalid norm method '{self.norm_method}', "
                    + f"expected one of '{('instance', 'batch')}'"
                )
        else:
            self.norm = None

    def forward(self, x, norm_output=False):
        if self.norm is not None:
            x = self.norm(x)
        y_hat = self.conv1(x)
        y_hat = F.relu(y_hat)
        y_hat = self.conv2(y_hat)
        y_hat = F.relu(y_hat)
        y_hat = self.conv3(y_hat)

        # Shuffle output.
        y_hat = self.output_shuffle(y_hat)

        # Normalize/standardize output, if requested.
        if norm_output:
            if isinstance(self.norm, torch.nn.InstanceNorm3d):
                y_hat = F.instance_norm(y_hat, eps=self.norm.eps)
            elif isinstance(self.norm, torch.nn.BatchNorm3d):
                y_hat = F.batch_norm(y_hat, eps=self.norm.eps)

        return y_hat


class FractDownReduceBy5Conv(torch.nn.Module):
    """Basic three-layer 3D conv network for DIQT.

    Layers insure that patch sizes will be reduced by 5 in each dimension.
    A final tri-linear interpolation is performed before output to compensate for any
        non-interger scaling factors. In other words, break up the upsampling into two
        steps for non-integer downsampling:
            1. Over-upsample an integer amount using an NN
            2. Downsample by a non-integer amount to get to the original target size.

    Assumes input shape is `B x C x H x W x D`."""

    def __init__(self, channels: int, downsample_factor: int, norm_method=None):
        super().__init__()
        self.channels = channels
        self.downsample_factor = downsample_factor
        self.norm_method = norm_method
        # Break up the upsampling into two steps for non-integer downsampling:
        # 1. Over-upsample an integer amount using an NN
        # 2. Downsample by a non-integer amount to get to the original target size.
        self._espcn_upsample_factor = np.ceil(self.downsample_factor).astype(int)

        # Set up Conv layers.
        self.conv1 = torch.nn.Conv3d(self.channels, 50, kernel_size=(4, 4, 4))
        self.conv2 = torch.nn.Conv3d(50, 100, kernel_size=(1, 1, 1))
        self.conv3 = torch.nn.Conv3d(
            100,
            self.channels * (self._espcn_upsample_factor ** 3),
            kernel_size=(3, 3, 3),
        )
        self.output_shuffle = pitn.nn.layers.upsample.ESPCNShuffle(
            self.channels, self._espcn_upsample_factor
        )
        self.shuffle_pad_amt = (
            3,
            2,
        ) * 3
        self.norm = None

        self._interp_downsample_factor = (
            self.downsample_factor / self._espcn_upsample_factor
        )

    def forward(
        self,
        x,
        norm_output=False,
        pad_reduced_shape=False,
        interp_to_spatial_shape=None,
    ):

        y_hat = self.conv1(x)
        y_hat = F.relu(y_hat)
        y_hat = self.conv2(y_hat)
        y_hat = F.relu(y_hat)
        y_hat = self.conv3(y_hat)

        # Shuffle output.
        y_hat = self.output_shuffle(y_hat)
        if pad_reduced_shape:
            # Pad the ESPCN output to maintain the correct network output shape.
            y_hat = F.pad(y_hat, self.shuffle_pad_amt, mode="constant", value=0)

        # Downsample by a non-integer amount with tri-linear interpolation.
        if interp_to_spatial_shape:
            interp_kwargs = {"size": interp_to_spatial_shape}
        else:
            interp_kwargs = {
                "scale_factor": self._interp_downsample_factor,
                "recompute_scale_factor": True,
            }

        y_hat = F.interpolate(
            y_hat, mode="trilinear", align_corners=False, **interp_kwargs
        )

        return y_hat


class FractionThreeConv(torch.nn.Module):
    """Three-layer 3D conv network for DIQT that handles fractional downsampling.

    Assumes input shape is `B x C x H x W x D`."""

    def __init__(
        self,
        channels: int,
        source_vox_size: float,
        target_vox_size: float,
    ):
        super().__init__()
        self.channels = channels
        self.source_vox_size = source_vox_size
        self.target_vox_size = target_vox_size
        self.downsample_factor = self.target_vox_size / self.source_vox_size

        self.conv1 = torch.nn.Conv3d(self.channels, 50, kernel_size=(3, 3, 3))
        self.conv2 = torch.nn.Conv3d(50, 75, kernel_size=(2, 2, 2))
        self.conv3 = torch.nn.Conv3d(75, 100, kernel_size=(1, 1, 1))
        self.conv4 = torch.nn.Conv3d(100, 75, kernel_size=(2, 2, 2))
        self.conv5 = torch.nn.Conv3d(75, 60, kernel_size=(2, 2, 2))

        rounded_downsample_factor = int(np.ceil(self.downsample_factor))
        unshuffled_n_channels = self.channels * (rounded_downsample_factor ** 3)
        self.conv6 = torch.nn.Conv3d(60, unshuffled_n_channels, kernel_size=(3, 3, 3))
        self.output_shuffle = pitn.nn.layers.upsample.ESPCNShuffle(
            self.channels, rounded_downsample_factor
        )

        self.norm = None

    def forward(self, x, norm_output=False):

        y_hat = self.conv1(x)
        y_hat = F.relu(y_hat)
        y_hat = self.conv2(y_hat)
        y_hat = F.relu(y_hat)
        y_hat = self.conv3(y_hat)
        y_hat = F.relu(y_hat)
        y_hat = self.conv4(y_hat)
        y_hat = F.relu(y_hat)
        y_hat = self.conv5(y_hat)
        y_hat = F.relu(y_hat)
        y_hat = self.conv6(y_hat)

        # Shuffle output.
        y_hat = self.output_shuffle(y_hat)

        return y_hat


# Single-layer FC/MPL for debugging
class DebugFC(torch.nn.Module):
    """Single-layer FC with no non-linearity, for debugging.

    Assumes inputs have dimensions `B x C x H x W x D`.

    Parameters:
        input_shape: tuple
            Shape of input *excluding* batch dimension.

        output_shape: tuple
            Shape of output *excluding* batch dimension.
    """

    def __init__(self, input_shape, output_shape):
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.lin1 = torch.nn.Linear(
            torch.prod(torch.as_tensor(self.input_shape)).int().item(),
            torch.prod(torch.as_tensor(self.output_shape)).int().item(),
            bias=True,
        )

    def forward(self, x, *args, **kwargs):

        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        y_hat = self.lin1(x)
        y_hat = y_hat.view(batch_size, *self.output_shape)

        return y_hat
