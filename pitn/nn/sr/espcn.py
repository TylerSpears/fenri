# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pitn


# Basic conv net definition.
class ESPCNBaseline(torch.nn.Module):
    """ESPCN baseline network, replica of the baseline from Tanno et. al. 2021.

    Assumes input shape is `B x C x H x W x D`."""

    STD_EPSILON = 1e-8

    def __init__(
        self,
        channels: int,
        upscale_factor: int,
        center_crop_output_side_amt=4,
    ):
        super().__init__()

        self.channels = channels
        self.upscale_factor = upscale_factor

        self.activation_fn = torch.nn.ReLU()
        # Set up Conv layers.
        self.conv1 = torch.nn.Conv3d(self.channels, 50, kernel_size=(3, 3, 3))
        self.conv2 = torch.nn.Conv3d(50, 100, kernel_size=(1, 1, 1))
        self.conv3 = torch.nn.Conv3d(
            100, self.channels * (self.upscale_factor**3), kernel_size=(3, 3, 3)
        )
        self.output_shuffle = pitn.nn.layers.upsample.ESPCNShuffle3d(
            self.channels, self.upscale_factor
        )

        # "Padding" by a negative amount will perform cropping!
        # <https://github.com/pytorch/pytorch/issues/1331>
        if center_crop_output_side_amt is not None and center_crop_output_side_amt > 0:
            crop = -center_crop_output_side_amt
            self.output_cropper = torch.nn.ConstantPad3d(crop, 0)
            self._crop = True
        else:
            self._crop = False
            self.output_cropper = torch.nn.Identity()

    def crop_full_output(self, x):
        if self._crop:
            x = self.output_cropper(x)
        return x

    def transform_input(self, x, means=None, stds=None, mask=None):
        if torch.is_tensor(means):
            x = x - means
        if torch.is_tensor(stds):
            x = x / (stds + self.STD_EPSILON)
        if torch.is_tensor(mask):
            x = x * mask
        return x

    def transform_ground_truth_for_training(
        self, y, means=None, stds=None, mask=None, crop=True
    ):
        if torch.is_tensor(means):
            y = y - means
        if torch.is_tensor(stds):
            y = y / (stds + self.STD_EPSILON)
        if crop:
            y = self.crop_full_output(y)
            if torch.is_tensor(mask):
                mask = self.crop_full_output(mask)
        if torch.is_tensor(mask):
            y = y * mask

        return y.float()

    def transform_output(self, y_pred, means=None, stds=None, mask=None, crop=True):
        if torch.is_tensor(stds):
            y_pred = y_pred * (stds + self.STD_EPSILON)
        if torch.is_tensor(means):
            y_pred = y_pred + means

        if crop:
            if torch.is_tensor(mask):
                mask = self.crop_full_output(mask)
        if torch.is_tensor(mask):
            y_pred = y_pred * mask

        return y_pred.float()

    def forward(
        self,
        x,
        transform_x=True,
        x_means=None,
        x_stds=None,
        x_mask=None,
        transform_y=True,
        y_means=None,
        y_stds=None,
        y_mask=None,
    ):

        if transform_x:
            x = self.transform_input(x, means=x_means, stds=x_stds, mask=x_mask)

        y = self.conv1(x)
        y = self.activation_fn(y)
        y = self.conv2(y)
        y = self.activation_fn(y)
        y = self.conv3(y)

        # Shuffle output.
        y = self.output_shuffle(y)
        if self._crop:
            if torch.is_tensor(y_mask) and y_mask.shape[2:] != y.shape[2:]:
                y_mask = self.crop_full_output(y_mask)

        if transform_y:
            y = self.transform_output(
                y, means=y_means, stds=y_stds, mask=y_mask, crop=False
            )

        return y


# Created by Stefano B. Blumberg to illustrate methodology utilised in:
# Deeper Image Quality Transfer: Training Low-Memory Neural Networks for 3D Images
# (MICCAI 2018)
# <https://github.com/sbb-gh/Deeper-Image-Quality-Transfer-Training-Low-Memory-Neural-Networks-for-3D-Images/blob/master/models.py>


class ESPCNRevNet(ESPCNBaseline):
    """ESPCN_RN-N (N:=no_RevNet_layers), from
    Deeper Image Quality Transfer: Training Low-Memory Neural Networks for 3D Images
    """

    def __init__(
        self,
        channels,
        upscale_factor,
        revnet_stack_size: int,
        center_crop_output_side_amt: int = 4,
        batch_norm=True,
    ):

        # Call the nn module init directly, so we don't have layers initialized in a
        # seperate init function, and we can more clearly define them all here.
        nn.Module.__init__(self)

        self.channels = channels
        self.upscale_factor = upscale_factor

        self.activation_fn = torch.nn.ReLU()

        self.rev_block1 = RevBlock(
            self.channels, revnet_stack_size, batch_norm=batch_norm
        )
        self.conv1 = torch.nn.Conv3d(self.channels, 50, kernel_size=(3, 3, 3))

        self.rev_block2 = RevBlock(50, revnet_stack_size, batch_norm=batch_norm)
        self.conv2 = torch.nn.Conv3d(50, 100, kernel_size=(1, 1, 1))

        self.rev_block3 = RevBlock(100, revnet_stack_size, batch_norm=batch_norm)
        self.conv3 = torch.nn.Conv3d(
            100, self.channels * (self.upscale_factor**3), kernel_size=(3, 3, 3)
        )

        self.output_shuffle = pitn.nn.layers.upsample.ESPCNShuffle3d(
            self.channels, self.upscale_factor
        )

        # "Padding" by a negative amount will perform cropping!
        # <https://github.com/pytorch/pytorch/issues/1331>
        if center_crop_output_side_amt is not None and center_crop_output_side_amt > 0:
            crop = -center_crop_output_side_amt
            self.output_cropper = torch.nn.ConstantPad3d(crop, 0)
            self._crop = True
        else:
            self._crop = False
            self.output_cropper = torch.nn.Identity()

    def forward(
        self,
        x,
        transform_x=True,
        x_means=None,
        x_stds=None,
        x_mask=None,
        transform_y=True,
        y_means=None,
        y_stds=None,
        y_mask=None,
    ):

        if transform_x:
            x = self.transform_input(x, means=x_means, stds=x_stds, mask=x_mask)

        y = self.rev_block1(x)
        y = self.conv1(y)
        y = self.activation_fn(y)
        y = self.rev_block2(y)
        y = self.conv2(y)
        y = self.activation_fn(y)
        y = self.rev_block3(y)
        y = self.conv3(y)

        # Shuffle output.
        y = self.output_shuffle(y)
        if self._crop:
            if torch.is_tensor(y_mask) and y_mask.shape[2:] != y.shape[2:]:
                y_mask = self.crop_full_output(y_mask)

        if transform_y:
            y = self.transform_output(
                y, means=y_means, stds=y_stds, mask=y_mask, crop=False
            )

        return y


class RevBlock(nn.Module):
    def __init__(self, channels: int, n_revnet_layers: int, batch_norm=True):
        super().__init__()
        self.batch_norm = batch_norm
        self.channels = channels
        self.n_revnet_layers = n_revnet_layers

        block = list()
        for _ in range(n_revnet_layers):
            layer_i = RevLayer(self.channels, self.channels, batch_norm=batch_norm)
            block.append(layer_i)

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class RevLayer(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        # Index that determines where channels are split for alpha and beta in the
        # forward pass.
        self.res_1_in = self.in_channels // 2
        self.res_2_in = self.in_channels - self.res_1_in

        self.res_1_out = self.out_channels // 2
        self.res_2_out = self.out_channels - self.res_1_out

        if self.batch_norm:
            self.F = torch.nn.Sequential(
                nn.BatchNorm3d(self.res_1_in),
                nn.ReLU(),
                nn.Conv3d(self.res_1_in, self.res_1_in, kernel_size=1, padding="same"),
                nn.BatchNorm3d(self.res_1_in),
                nn.ReLU(),
                nn.Conv3d(self.res_1_in, self.res_1_in, kernel_size=3, padding="same"),
                nn.BatchNorm3d(self.res_1_in),
                nn.ReLU(),
                nn.Conv3d(self.res_1_in, self.res_1_out, kernel_size=1),
            )
            self.G = torch.nn.Sequential(
                nn.BatchNorm3d(self.res_2_in),
                nn.ReLU(),
                nn.Conv3d(self.res_2_in, self.res_2_in, kernel_size=1, padding="same"),
                nn.BatchNorm3d(self.res_2_in),
                nn.ReLU(),
                nn.Conv3d(self.res_2_in, self.res_2_in, kernel_size=3, padding="same"),
                nn.BatchNorm3d(self.res_2_in),
                nn.ReLU(),
                nn.Conv3d(self.res_2_in, self.res_2_out, kernel_size=1),
            )

        else:
            self.F = torch.nn.Sequential(
                nn.ReLU(),
                nn.Conv3d(self.res_1_in, self.res_1_in, kernel_size=1, padding="same"),
                nn.ReLU(),
                nn.Conv3d(self.res_1_in, self.res_1_in, kernel_size=3, padding="same"),
                nn.ReLU(),
                nn.Conv3d(self.res_1_in, self.res_1_out, kernel_size=1),
            )
            self.G = torch.nn.Sequential(
                nn.ReLU(),
                nn.Conv3d(self.res_2_in, self.res_2_in, kernel_size=1, padding="same"),
                nn.ReLU(),
                nn.Conv3d(self.res_2_in, self.res_2_in, kernel_size=3, padding="same"),
                nn.ReLU(),
                nn.Conv3d(self.res_2_in, self.res_2_out, kernel_size=1),
            )

    def forward(self, x):

        x_a, x_b = torch.split(x, (self.res_1_in, self.res_2_in), dim=1)
        z = x_a + self.F(x_b)
        y_b = x_b + self.G(z)
        y_a = z
        y = torch.concat([y_a, y_b], dim=1)

        return y
