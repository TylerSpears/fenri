# -*- coding: utf-8 -*-
import numpy as np
import torch
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
        center_crop_output_side_amt=None,
    ):
        super().__init__()

        self.channels = channels
        self.upscale_factor = upscale_factor

        self.activation_fn = torch.nn.ReLU()
        # Set up Conv layers.
        self.conv1 = torch.nn.Conv3d(self.channels, 50, kernel_size=(3, 3, 3))
        self.conv2 = torch.nn.Conv3d(50, 100, kernel_size=(1, 1, 1))
        self.conv3 = torch.nn.Conv3d(
            100, self.channels * (self.upscale_factor ** 3), kernel_size=(3, 3, 3)
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

        y_hat = self.conv1(x)
        y_hat = self.activation_fn(y_hat)
        y_hat = self.conv2(y_hat)
        y_hat = self.activation_fn(y_hat)
        y_hat = self.conv3(y_hat)

        # Shuffle output.
        y_hat = self.output_shuffle(y_hat)
        if self._crop:
            if torch.is_tensor(y_mask) and y_mask.shape[2:] != y_hat.shape[2:]:
                y_mask = self.crop_full_output(y_mask)

        if transform_y:
            y_hat = self.transform_output(
                y_hat, means=y_means, stds=y_stds, mask=y_mask, crop=False
            )

        return y_hat
