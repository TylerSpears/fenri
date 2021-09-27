# -*- coding: utf-8 -*-
import itertools

import torch
import torch.nn.functional as F

from .. import layers


class DiscriminatorSubNet(torch.nn.Module):
    def __init__(self, num_input_channels: int):
        super().__init__()

        self.block1 = layers.DiscriminatorBlock(num_input_channels, 16, normalize=False)
        self.block2 = layers.DiscriminatorBlock(16, 32, normalize=True)
        self.block3 = layers.DiscriminatorBlock(32, 64, normalize=True)
        self.block4 = layers.DiscriminatorBlock(64, 128, normalize=True)
        # self.block5 = layers.DiscriminatorBlock(128, 128, normalize=True)
        # If the input to the final conv is between (kernel_size, kernel_size + 2), then
        # just chop off the extra values with the 'valid' padding method.
        # ! This assumes the conv input size is not >= kernel_size + 2, because we
        # ! are trying to produce a 1-dimensional output in the end.
        self.conv_end = torch.nn.Conv3d(128, 1, kernel_size=8, padding="valid")

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        # y = self.block5(y)
        # If y is smaller than the final conv layer's kernel size, then pad y to equal
        # the kernel size. If y is larger than the kernel size, then padding should be 0
        # and the final conv layer should handle the truncation.
        pad_amt = torch.clamp_min(
            torch.as_tensor(self.conv_end.kernel_size) - torch.as_tensor(y.shape[-3:]),
            0,
        )
        # Calculate upper and lower padding amounts.
        pad_lower = torch.floor(pad_amt / 2).int().tolist()
        pad_upper = torch.ceil(pad_amt / 2).int().tolist()
        # Reverse the dimensions to match pytorch's crazy padding order.
        pad_lower = reversed(pad_lower)
        pad_upper = reversed(pad_upper)
        # Pair the lower and upper padding amounts with each dimension, flatten the
        # list.
        padding = list(itertools.chain.from_iterable(zip(pad_lower, pad_upper)))
        y = F.pad(y, pad=padding, mode="replicate")

        y = self.conv_end(y)

        return y


class MultiDiscriminator(torch.nn.Module):
    def __init__(
        self, num_input_channels: int, discrim_downsample_factors: list = [1, 2, 4]
    ):
        super().__init__()
        self._num_input_channels = num_input_channels
        self._downsample_factors = list(discrim_downsample_factors)
        self._scale_factors = list(map(lambda x: x ** -1, self._downsample_factors))
        discriminators = list()
        for _ in self._downsample_factors:
            discriminators.append(DiscriminatorSubNet(self._num_input_channels))

        self.discriminators = torch.nn.ModuleList(discriminators)

    @staticmethod
    def _downsample(x, scale_factor: int):
        if scale_factor == 1:
            return x
        else:
            return F.interpolate(
                x, scale_factor=scale_factor, mode="area", recompute_scale_factor=False
            )

    def forward(self, x):
        y = list()
        for i_scale, scale_factor in enumerate(self._scale_factors):
            x_down = self._downsample(x, scale_factor)
            y_i = self.discriminators[i_scale](x_down)
            # Flatten prediction into B x 1 outputs.
            y.append(y_i.reshape(-1, 1))

        # Combine predictions for each scale's sub-network into a B x N_scales output.
        y = torch.cat(y, dim=1)
        return y
