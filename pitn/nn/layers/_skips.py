# -*- coding: utf-8 -*-
import torch


class ResBlock3dNoBN(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size,
        activate_fn,
        **conv_kwargs,
    ):

        super().__init__()

        explicit_conv_kwargs = dict(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
        )
        self._conv_kwargs = {**conv_kwargs, **explicit_conv_kwargs}

        self.conv1 = torch.nn.Conv3d(**self._conv_kwargs)
        self.conv2 = torch.nn.Conv3d(**self._conv_kwargs)

        self.active_fn = activate_fn

    def forward(self, x):
        y_res = self.conv1(x)
        y_res = self.active_fn(y_res)
        y_res = self.conv2(y_res)
        y = x + y_res
        y = self.active_fn(y)

        return y


class DenseCascadeBlock3d(torch.nn.Module):
    def __init__(self, channels: int, *base_layers):
        super().__init__()

        self.in_channels = channels
        self.out_channels = channels
        # If base_layers was not unpacked, then assume that a sequence of Modules was
        # passed instead.
        if len(base_layers) == 1 and not isinstance(base_layers[0], torch.nn.Module):
            base_layers = base_layers[0]
        # Assume the given base layers all have the same input size == output size.
        self.base_layers = torch.nn.ModuleList(base_layers)

        self.combiner_convs = [
            torch.nn.LazyConv3d(self.out_channels, kernel_size=1, stride=1, padding=0)
            for _ in self.base_layers
        ]
        self.combiner_convs = torch.nn.ModuleList(self.combiner_convs)

    def forward(self, x):

        # Unit (primary layer) input for layer l.
        x_unit_l = x
        # Combiner's input for the previous layer l-1.
        x_cascade_lm1 = x

        for base_layer, combiner in zip(self.base_layers, self.combiner_convs):
            y_unit_l = base_layer(x_unit_l)
            # Concatenate the previous cascaded input with the current base layer's
            # output to form the input for this block's combiner.
            x_cascade_l = torch.cat([x_cascade_lm1, y_unit_l], dim=1)
            y_l = combiner(x_cascade_l)

            # Set up next iteration.
            # Output of combiner at this step is the input to the next step.
            x_unit_l = y_l
            # The unit's output concatenated with the previous cascade inputs becomes
            # the new cascaded input.
            x_cascade_lm1 = x_cascade_l

        return y_l
