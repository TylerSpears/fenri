# -*- coding: utf-8 -*-
import itertools
import numpy as np
import torch
import monai


def normalize_batch(batch, mean, var, eps=1.0e-10):
    """Normalize a batch of patches to have mean 0 and variance 1 in the given dimensions.

    The normalization occurs according to the dimensionality of the mean and var passed.
        For example, to perform channel-wise normalization, mean and var must have shape
        (1, C, 1, 1, 1)
    """

    return (batch - mean) / torch.sqrt(var + eps)


def denormalize_batch(normal_batch, mean, var, eps=1.0e-10):
    """De-normalize a batch to have given mean and variance.

    Inverse of `normalize_batch`.

    The normalization occurs according to the dimensionality of the mean and var passed.
    """

    return (normal_batch * torch.sqrt(var + eps)) + mean


def normalize_dti(dti, mean, var, eps=1.0e-10):
    """Normalize a 6-channel DTI tensor to have mean 0 and variance 1.

    The normalization occurs according to the dimensionality of the mean and var passed.
        To perform channel-wise normalization, mean and var must have shape
        (C x 1 x 1 x 1)
    """

    return (dti - mean) / torch.sqrt(var + eps)


def denormalize_dti(normal_dti, mean, var, eps=1.0e-10):
    """Inverse of DTI normalization."""

    return (normal_dti * torch.sqrt(var + eps)) + mean


class DTIMinMaxScaler:
    def __init__(
        self,
        min,
        max,
        dim,
        quantile_low=0.0,
        quantile_high=1.0,
        channel_size=None,
        clip=False,
    ):
        """Container for channel-wise reversable min-max scaling of n-dim data.

        Parameters
        ----------
        min : Target min feature value, or list of minimums per-channel.
        max : Target max feature value, or list of maximums per-channel.
        dim : int, Tuple[Int]
            Dimension or dimensions to scale over.
        quantile_low : float, optional
            Lower quantile of input data in range [0.0, 1.0), by default 0.0
        quantile_high : float, optional
            Upper quantile of input data in range (0.0, 1.0], by default 1.0
        channel_size : int, optional
            Size of the channel dimension, if there is any. Set to `None` to indicate
            no channels will be in any inputs. By default None
        clip : bool, optional
            When scaling data, should values outside the of min/max be clipped, by
            default False.
        """

        if channel_size is None:
            self._channel_wise = False
            self._channel_size = 1
            num_reps = 1
        else:
            self._channel_wise = True
            self._channel_size = channel_size
            num_reps = channel_size
        self.min = monai.utils.misc.ensure_tuple_rep(min, num_reps)
        self.max = monai.utils.misc.ensure_tuple_rep(max, num_reps)
        q_low = monai.utils.misc.ensure_tuple_rep(quantile_low, num_reps)
        q_high = monai.utils.misc.ensure_tuple_rep(quantile_high, num_reps)
        self.quantile_ranges = torch.as_tensor([q_low, q_high]).T
        self.clip = clip
        if np.isscalar(dim):
            dim = (dim,)
        self.dim_to_reduce = dim
        self._data_range = [
            None,
        ] * num_reps
        self.clip = clip

    def scale(self, x: torch.Tensor, stateful=True):
        """MinMax scale the input Tensor, optionally saving scale information.

        Assumes x is channel-first, if any channels exist.

        Parameters
        ----------
        x : torch.Tensor
        stateful : bool, optional
            By default True
        """

        x_standard = x
        if not x.is_floating_point():
            x_standard = x_standard.float()
        # Add channel dim if not expecting channel dimension, remove later.
        if not self._channel_wise:
            x_standard = x_standard.view(1, *x_standard.shape)

        y_scaled = torch.empty_like(x_standard)

        for i_channel in range(self._channel_size):
            x_min_max = torch.quantile(
                x_standard[i_channel],
                q=self.quantile_ranges[i_channel].to(x.device),
                keepdim=True,
            )

            # Formula taken from
            # <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html>
            scale = (self.max[i_channel] - self.min[i_channel]) / (
                x_min_max[1] - x_min_max[0]
            )
            scaled = (
                scale * x_standard[i_channel]
                + self.min[i_channel]
                - x_min_max[0] * scale
            )

            if self.clip:
                scaled = torch.clamp(scaled, self.min[i_channel], self.max[i_channel])

            y_scaled[i_channel] = scaled

            if stateful:
                self._data_range[i_channel] = x_min_max

        y_scaled = y_scaled.reshape_as(x)

        return y_scaled

    def descale(self, x_scaled: torch.Tensor):
        x_standard = x_scaled
        if not x_scaled.is_floating_point():
            x_standard = x_standard.float()
        # Add channel dim if not expecting channel dimension, remove later.
        if not self._channel_wise:
            x_standard = x_standard.view(1, *x_standard.shape)

        y_descaled = torch.empty_like(x_standard)

        for i_channel in range(self._channel_size):
            x_min_max = self._data_range[i_channel]
            # Formula taken from
            # <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html>
            scale = (self.max[i_channel] - self.min[i_channel]) / (
                x_min_max[1] - x_min_max[0]
            )
            descaled = (
                x_standard[i_channel] - (self.min[i_channel] - x_min_max[0] * scale)
            ) / scale

            y_descaled[i_channel] = descaled

        y_descaled = y_descaled.reshape_as(x_scaled)

        return y_descaled
