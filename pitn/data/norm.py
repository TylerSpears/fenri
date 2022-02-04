# -*- coding: utf-8 -*-
import itertools
import numpy as np
import torch
import monai
import skimage
import scipy


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
        self._orig = None

    def __str__(self):
        data_range_str = (
            "Not defined"
            if all(map(lambda r: r is None, self._data_range))
            else f"{torch.squeeze(torch.stack(self._data_range)).cpu().numpy()}"
        )

        return (
            "MinMax scaler"
            + f"\n    Quantile ranges: {self.quantile_ranges.cpu().numpy().tolist()}"
            + "\n    Data range: "
            + data_range_str
        )

    @torch.no_grad()
    def scale(self, x: torch.Tensor, stateful=True, keep_orig=False):
        """MinMax scale the input Tensor, optionally saving scale information.

        Assumes x is channel-first, if any channels exist.

        Parameters
        ----------
        x : torch.Tensor
        stateful : bool, optional
            By default True
        keep_orig: bool, optional
            By default False
        """

        if keep_orig:
            self._orig = torch.clone(x.detach()).cpu()
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

    @property
    def orig(self):
        return self._orig

    @torch.no_grad()
    def descale(self, x_scaled: torch.Tensor):
        x_standard = x_scaled
        if not x_scaled.is_floating_point():
            x_standard = x_standard.float()
        # Add channel dim if not expecting channel dimension, remove later.
        if not self._channel_wise:
            x_standard = x_standard.view(1, *x_standard.shape)
        y_descaled = torch.empty_like(x_standard)
        for i_channel in range(self._channel_size):
            x_min_max = self._data_range[i_channel].to(x_standard)
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


def mask_constrain_clamp(
    vol: torch.Tensor,
    mask: torch.Tensor,
    quantile_clamp: tuple,
    selection_st_elem=None,
):
    """Clamp brain volume based on quantiles of the volume's outer edges, channel-wise.

    Parameters
    ----------
    vol : torch.Tensor
    mask : torch.Tensor
    quantile_clamp : float
    selection_st_elem : Union[np.ndarray, torch.Tensor], optional

    Returns
    -------
    torch.Tensor
        Same type and shape as vol, but with some voxels clamped to the given quantile.
    """

    n_channel = vol.shape[0]
    st_elem = (
        skimage.morphology.ball(4) if selection_st_elem is None else selection_st_elem
    )
    if torch.is_tensor(st_elem):
        st_elem = st_elem.detach().cpu().numpy()
    st_elem = st_elem.astype(bool)
    if mask.ndim == 4:
        mask = mask[0]
    mask = mask.bool()
    np_mask = mask.detach().cpu().numpy()
    eroded = torch.from_numpy(skimage.morphology.binary_erosion(np_mask, st_elem)).to(
        mask
    )
    # Select the outer edge of the mask that was eroded and operate only on that.
    outer_mask = mask ^ eroded
    selected = torch.masked_select(vol, outer_mask).view(n_channel, -1)
    q = torch.quantile(selected, torch.as_tensor(quantile_clamp), dim=1)
    q = q.view(*((2, n_channel) + (1,) * (vol.ndim - 1)))
    v = vol.clamp(q[0], q[1])
    # selected = selected.clamp_max(q)

    # Only apply clamping to the outer edge of the volume.
    # v = torch.masked_scatter(vol, outer_mask, selected)

    return v


def correct_edge_noise_with_median(
    vol: torch.Tensor,
    mask: torch.Tensor,
    max_num_vox_to_change: int,
    erosion_st_elem=skimage.morphology.ball(2),
    median_st_elem=skimage.morphology.cube(2),
) -> torch.Tensor:

    if torch.is_tensor(erosion_st_elem):
        erosion_se = erosion_st_elem.detach().cpu().numpy()
    else:
        erosion_se = np.asarray(erosion_st_elem)
    erosion_se = erosion_se.astype(bool)
    if torch.is_tensor(median_st_elem):
        median_se = median_st_elem.detach().cpu().numpy()
    else:
        median_se = np.asarray(median_st_elem)
    median_se = median_se.astype(bool)
    if mask.ndim == 4:
        mask = mask[0]

    np_mask = mask.detach().cpu().numpy().astype(bool)
    eroded = skimage.morphology.binary_erosion(np_mask, erosion_se)
    # Select the outer edge of the mask that was eroded and operate only on that.
    outer_mask = np_mask ^ eroded

    # Operate on each channel seperately.
    corrected_channels = list()
    vs = vol.detach().cpu().numpy()
    for v in vs:
        # Calculate the difference between the median filtered vol and the non-filtered
        # vol.
        median = skimage.filters.median(v, median_se)
        # Only work on the diff within the outer mask.
        diff = np.abs(v - median) * outer_mask
        # Select top 25th percentile or the arithmetic mean for the initial step towards
        # isolating the outliers.
        diff_min = max(diff[diff > 0].mean(), np.quantile(diff[diff > 0], 0.75))
        # Remove all values lower than the 75th percentile of diff values
        diff[diff <= diff_min] = 0
        # Total allowed vox to change equates to some quantile in the diff map.
        # Ensure the quantile cannot go out of [0, 1]
        channel_max_vox = min(max_num_vox_to_change, len(diff[diff > 0]))
        target_diff_q = 1 - (channel_max_vox / len(diff[diff > 0]))
        diff_min = np.quantile(diff[diff > 0], target_diff_q)
        diff[diff <= diff_min] = 0

        corrected_v = np.copy(v)
        # Replace the outlier with the value found in the median filter.
        corrected_v[np.where(diff > 0)] = median[np.where(diff > 0)]
        corrected_channels.append(corrected_v)

    corrected = np.stack(corrected_channels, 0)
    corrected = torch.from_numpy(corrected).to(vol)

    return corrected
