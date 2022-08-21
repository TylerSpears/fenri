# -*- coding: utf-8 -*-
from typing import Optional

import pyrsistent
import torch

EPSILON = 1e-8


class MinMaxScaler:
    def __init__(
        self,
        feature_min=None,
        feature_max=None,
        data_min=None,
        data_max=None,
    ):
        self._feat_min = feature_min
        self._feat_max = feature_max
        self._data_min = data_min
        self._data_max = data_max

    def __str__(self):
        s = f"""MinMaxScaler
        From data range [{self._data_min}, {self._data_max}]
        to feature range [{self._feat_min}, {self._feat_max}]
        and vice-versa
        """

        return s

    def _select_scaler(self, passed_val, internal_val, var_name: str, none_okay=False):
        result = None
        if passed_val is not None:
            result = passed_val
        elif internal_val is not None:
            result = internal_val
        elif not none_okay:
            raise ValueError(
                f"ERROR: Expected {var_name} to be not None, but was "
                + f"given value {passed_val} and init with value {internal_val}."
            )
        return result

    def scale_to(
        self,
        x: torch.Tensor,
        feature_min: torch.Tensor = None,
        feature_max: torch.Tensor = None,
        data_min: torch.Tensor = None,
        data_max: torch.Tensor = None,
    ):
        x = x.float()
        feat_min = self._select_scaler(feature_min, self._feat_min, "feature_min").to(x)
        feat_max = self._select_scaler(feature_max, self._feat_max, "feature_max").to(x)
        dat_min = self._select_scaler(data_min, self._data_min, "data_min").to(x)
        dat_max = self._select_scaler(data_max, self._data_max, "data_max").to(x)

        # Formula taken from
        # <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html>
        scale = (feat_max - feat_min) / (dat_max - dat_min)
        x_scaled = scale * x + feat_min - dat_min * scale

        return x_scaled

    def unscale_from(
        self,
        x: torch.Tensor,
        feature_min: torch.Tensor = None,
        feature_max: torch.Tensor = None,
        data_min: torch.Tensor = None,
        data_max: torch.Tensor = None,
    ):
        # These will need to be floating point numbers.
        x = x.float()
        feat_min = self._select_scaler(feature_min, self._feat_min, "feature_min").to(x)
        feat_max = self._select_scaler(feature_max, self._feat_max, "feature_max").to(x)
        dat_min = self._select_scaler(data_min, self._data_min, "data_min").to(x)
        dat_max = self._select_scaler(data_max, self._data_max, "data_max").to(x)

        # Formula taken from
        # <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html>
        scale = (feat_max - feat_min) / (dat_max - dat_min)

        x_scaled = x
        x_unscaled = ((x_scaled - feat_min) / scale) + dat_min

        return x_unscaled


class StandardNormalScaler:

    STD_EPSILON = 1e-8

    scale_kwarg_names = ("data_mean", "data_std")

    def __init__(self, data_mean=None, data_std=None, mask=None):
        self._data_mean = data_mean
        self._data_std = data_std
        self._mask = mask

    def __str__(self):
        s = f"""StandardNormalScaler
        From data mean {self._data_mean}, std {self._data_std}
        to feature range standard normal mean of 0.0, std of 1.0
        """

        return s

    @classmethod
    def compute_scale_kwargs(cls, x, mask=None, batched=False):
        kwargs = dict()
        if not batched:
            x = x[
                None,
            ]
            if torch.is_tensor(mask):
                mask = mask[
                    None,
                ]
        batch_size = x.shape[0]
        n_channels = x.shape[1]
        n_spatial_dims = len(x.shape[2:])
        # Each mask in a batch element may select a different number of elements, so
        # means and stds will need to be calculated one batch element at a time.
        if torch.is_tensor(mask):
            x = [
                torch.masked_select(x_i, m_i).reshape(n_channels, -1)
                for (x_i, m_i) in zip(x, mask)
            ]
        else:
            x = list(x.reshape(x.shape[0], n_channels, -1))

        means = list()
        stds = list()
        for x_i in x:
            s, m = torch.std_mean(x_i, dim=-1)
            means.append(m)
            stds.append(s)

        means = torch.stack(means)
        means = means.reshape(*((batch_size, n_channels) + (1,) * n_spatial_dims))
        stds = torch.stack(stds)
        stds = stds.reshape(*((batch_size, n_channels) + (1,) * n_spatial_dims))

        if not batched:
            means = means[0]
            stds = stds[0]

        kwargs["data_mean"] = means
        kwargs["data_std"] = stds

        return kwargs

    def _select_scaler(self, passed_val, internal_val, var_name: str, none_okay=False):
        result = None
        if passed_val is not None:
            result = passed_val
        elif internal_val is not None:
            result = internal_val
        elif not none_okay:
            raise ValueError(
                f"ERROR: Expected {var_name} to be not None, but was "
                + f"given value {passed_val} and init with value {internal_val}."
            )
        return result

    def scale_to(
        self,
        x: torch.Tensor,
        data_mean: torch.Tensor = None,
        data_std: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        x = x.float()
        mean = self._select_scaler(data_mean, self._data_mean, "data_mean").to(x)
        std = self._select_scaler(data_std, self._data_std, "data_std").to(x)
        mask = self._select_scaler(mask, self._mask, "mask", none_okay=True)

        x_standard_norm = (x - mean) / (torch.clamp_min(std, self.STD_EPSILON))

        if torch.is_tensor(mask):
            x_standard_norm = x_standard_norm * mask.to(x)

        return x_standard_norm

    def unscale_from(
        self,
        x_standard_norm: torch.Tensor,
        data_mean: torch.Tensor = None,
        data_std: torch.Tensor = None,
        mask: torch.Tensor = None,
    ):
        x_normed = x_standard_norm.float()
        mean = self._select_scaler(data_mean, self._data_mean, "data_mean").to(x_normed)
        std = self._select_scaler(data_std, self._data_std, "data_std").to(x_normed)
        mask = self._select_scaler(mask, self._mask, "mask", none_okay=True).to(
            x_normed
        )

        x = (x_normed * torch.clamp_min(std, self.STD_EPSILON)) + mean

        if torch.is_tensor(mask):
            x = x * mask

        return x

    # def transform_input(self, x, means=None, stds=None, mask=None):
    #     if torch.is_tensor(means):
    #         x = x - means
    #     if torch.is_tensor(stds):
    #         x = x / (stds + self.STD_EPSILON)
    #     if torch.is_tensor(mask):
    #         x = x * mask
    #     return x

    # def transform_ground_truth_for_training(
    #     self, y, means=None, stds=None, mask=None, crop=True
    # ):
    #     if torch.is_tensor(means):
    #         y = y - means
    #     if torch.is_tensor(stds):
    #         y = y / (stds + self.STD_EPSILON)
    #     if crop:
    #         y = self.crop_full_output(y)
    #         if torch.is_tensor(mask):
    #             mask = self.crop_full_output(mask)
    #     if torch.is_tensor(mask):
    #         y = y * mask

    #     return y.float()

    # def transform_output(self, y_pred, means=None, stds=None, mask=None, crop=True):
    #     if torch.is_tensor(stds):
    #         y_pred = y_pred * (stds + self.STD_EPSILON)
    #     if torch.is_tensor(means):
    #         y_pred = y_pred + means

    #     if crop:
    #         if torch.is_tensor(mask):
    #             mask = self.crop_full_output(mask)
    #     if torch.is_tensor(mask):
    #         y_pred = y_pred * mask

    #     return y_pred.float()


# Names taken from the sklearn preprocessing transformers.
norm_method_lookup = pyrsistent.pmap(
    {"minmax": MinMaxScaler, "standard": StandardNormalScaler}
)
