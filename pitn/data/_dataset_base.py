# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Sequence, Optional, Callable, Tuple

import numpy as np
import torch
import monai
import nibabel as nib

import pitn


class VolDataset(monai.data.Dataset):
    def __init__(self, im, transform=None, patch_ds_kwargs=None, **meta_vals):
        if patch_ds_kwargs is None:
            patch_ds_kwargs = dict()
        super().__init__([{"im": im, **meta_vals}], transform=transform)
        self._patch_ds_kwargs = patch_ds_kwargs
        self._patches = None
        self._cache_data = None

    @property
    def patches(self):
        if self._patches is None:
            try:
                self._patches = _VolPatchDataset(
                    self[0][0]["im"], **self._patch_ds_kargs
                )
            except TypeError as e:
                raise (RuntimeError("ERROR: Cannot create patch Dataset"), e)
        return self._patches

    def __getitem__(self, idx):
        if self._cache_data is None:
            sample = super().__getitem__(idx)
            self._cache_data = sample

        return self._cache_data


def patch_select_any_in_mask(patches, coords, mask, **kwargs):
    return torch.any(mask, dim=tuple(range(1, patches.ndim)))


class _VolPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_im,
        patch_shape,
        stride: int = 1,
        patch_select_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor, ...], torch.Tensor]
        ] = None,
        patch_select_batch_size=1000,
        transform=None,
        **meta_vals,
    ):
        self._spatial_dims = len(source_im.shape[2:])
        self._patch_shape = monai.utils.misc.ensure_tuple_rep(patch_shape, self._spatial_dims)
        self._stride = monai.utils.misc.ensure_tuple_rep(stride, self._spatial_dims)


        super().__init__(patches, transform=transform)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
