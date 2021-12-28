# -*- coding: utf-8 -*-
from pathlib import Path
import collections
from typing import Sequence, Optional, Callable, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Subset
import monai
import nibabel as nib

import pitn


class VolDataset(monai.data.Dataset):
    def __init__(self, im, transform=None, patch_ds_kwargs=None, **meta_vals):
        if patch_ds_kwargs is None:
            patch_ds_kwargs = dict()
        super().__init__([{"im": im, **meta_vals}], transform=transform)
        self._patch_ds_kwargs = {**meta_vals, **patch_ds_kwargs}
        self._patches = None
        self._cache_data = None

    @property
    def patches(self):
        if self._patches is None:
            self._patches = _VolPatchDataset(
                self[0]["im"], parent_ds=self, **self._patch_ds_kwargs
            )
        return self._patches

    def __getitem__(self, idx):
        if self._cache_data is None:
            sample = super().__getitem__(idx)
            self._cache_data = sample

        return self._cache_data


def patch_select_any_in_mask(im, idx, mask, **kwargs):
    return torch.any(mask, dim=tuple(range(1, im.ndim))).flatten()


class _VolPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_im: torch.Tensor,
        patch_shape: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        patch_select_fn: Optional[
            Callable[[torch.Tensor, Tuple[torch.Tensor]], torch.Tensor]
        ] = None,
        patch_select_batch_size: int = 1000,
        transform=None,
        meta_keys_to_patch_index=set(),
        parent_ds=None,
        **meta_vals,
    ):
        """Dataset of (optionally filtered) patches of an N-dim Tensor.

        The behavior of patch filtering is expressive, but slightly convoluted.

        If given, the 'patch_select_fn' will be called with two positional parameters,
        the 'source_im' patches and the indices of those patches. The remaining args
        will be keyword only, coming from the 'meta_vals' argument. All keys in
        'meta_keys_to_patch_index' will be passed as patches with the same indices as
        the 'source_im' patches. This way, both positional information (such as a
        segmentation mask that corresponds to the im patches) and global metadata (such
        as 'subject_id') can be used in the patch filtering.

        Iterative patch filtering can be slow, so patches are processed as batches of
        patches (swatches) to reduce time in the python loop (equivalent to setting
        the batch size to 1). At the same time, this does not require as much memory as
        processing all patches at once (setting batch size to infinite).

        Some code taken from the `monai.data.Dataset` class, see
        <https://docs.monai.io/en/stable/_modules/monai/data/dataset.html#Dataset>.

        Parameters
        ----------
        source_im : torch.Tensor
            Primary Tensor from which patches will be sampled.

            Must be shape [C x (spatial_dim_1, spatial_dim_2, ...)].
        patch_shape : tuple

        stride : int, optional
            by default 1
        patch_select_fn : Optional[ Callable[[torch.Tensor, Tuple[torch.Tensor]], torch.Tensor] ], optional
            Function to select valid patches, by default None

            This function should have the following signature:
            fn(patches: torch.Tensor, patches_idx: Tuple[torch.Tensor], other_keyword_arg1, other_keyword_arg2, ..., **kwargs)
                -> torch.BoolTensor

            The 'patches' and 'patches_idx' fields will be called by position. All other
            arguments will be called by keyword. Other keyword arguments may be used
            (outside of **kwargs) so long as the function uses '**kwargs' to accept any
            number of keyword args.

            Additional kwargs are given according to the 'meta_vals' argument.

        patch_select_batch_size : int, optional
            by default 1000
        transform : optional
            by default None
        meta_keys_to_patch_index : Sequence[str], optional
            Sequence of keys in 'meta_vals' to be indexed by patches, by default set()

            Values in 'meta_vals' that have keys in 'meta_keys_to_patch_index' are
            assumed to be Tensors with the same number of dimensions as 'source_im' and
            the same spatial dimension size(s). Batch and channel size may be different,
            but must exist with at least size 1.

            Each Tensor indexed by these keys will be indexed with patches in an
            equivalent manner to the original 'source_im'. I.e., swatches of patches
            will index into the spatial dimensions of each indicated Tensor. The
            source_im key is implicit, and will be keyed with 'im'.

            When a patch filtering function is used, the patches of each selected
            Tensor will be passed to the function with the key's name.

        parent_ds : VolDataset, optional
            by default None
        """
        self._parent_ds = parent_ds
        self.transform = transform
        self._source_im = source_im
        self._n_spatial_dims = len(self._source_im.shape[1:])
        self._spatial_shape = self._source_im.shape[1:]
        self._patch_shape = monai.utils.misc.ensure_tuple_rep(
            patch_shape, self._n_spatial_dims
        )
        self._stride = monai.utils.misc.ensure_tuple_rep(stride, self._n_spatial_dims)

        # Make sure 'im' from the source_im is added into the different fields.
        keys_to_index = tuple(
            set(meta_keys_to_patch_index).union(
                {
                    "im",
                }
            )
        )

        meta_vals = {**meta_vals, "im": self._source_im}

        # Differentiate between fields that are constant with every sample from this
        # Dataset, and those that change according to the index.
        constant_fields = tuple(set(meta_vals.keys()) - set(keys_to_index))

        # Store a reference to the full data.
        self._ims_to_index = {k: meta_vals[k] for k in keys_to_index}
        # Store a reference to the constant metadata.
        self._meta = {k: meta_vals[k] for k in constant_fields}

        self._patch_start_idx = list()
        sample_gen = pitn.utils.patch.batched_patches_iter(
            # Add a batch dimension to each image that is to be indexed.
            tuple(torch.as_tensor(im)[None, ...] for im in self._ims_to_index.values()),
            patch_shape=self._patch_shape,
            stride=self._stride,
            max_swatch_size=patch_select_batch_size,
        )

        for swatch_i, full_idx, start_idx in sample_gen:
            # Store start indices as a "S x N_dim x ..." Tensor, for more compact
            # storage.
            start_idx = torch.stack(start_idx, 0).swapdims(0, 1)
            if patch_select_fn is not None:
                # Select the patches that have been indexed into, moving the Swatch dim
                # to the front, and removing the batch dim.
                swatch_i = tuple(im[0].swapaxes(0, 1) for im in swatch_i)
                iter_dict = dict(zip(self._ims_to_index.keys(), swatch_i))
                # Add constant metadata fields as function kwargs.
                kwargs_dict = {**iter_dict, **self._meta}
                # Place the primary source image swatch and the full patch index
                # as the first two positional args. Then dump the rest as keyword args.
                kwargs_dict.pop("im")
                patch_selection = patch_select_fn(
                    iter_dict["im"], full_idx, **kwargs_dict
                )
                # Sub-select only patches with a True value from the selection function.
                select_start_idx = start_idx[patch_selection]
            else:
                select_start_idx = start_idx

            self._patch_start_idx.append(select_start_idx)

        self._patch_start_idx = torch.concat(self._patch_start_idx, dim=0).to(
            torch.int32
        )

    @property
    def parent_ds(self):
        return self._parent_ds

    def __len__(self):
        return len(self._patch_start_idx)

    def _transform(self, data: dict):
        """
        Fetch single data item from `self.data`.
        """
        return (
            monai.transforms.apply_transform(self.transform, data)
            if self.transform is not None
            else data
        )

    @staticmethod
    def _get_patch_from_tensor_start_idx(
        im: torch.Tensor,
        start_idx: torch.Tensor,
        patch_shape: tuple,
        num_channels: int,
        return_idx=False,
    ):
        """Assumes im is shape C x spatial_dim_1 x spatial_dim_2 x ...
        start_idx a Tensor of shape patch_size_1 x patch_size_2 x ...

        *Note* there is *no* batch dimension used here, a batch size of 1 is assumed.

        Parameters
        ----------
        im : torch.Tensor
        start_idx : torch.Tensor
        patch_shape : tuple
        num_channels : int
        return_idx : bool, optional
            by default False
        """

        # if num_channels is not None and num_channels > 0:
        #     # Expand the indexing into channels by expanding the patches over the
        #     # *entire* size of the channel dimension. So patches become
        #     # C x p_1 x p_2 x ...
        #     patch_shape = (num_channels, *patch_shape)
        #     # Add a channel dimension
        #     channel_idx = torch.zeros(
        #         1,
        #     )
        #     start_idx = torch.concat([channel_idx, start_idx])
        #     start_idx = tuple(start_idx[:, None, ...])
        # else:
        #     start_idx = tuple(start_idx[:, None, ...])

        if num_channels is not None and num_channels > 0:
            channel_size = (num_channels,)
        else:
            channel_size = tuple()
        full_idx = pitn.utils.patch.extend_start_patch_idx(
            start_idx, patch_shape=patch_shape, span_extra_dims_sizes=channel_size
        )

        return (im[full_idx], full_idx) if return_idx else im[full_idx]

    def __getitem__(self, idx: Union[int, slice, Sequence[int]]) -> dict:
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """

        if isinstance(idx, slice):
            # dataset[:42]
            start, stop, step = idx.indices(len(self))
            indices = range(start, stop, step)
            result = Subset(dataset=self, indices=indices)
        elif isinstance(idx, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            result = Subset(dataset=self, indices=idx)
        else:
            # Start constructing the sample dictionary to return.
            sample = dict()
            # Add the fixed fields to the sample.
            sample.update(self._meta)

            patch_start_idx = self._patch_start_idx[idx]
            for k_im in self._ims_to_index.keys():
                im_patch, full_idx = self._get_patch_from_tensor_start_idx(
                    self._ims_to_index[k_im],
                    patch_start_idx,
                    self._patch_shape,
                    num_channels=self._ims_to_index[k_im].shape[0],
                    return_idx=True,
                )
                # Rename primary source image patch from 'im' to 'patch'.
                if k_im == "im":
                    sample["patch"] = im_patch
                else:
                    sample[k_im] = im_patch
            sample["idx"] = full_idx
            result = self._transform(sample)

        return result
