# -*- coding: utf-8 -*-
import collections
from multiprocessing.sharedctypes import Value
from typing import Sequence, Optional, Callable, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
import einops
import monai
import nibabel as nib

import pitn
import pitn.utils


class MaskFilteredPatchDataset3d(torch.utils.data.Dataset):
    def __init__(self, img: torch.Tensor, mask: torch.Tensor, patch_size: tuple):
        """Dataset of static patches where the center voxels lies in the foreground/mask.

        Parameters
        ----------
        img : torch.Tensor
            Source image, of shape [C x H x W x D]
        mask : torch.Tensor
            Mask that corresponds to the target locations in the img, of shape [H x W x D]
        patch_size : tuple

        NOTE In the case of an odd-valued patch shape, the "left" side is rounded down
            in size, and the "right" side is rounded up.
        """

        super().__init__()
        self.patch_size = monai.utils.misc.ensure_tuple_rep(patch_size, 3)
        self.img = img

        if mask.ndim == 4:
            mask = mask[0]
        if self.img.ndim == 3:
            self.img = self.img[None, ...]

        # Grab all voxel coordinates in the given mask.
        patch_centers = torch.stack(torch.where(mask))
        # Shave off patch centers that would have a patch go out of bounds of the image.
        for i_dim, patch_dim_size in enumerate(self.patch_size):
            offset_lower = int(np.floor(patch_dim_size / 2))
            offset_upper = int(np.ceil(patch_dim_size / 2))
            patch_centers = patch_centers[
                :,
                (patch_centers[i_dim] >= offset_lower)
                & (patch_centers[i_dim] <= img.shape[i_dim + 1] - offset_upper),
            ]

        # Switch to the starting index per-patch, for easier sampling.
        self.patch_starts = (
            patch_centers
            - torch.floor(torch.as_tensor(self.patch_size)[:, None] / 2).int()
        ).T

    def __len__(self):
        return len(self.patch_starts)

    def __getitem__(self, index):
        patch_start = self.patch_starts[index]
        patch = self.img[
            :,
            patch_start[0] : patch_start[0] + self.patch_size[0],
            patch_start[1] : patch_start[1] + self.patch_size[1],
            patch_start[2] : patch_start[2] + self.patch_size[2],
        ]
        return patch


class DTIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: Path,
        subj_ids: list,
        dti_file_glob: str,
        mask_file_glob: str = None,
    ):
        super().__init__()
        self._data_dir = Path(data_dir)
        self._subj_ids = tuple(subj_ids)
        self._dti_glob = str(dti_file_glob)
        self._mask_glob = str(mask_file_glob)
        self._subjs = dict()

    @property
    def subj_ids(self):
        return self._subj_ids

    def __len__(self):
        return len(self.subj_ids)

    def __getitem__(self, idx):
        subj_id = self.subj_ids[idx]

        try:
            subj_data = self._subjs[subj_id]
        except KeyError:
            subj_data = dict()
            subj_data["subj_id"] = subj_id
            dti_file = pitn.utils.system.get_file_glob_unique(
                self._data_dir, self._dti_glob
            )
            dti = nib.load(str(dti_file))
            subj_data["dti"] = torch.from_numpy(dti.get_fdata())
            subj_data["affine"] = torch.from_numpy(dti.affine)
            subj_data["header"] = dict(dti.header)

            if self._mask_glob is not None:

                mask_file = pitn.utils.system.get_file_glob_unique(
                    self._data_dir, self._mask_glob
                )
                subj_data["mask"] = torch.from_numpy(
                    nib.load(mask_file).get_fdata()
                ).to(torch.uint8)
            else:
                subj_data["mask"] = None

            self._subjs[subj_id] = subj_data
            subj_data = self._subjs[subj_id]

        return subj_data


class SubjSesDataset(monai.data.Dataset):
    def __init__(
        self,
        vols: dict,
        primary_vol_name: str,
        special_secondary2primary_coords_fns: dict = dict(),
        transform=None,
        primary_patch_kwargs: dict = dict(),
        **meta_vals,
    ):
        """Dataset for a single subject-session, supports multiple volumes.

        Parameters
        ----------
        vols : dict
            Dict of volume Tensors (BxCxDxWxH), indexed by their names.
        primary_vol_name : str
            Source of indices for all secondary volumes.
        special_secondary2primary_coords_fns : dict, optional
            [description], by default None

            Keys in this dict require special handling when indexing into the volumes.
            This is dealt with by including a function that converts a set of coordinates
            in the primary volume to a corresponding set of coordinates in this "special"
            secondary volume.

        transform : [type], optional
            [description], by default None
        primary_patch_kwargs : dict, optional
            [description], by default None
        """

        super().__init__(
            [
                {**vols, **meta_vals},
            ],
            transform=transform,
        )
        self._cache = None
        self._patches = None
        self._prime_name = primary_vol_name

        self._second_names: set = set(vols.keys()) - {
            self._prime_name,
        }

        self._prime2second_coords = special_secondary2primary_coords_fns
        self._special_second_names = set(self._prime2second_coords.keys())
        # Set defaults for the prime patch kwargs.
        self._prime_patch_kw = {
            **dict(
                primary_name=self._prime_name,
                secondary_names=self._second_names,
                special_second_coord_fns=self._prime2second_coords,
            ),
            **primary_patch_kwargs,
        }
        self._patches_init_kwargs = self._prime_patch_kw.copy()
        # self._prime_patch_kw = primary_patch_kwargs
        self._meta_vals = meta_vals

    def set_patch_params(self, **primary_patch_kwargs):
        self._prime_patch_kw.update(**primary_patch_kwargs)

    def set_patch_sample_keys(self, primary_key: str, *keys):
        old_kwargs = self._prime_patch_kw
        new_kwargs = self._prime_patch_kw.copy()
        # Empty out the kwargs that relate to sample names and fill them in with the
        # given keys.
        new_kwargs["primary_name"] = primary_key
        new_kwargs["secondary_names"] = list()
        new_kwargs["special_second_coord_fns"] = dict()
        new_kwargs["meta_keys_to_patch_index"] = list([primary_key])

        for k in keys:
            if k in list(old_kwargs["secondary_names"]):
                new_kwargs["secondary_names"].append(k)
                if k in list(old_kwargs["special_second_coord_fns"].keys()):
                    new_kwargs["special_second_coord_fns"][k] = old_kwargs[
                        "special_second_coord_fns"
                    ][k]
                else:
                    new_kwargs["meta_keys_to_patch_index"].append(k)
            else:
                raise ValueError(
                    f"ERROR: Patch sample key {k} not valid, expected one of "
                    + f"{list(old_kwargs['secondary_names'])}"
                )
        # Update the patch sample names via the patch kwargs.
        self._prime_patch_kw = new_kwargs

    @property
    def patches(self):
        if self._patches is None or self._patches_init_kwargs != self._prime_patch_kw:
            self._patches = _MaskAnyPatchDataset(
                data=self[0],
                parent_ds=self,
                **self._prime_patch_kw,
            )
            self._patches_init_kwargs = self._prime_patch_kw.copy()
        return self._patches

    def __getitem__(self, idx):
        if self._cache is None:
            sample = super().__getitem__(idx)
            self._cache = sample

        return self._cache


class _SubjSesPatchesDataset(pitn.data._dataset_base._VolPatchDataset):
    def __init__(
        self,
        data: dict,
        primary_name: str,
        secondary_names: set,
        special_second_coord_fns: dict,
        parent_ds=None,
        transform=None,
        **vol_patches_kwargs,
    ):
        # Build the kwargs to pass to the superclass.
        super_kwargs = dict()
        # Shallow copy data to avoid in-place alteration.
        d = data.copy()
        # The primary im should not go into the catch-all meta_vals.
        source_im = d.pop(primary_name)
        super_kwargs.update(source_im=source_im, source_im_name=primary_name)

        # Handle the secondary vols that cannot be indexed from the superclass.
        self._special_coord_fns = special_second_coord_fns
        special_second_names = set(self._special_coord_fns.keys())
        self._special_seconds = dict()
        # Pop special secondaries from the data and store them for later indexing.
        for k in special_second_names:
            self._special_seconds[k] = d.pop(k)

        # Pass the non-special secondaries as meta keys to be indexed like the primary
        # volume.
        non_special_second_names = set(secondary_names) - special_second_names
        super_kwargs.update(meta_keys_to_patch_index=non_special_second_names)
        self._parent_ds = parent_ds
        super_kwargs.update(parent_ds=self._parent_ds)
        super_kwargs.update(transform=None)
        self.__transform = transform
        # Add non-special secondaries and other metadata as meta_vals.
        super_kwargs.update(**d)
        # Add patch-specific kwargs.
        super_kwargs.update(**vol_patches_kwargs)

        # Finally, initialize.
        super().__init__(**super_kwargs)

    def _transform(self, data: dict):

        # Allow for local override of transform variable; the self.__transform
        # attribute will *not* be accessed from the parent classes.
        try:
            return (
                monai.transforms.apply_transform(self.__transform, data)
                if self.__transform is not None
                else data
            )
        except AttributeError:
            return super()._transform(data)

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
            sample = super().__getitem__(idx)
            idx = sample["idx"]
            # Add a batch/swatch dim into the idx, as expected by the coord_fn funcs.
            b_idx = tuple(
                i[
                    None,
                ]
                for i in idx
            )
            # Process the volumes that need coordinate conversion.
            for name in self._special_seconds.keys():
                coord_fn = self._special_coord_fns[name]
                im = self._special_seconds[name]
                new_idx = coord_fn(b_idx)
                # Remove the batch/swatch dim from the converted idx.
                new_idx = tuple(i[0] for i in new_idx)
                # TODO Replace this patch with something more robust.
                sliced_new_idx = tuple(
                    slice(int(dim.min()), int(dim.min() + (dim.max() - dim.min() + 1)))
                    for dim in new_idx
                )
                sample[name] = im[sliced_new_idx]
                # sample[name] = im[new_idx]
                sample["idx_" + name] = new_idx
            # Make sure to transform() everything at once.
            result = self._transform(sample)
        return result


class _MaskAnyPatchDataset(_SubjSesPatchesDataset):
    def __init__(
        self,
        data: dict,
        primary_name: str,
        mask_name: str,
        secondary_names: set,
        special_second_coord_fns: dict,
        parent_ds=None,
        transform=None,
        **vol_patches_kwargs,
    ):
        """Dataset of static patches where the center voxels lies in the foreground/mask.

        NOTE In the case of an odd-valued patch shape, the "left" side is rounded down
            in size, and the "right" side is rounded up.
        """

        vol_patches_kwargs.update(patch_select_fn=None, stride=1)
        self._mask_name = mask_name
        super().__init__(
            data=data,
            primary_name=primary_name,
            secondary_names=secondary_names,
            special_second_coord_fns=special_second_coord_fns,
            parent_ds=parent_ds,
            transform=transform,
            **vol_patches_kwargs,
        )

    def select_patch_start_idx(self):
        # Grab all voxel coordinates in the given mask.
        mask = self._ims_to_index[self._mask_name].squeeze()
        mask_limits = tuple(slice(0, -s) for s in self._patch_shape)
        mask = mask[mask_limits]
        patch_start_idx = torch.nonzero(mask, as_tuple=False)

        return patch_start_idx
