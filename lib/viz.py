# -*- coding: utf-8 -*-
import collections

import addict
from addict import Addict

import numpy as np
import torch
import torchio

import dipy
import dipy.core
import dipy.reconst
import dipy.reconst.dti
import dipy.segment.mask

# Create FA map from DTI's
def fa_map(dti, channels_first=True) -> np.ndarray:
    if torch.is_tensor(dti):
        t = dti.cpu().numpy()
    else:
        t = np.asarray(dti)
    # Reshape to work with dipy.
    if channels_first:
        t = t.transpose(1, 2, 3, 0)

    # Re-create the symmetric DTI's (3x3) from the lower-triangular portion (6).
    t = dipy.reconst.dti.from_lower_triangular(t)
    eigvals, eigvecs = dipy.reconst.dti.decompose_tensor(t)

    fa = dipy.reconst.dti.fractional_anisotropy(eigvals)

    return fa


# Generate FA-weighted diffusion direction map.
def direction_map(dti, channels_first=True) -> np.ndarray:

    if torch.is_tensor(dti):
        t = dti.cpu().numpy()
    else:
        t = np.asarray(dti)
    # Reshape to work with dipy.
    if channels_first:
        t = t.transpose(1, 2, 3, 0)

    # Re-create the symmetric DTI's (3x3) from the lower-triangular portion (6).
    t = dipy.reconst.dti.from_lower_triangular(t)
    eigvals, eigvecs = dipy.reconst.dti.decompose_tensor(t)

    fa = dipy.reconst.dti.fractional_anisotropy(eigvals)
    direction_map = dipy.reconst.dti.color_fa(fa, eigvecs)

    if channels_first:
        return direction_map.transpose(3, 0, 1, 2)

    return direction_map


# Return type wrapper
MultiresGridSample = collections.namedtuple(
    "MultiresGridSample",
    ("low_res", "full_res", "full_res_locations", "full_res_masks", "subj_ids"),
)

# Collate function for the DataLoader to combine multiple samples with locations.
def collate_locations(
    samples,
    full_res_key: str,
    low_res_key: str,
    subj_id_key: str,
    full_res_mask_key: str = None,
):
    full_res_stack = torch.stack([subj[full_res_key].data for subj in samples])
    full_res_location_stack = torch.stack(
        [torch.as_tensor(subj["location"]) for subj in samples]
    )
    # Assume the low-res data are dicts, not `torchio.Image`'s
    low_res_stack = torch.stack([subj[low_res_key]["data"] for subj in samples])

    # Grab mask tensors, if requested.
    if full_res_mask_key is not None:
        fr_masks = torch.stack(
            [torch.as_tensor(subj[full_res_mask_key]) for subj in samples]
        )
        fr_masks = fr_masks.bool()
    else:
        fr_masks = torch.ones_like(full_res_stack).bool()

    # If the masks do not have a channel dimension, add one right after the batch
    # dimension.
    if fr_masks.ndim == 4:
        fr_masks = fr_masks[:, None, ...]

    # Grab subject IDs
    subj_ids = torch.as_tensor(
        [subj[subj_id_key] for subj in samples], dtype=torch.int
    ).to(full_res_stack.device)

    return MultiresGridSample(
        low_res=low_res_stack,
        full_res=full_res_stack,
        full_res_locations=full_res_location_stack,
        full_res_masks=fr_masks,
        subj_ids=subj_ids,
    )


def collate_locs_and_keys(
    samples,
    full_res_key: str,
    low_res_key: str,
    full_res_mask_key: str = None,
    **extra_keys,
) -> Addict:

    # # Return type wrapper
    # MultiKeySample = collections.namedtuple(
    #     "MultiKeySample",
    #     ("low_res", "full_res", "full_res_masks", "full_res_locs", *extra_keys.keys()),
    # )

    # Grab the mandatory and specialized values first.
    full_res_stack = torch.stack([subj[full_res_key][torchio.DATA] for subj in samples])
    full_res_location_stack = torch.stack(
        [torch.as_tensor(subj[torchio.LOCATION]) for subj in samples]
    )
    # Assume the low-res data are dicts, not `torchio.Image`'s
    low_res_stack = torch.stack([subj[low_res_key]["data"] for subj in samples])

    # Grab mask tensors, if requested.
    if full_res_mask_key is not None:
        fr_masks = torch.stack(
            [torch.as_tensor(subj[full_res_mask_key]) for subj in samples]
        )
        fr_masks = fr_masks.bool()
    else:
        fr_masks = torch.ones_like(full_res_stack).bool()

    # If the masks do not have a channel dimension, add one right after the batch
    # dimension.
    if fr_masks.ndim == 4:
        fr_masks = fr_masks[:, None, ...]

    specified_keys = dict()
    for k_new, k_old in extra_keys.items():
        v = [subj[k_old] for subj in samples]
        try:
            v = torch.as_tensor(v).to(full_res_stack.device)
        except ValueError:
            pass

        specified_keys[k_new] = v

    return dict(
        low_res=low_res_stack,
        full_res=full_res_stack,
        full_res_masks=fr_masks,
        full_res_locs=full_res_location_stack,
        **specified_keys,
    )


class SubGridAggregator(torchio.GridAggregator):
    def __init__(
        self,
        spatial_shape,
        patch_overlap,
        location_offset=0,
        overlap_mode: str = "crop",
    ):

        if np.isscalar(patch_overlap):
            patch_overlap = (patch_overlap,) * 3
        self.patch_overlap = torch.as_tensor(patch_overlap)
        if (self.patch_overlap == 0).all():
            self.volume_padded = False
        else:
            self.volume_padded = True

        self.spatial_shape = spatial_shape
        if np.isscalar(location_offset):
            location_offset = (location_offset,) * 3
        self.location_offset = torch.as_tensor(location_offset)

        self._output_tensor = None
        self.parse_overlap_mode(overlap_mode)
        self.overlap_mode = overlap_mode
        self._avgmask_tensor = None

    def add_batch(self, batch_tensor: torch.Tensor, locations: torch.Tensor) -> None:
        adjusted_locs = locations.clone()
        adjusted_locs[:, :3] = adjusted_locs[:, :3] - self.location_offset
        adjusted_locs[:, 3:] = adjusted_locs[:, 3:] - self.location_offset

        return super().add_batch(batch_tensor, adjusted_locs)
