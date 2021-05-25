import collections

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
    ("low_res", "full_res", "full_res_locations"),
)

# Collate function for the DataLoader to combine multiple samples with locations.
def collate_locations(samples, full_res_key: str, low_res_key: str):
    full_res_stack = torch.stack([subj[full_res_key].data for subj in samples])
    full_res_location_stack = torch.stack(
        [torch.as_tensor(subj["location"]) for subj in samples]
    )
    # Assume the low-res data are dicts, not `torchio.Image`'s
    low_res_stack = torch.stack([subj[low_res_key]["data"] for subj in samples])

    return MultiresGridSample(
        low_res=low_res_stack,
        full_res=full_res_stack,
        full_res_locations=full_res_location_stack,
    )


class SubGridAggregator(torchio.GridAggregator):
    def __init__(
        self,
        spatial_shape,
        patch_overlap,
        location_offset=0,
        overlap_mode: str = "crop",
    ):

        self.volume_padded = True
        if np.isscalar(patch_overlap):
            patch_overlap = (patch_overlap,) * 3
        self.patch_overlap = patch_overlap
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
