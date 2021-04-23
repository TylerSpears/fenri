import collections
import typing
from typing import Generator

import numpy as np
import torch
import torchio

# Return type wrapper
MultiresSample = collections.namedtuple("MultiresSample", ("low_res", "full_res"))

# Collate function for the DataLoader to combine multiple samples.
def collate_subj(samples, full_res_key: str, low_res_key: str):
    full_res_stack = torch.stack([subj[full_res_key].data for subj in samples])
    # Assume the low-res data are dicts, not `torchio.Image`'s
    low_res_stack = torch.stack([subj[low_res_key]["data"] for subj in samples])

    return MultiresSample(low_res=low_res_stack, full_res=full_res_stack)


def extract_patch(img, img_spatial_shape, index_ini, patch_size) -> torchio.Image:
    """Draws a patch from img, given an initial index and patch size."""

    # Just take it straight from `torchio.transforms.Copy.apply_transform`
    # and `torchio.sampler.Sampler`!

    shape = np.array(img_spatial_shape, dtype=np.uint16)
    index_ini = np.array(index_ini, dtype=np.uint16)
    patch_size = np.array(patch_size, dtype=np.uint16)
    index_fin = index_ini + patch_size

    crop_ini = index_ini.tolist()
    crop_fin = (shape - index_fin).tolist()
    start = ()
    cropping = sum(zip(crop_ini, crop_fin), start)

    low = cropping[::2]
    high = cropping[1::2]
    initial_idx = low
    final_idx = np.array(img_spatial_shape) - high

    i0, j0, k0 = initial_idx
    i1, j1, k1 = final_idx

    return img[:, i0:i1, j0:j1, k0:k1]


# Custom sampler for sampling multiple volumes of different resolutions.
class MultiresSampler(torchio.LabelSampler):
    """

    source_img_key: Key to the Subject that will fetch the source (a.k.a., the high-res
        or full-res) image.

    low_res_key: Key to the Subject that will fetch the low-res image. This image is
        assumed to be a dictionary with a 'data' key.

    downsample_factor_key: Key to the low-res image dict that gives the downsample
        factor.

    source_spatial_patch_size: 3-tuple of `(W, H, D)` that gives the spatial size of
        patches drawn from the source image.

    low_res_spatial_patch_size: 3-tuple of `(W, H, D)` that gives the spatial size of
        patches drawn from the low-res image.

    subj_keys_to_copy: Tuple of keys to copy from the Subject into the returned sample
        patch(es).
    """

    def __init__(
        self,
        source_img_key,
        low_res_key,
        downsample_factor_key,
        source_spatial_patch_size: tuple,
        low_res_spatial_patch_size: tuple,
        label_name,
        subj_keys_to_copy=tuple(),
        **kwargs,
    ):

        super().__init__(
            patch_size=source_spatial_patch_size, label_name=label_name, **kwargs
        )
        self.source_img_key = source_img_key
        self.low_res_key = low_res_key
        self.downsample_factor_key = downsample_factor_key
        self.subj_keys_to_copy = subj_keys_to_copy
        self.source_spatial_patch_size = source_spatial_patch_size
        self.low_res_spatial_patch_size = low_res_spatial_patch_size

    def __call__(
        self, subject: torchio.Subject, num_patches=None
    ) -> Generator[torchio.Subject, None, None]:

        # Setup copied from the `torchio.WeightedSampler.__call__` function definition.
        subject.check_consistent_space()
        if np.any(self.patch_size > subject.spatial_shape):
            message = (
                f"Patch size {tuple(self.patch_size)} cannot be"
                f" larger than image size {tuple(subject.spatial_shape)}"
            )
            raise RuntimeError(message)
        probability_map = self.get_probability_map(subject)
        probability_map = self.process_probability_map(probability_map, subject)
        cdf = self.get_cumulative_distribution_function(probability_map)

        patches_left = num_patches if num_patches is not None else True
        while patches_left:
            subj_fields_transfer = dict(
                ((k, subject[k]) for k in self.subj_keys_to_copy)
            )

            # Sample an index from the full-res image.
            source_index_ini = self.get_random_index_ini(probability_map, cdf)
            # Create a new subject that only contains patches.
            # Add the patch from the full-res image into the subject.
            source_tensor = extract_patch(
                subject[self.source_img_key].data,
                img_spatial_shape=subject[self.source_img_key].shape[1:],
                index_ini=source_index_ini,
                patch_size=self.source_spatial_patch_size,
            )

            patch_subj = torchio.Subject(
                **(
                    dict(
                        [
                            (
                                self.source_img_key,
                                torchio.ScalarImage(
                                    tensor=source_tensor,
                                    affine=subject[self.source_img_key].affine,
                                ),
                            ),
                            *subj_fields_transfer.items(),
                        ],
                    )
                ),
            )

            # Include the index in the subject.
            patch_subj["index_ini"] = np.array(source_index_ini).astype(int)
            # Crop low-res image and add to the subject.
            lr_index_ini = tuple(
                np.array(source_index_ini).astype(int)
                // subject[self.low_res_key][self.downsample_factor_key]
            )

            lr_patch = extract_patch(
                subject[self.low_res_key]["data"],
                img_spatial_shape=subject[self.low_res_key]["data"].shape[1:],
                index_ini=lr_index_ini,
                patch_size=self.low_res_spatial_patch_size,
            )
            if lr_patch.numel() == 0:
                raise RuntimeError(
                    f"ERROR: Invalid low-res patch: {lr_patch}, {lr_patch.shape} |"
                    + f"Index: {lr_index_ini}"
                )
            # Add a dict to the subject patch, rather than a `torchio.Image`,
            # because the fr and lr patch shapes will be different, and fail
            # `torchio`'s shape consistency checks.)
            lr_patch_dict = dict()
            lr_patch_dict.update(subject[self.low_res_key])
            lr_patch_dict.update({"data": lr_patch})

            patch_subj[self.low_res_key] = lr_patch_dict
            # Return the new patch subject.
            yield patch_subj
            if num_patches is not None:
                patches_left -= 1


class MultiresGridSampler(torchio.GridSampler):
    def __init__(
        self,
        source_img_key,
        low_res_key,
        downsample_factor_key,
        source_spatial_patch_size: tuple,
        low_res_spatial_patch_size: tuple,
        subj_keys_to_copy=tuple(),
        **kwargs,
    ):

        super().__init__(patch_size=source_spatial_patch_size, **kwargs)
        self.source_img_key = source_img_key
        self.low_res_key = low_res_key
        self.downsample_factor_key = downsample_factor_key
        self.subj_keys_to_copy = subj_keys_to_copy
        self.source_spatial_patch_size = source_spatial_patch_size
        self.low_res_spatial_patch_size = low_res_spatial_patch_size

    def __getitem__(self, index):

        location = self.locations[index]
        source_index_ini = location[:3]

        subj_fields_transfer = dict(
            ((k, self.subject[k]) for k in self.subj_keys_to_copy)
        )

        # Create a new subject that only contains patches.
        # Add the patch from the full-res image into the subject.
        source_tensor = extract_patch(
            self.subject[self.source_img_key].data,
            img_spatial_shape=self.subject[self.source_img_key].shape[1:],
            index_ini=source_index_ini,
            patch_size=self.source_spatial_patch_size,
        )

        patch_subj = torchio.Subject(
            **(
                dict(
                    [
                        (
                            self.source_img_key,
                            torchio.ScalarImage(
                                tensor=source_tensor,
                                affine=self.subject[self.source_img_key].affine,
                            ),
                        ),
                        *subj_fields_transfer.items(),
                    ],
                )
            ),
        )

        # Include the index in the subject.
        patch_subj["index_ini"] = np.array(source_index_ini).astype(int)
        patch_subj[torchio.LOCATION] = location
        # Crop low-res image and add to the subject.
        lr_index_ini = tuple(
            np.array(source_index_ini).astype(int)
            // self.subject[self.low_res_key][self.downsample_factor_key]
        )

        lr_patch = extract_patch(
            self.subject[self.low_res_key]["data"],
            img_spatial_shape=self.subject[self.low_res_key]["data"].shape[1:],
            index_ini=lr_index_ini,
            patch_size=self.low_res_spatial_patch_size,
        )
        if lr_patch.numel() == 0:
            raise RuntimeError(
                f"ERROR: Invalid low-res patch: {lr_patch}, {lr_patch.shape} |"
                + f"Index: {lr_index_ini}"
            )
        # Add a dict to the subject patch, rather than a `torchio.Image`,
        # because the fr and lr patch shapes will be different, and fail
        # `torchio`'s shape consistency checks.)
        lr_patch_dict = dict()
        lr_patch_dict.update(self.subject[self.low_res_key])
        lr_patch_dict.update({"data": lr_patch})

        patch_subj[self.low_res_key] = lr_patch_dict

        return patch_subj
