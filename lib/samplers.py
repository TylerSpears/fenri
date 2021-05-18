import collections
import typing
from typing import Generator
import numbers

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


def map_fr_locations_to_lr(fr_locations, downsample_factor, low_res_sample_extension):
    loc_ini, loc_fin = np.split(fr_locations, 2, axis=-1)
    lengths = loc_fin - loc_ini
    # Find length of extension for each dimension in fr voxel space.
    one_side_ext_len = (lengths * low_res_sample_extension) - lengths
    ext_fr_ini = loc_ini - one_side_ext_len
    ext_fr_fin = loc_fin + one_side_ext_len
    lr_ini = np.round(ext_fr_ini / downsample_factor).astype(int)
    lr_fin = np.round(ext_fr_fin / downsample_factor).astype(int)
    return np.concatenate([lr_ini, lr_fin], axis=-1)


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

    low_res_sample_extension: float (usually between 1.0 and 2.0) that determines the
        amount of "extra spatial context" is given to the low-res patch; i.e., the
        "extension" of the low-res sample around the high-res source.

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
        low_res_sample_extension=1.0,
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
        self.low_res_sample_extension = low_res_sample_extension

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
            downsample_factor = subject[self.low_res_key][self.downsample_factor_key]

            # Need the full 6-coordinate location for the map_fr_locations_to_lr
            # function.
            fr_patch_coords = np.concatenate(
                [
                    source_index_ini,
                    np.asarray(source_index_ini) + self.source_spatial_patch_size,
                ],
                axis=0,
            )
            lr_patch_coords = map_fr_locations_to_lr(
                fr_patch_coords,
                downsample_factor=downsample_factor,
                low_res_sample_extension=self.low_res_sample_extension,
            )
            lr_index_ini = lr_patch_coords[:3]
            lr_patch = extract_patch(
                subject[self.low_res_key]["data"],
                img_spatial_shape=subject[self.low_res_key]["data"].shape[1:],
                index_ini=lr_index_ini,
                patch_size=self.low_res_spatial_patch_size,
            )

            # print(
            #    f"FR patch index: {source_index_ini} | LR patch index: {lr_index_ini}"
            # )
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
        downsample_factor,
        source_spatial_patch_size: tuple,
        low_res_spatial_patch_size: tuple,
        low_res_sample_extension=1.0,
        subj_keys_to_copy=tuple(),
        **kwargs,
    ):

        super().__init__(patch_size=source_spatial_patch_size, **kwargs)
        self.source_img_key = source_img_key
        self.low_res_key = low_res_key
        self.downsample_factor = downsample_factor
        self.subj_keys_to_copy = subj_keys_to_copy
        self.source_spatial_patch_size = source_spatial_patch_size
        self.low_res_spatial_patch_size = low_res_spatial_patch_size
        self.low_res_sample_extension = low_res_sample_extension

        # Pad the FR volume further to account for lr sample extension, adjust
        # the FR locations for the same reason.
        self.subject, self.locations = self._adjust_fr_for_extension(
            subject=self.subject,
            # fr_key=self.source_img_key,
            locations=self.locations,
            low_res_sample_extension=self.low_res_sample_extension,
            fr_patch_size=self.source_spatial_patch_size,
            padding_mode=self.padding_mode,
        )

        # Pad the LR volume to match the patch overlap in the FR and the LR sample
        # extension.
        self.subject = self._pad_lr(
            subject=self.subject,
            lr_key=self.low_res_key,
            downsample_factor=self.downsample_factor,
            fr_patch_overlap=self.patch_overlap,
            low_res_sample_extension=self.low_res_sample_extension,
            fr_patch_size=self.source_spatial_patch_size,
            padding_mode=self.padding_mode,
        )

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

        # Need the full 6-coordinate location for the map_fr_locations_to_lr function.
        fr_patch_coords = location
        # Crop low-res image and add to the subject.
        lr_patch_coords = map_fr_locations_to_lr(
            fr_patch_coords,
            downsample_factor=self.downsample_factor,
            low_res_sample_extension=self.low_res_sample_extension,
        )
        lr_index_ini = lr_patch_coords[:3]
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

    def _adjust_fr_for_extension(
        self,
        subject,
        # fr_key,
        locations,
        low_res_sample_extension,
        fr_patch_size,
        padding_mode,
    ):

        if padding_mode is not None and low_res_sample_extension != 1:

            padding = np.ceil(
                np.asarray(fr_patch_size) * low_res_sample_extension
            ).astype(int)
            padding = padding.repeat(2)
            padder = torchio.transforms.Pad(padding, padding_mode=padding_mode)
            subject = padder(subject)
            locations[:3] += padding
            locations[3:] += padding

        # return both subject & adjusted locations
        return subject, locations

    def _pad_lr(
        self,
        subject,
        lr_key,
        downsample_factor,
        fr_patch_overlap,
        low_res_sample_extension,
        fr_patch_size,
        padding_mode,
    ):
        # Don't pad if the pad mode isn't specified.
        if padding_mode is not None:
            lr = subject[lr_key]["data"]

            if isinstance(padding_mode, numbers.Number):
                kwargs = {"mode": "constant", "constant_values": padding_mode}
            else:
                kwargs = {"mode": padding_mode}

            # Don't pad for patch overlap if none exists.
            if not (0 == np.asarray(fr_patch_overlap)).all():
                lr_patch_overlap = np.asarray(fr_patch_overlap) // downsample_factor
                # Now convert padding values to match `np.pad`. This is a nested
                # sequence of padding values for each dimension:
                # ((before_1, after_1), (before_2, after_2), ...)
                lr_overlap_padding = (lr_patch_overlap // 2).astype(int)
                lr_overlap_padding = lr_overlap_padding.repeat(2)
                # We don't want to pad the channels dim (dimension 0), so indicate 0
                # padding in the front.
                lr_overlap_padding = np.concatenate(([0, 0], lr_overlap_padding))
                # Split to create a nested iterable of iterables.
                lr_overlap_padding = np.split(lr_overlap_padding, 4)
                lr = np.pad(lr, lr_overlap_padding, **kwargs)

            # An extension of 1 does not need to be padded.
            if low_res_sample_extension != 1:

                ext_padding = np.ceil(
                    np.asarray(fr_patch_size)
                    / downsample_factor
                    * low_res_sample_extension
                ).astype(int)
                ext_padding = ext_padding.repeat(2)
                ext_padding = np.concatenate(([0, 0], ext_padding))
                ext_padding = np.split(ext_padding, 4)
                lr = np.pad(lr, ext_padding, **kwargs)

            subject[lr_key]["data"] = lr

        return subject
