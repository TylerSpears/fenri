# -*- coding: utf-8 -*-
import collections
import typing
from typing import Generator
import numbers
import warnings
import itertools

# Use lazy-loader of slow, unoptimized, or rarely-used module imports.
from pitn._lazy_loader import LazyLoader

import numpy as np
import torch
import torch.nn.functional as F

torchio = LazyLoader("torchio", globals(), "torchio")
monai = LazyLoader("monai", globals(), "monai")

# Return type wrapper
MultiresSample = collections.namedtuple("MultiresSample", ("low_res", "full_res"))

# Collate function for the DataLoader to combine multiple samples.
def collate_subj(samples, full_res_key: str, low_res_key: str):
    full_res_stack = torch.stack([subj[full_res_key].data for subj in samples])
    # Assume the low-res data are dicts, not `torchio.Image`'s
    low_res_stack = torch.stack([subj[low_res_key]["data"] for subj in samples])

    return MultiresSample(low_res=low_res_stack, full_res=full_res_stack)


# Similar setup, but for collating with the full-res mask.
MultiresMaskSample = collections.namedtuple(
    "MultiresMaskSample", ("low_res", "full_res", "full_res_mask")
)


def collate_subj_mask(
    samples, full_res_key: str, low_res_key: str, full_res_mask_key: str
):
    lr, fr = collate_subj(samples, full_res_key=full_res_key, low_res_key=low_res_key)

    fr_masks = torch.stack(
        [torch.as_tensor(subj[full_res_mask_key]) for subj in samples]
    )
    fr_masks = fr_masks.bool()

    return MultiresMaskSample(low_res=lr, full_res=fr, full_res_mask=fr_masks)


def collate_dicts(samples, *keys, **renamed_keys) -> dict:
    select_keys = dict(zip(keys, keys))
    select_keys.update(renamed_keys)
    collated = dict()

    for k_new, k_old in select_keys.items():
        v = [sample[k_old] for sample in samples]
        # If all entries are nested dictionaries, recursively convert them.
        if v and all(k_old in sample.keys() for sample in samples):
            if all(map(lambda obj: isinstance(obj, dict), v)):
                v = collate_dicts(v, **dict(zip(v[0].keys(), v[0].keys())))
            else:
                # If the values are a tensor or ndarray, try and stack them together.
                if torch.is_tensor(v[0]):
                    try:
                        v = torch.stack(v).to(v[0])
                    except (ValueError, RuntimeError):
                        pass
                elif isinstance(v[0], np.ndarray):
                    try:
                        v = np.stack(v)
                    except (ValueError, RuntimeError):
                        pass
        else:
            raise KeyError(f"ERROR: Key '{k_old}' not found in one or more entries.")

        collated[k_new] = v

    return collated


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


def map_fr_init_idx_to_lr(
    fr_idx, downsample_factor, low_res_sample_extension, fr_patch_size
) -> np.ndarray:
    """Maps starting indices of patches in FR space to LR space."""

    # Batch-ify the input, if not done already.
    if len(fr_idx.shape) == 1:
        batch_fr_idx = fr_idx.reshape(1, -1)
    else:
        batch_fr_idx = fr_idx

    # Find offset between the original FR patch index and the "oversampled" FR patch
    # index.
    idx_delta = np.round(
        (
            np.round(np.asarray(fr_patch_size) * low_res_sample_extension)
            - np.asarray(fr_patch_size)
        )
        / 2
    )
    expanded_fr_idx = batch_fr_idx - idx_delta

    # Now perform simple downscaling.
    lr_idx = np.floor(expanded_fr_idx / downsample_factor)

    # Undo batch-ification, if necessary.
    if len(fr_idx.shape) == 1:
        lr_idx = lr_idx.squeeze()

    return lr_idx.astype(int)


def map_fr_coord_to_lr(
    fr_coords, downsample_factor, low_res_sample_extension, fr_patch_size
):
    fr_index_ini = fr_coords[:, :3]
    lr_index_ini = map_fr_init_idx_to_lr(
        fr_index_ini,
        downsample_factor=downsample_factor,
        low_res_sample_extension=low_res_sample_extension,
        fr_patch_size=fr_patch_size,
    )
    lr_lengths = np.floor(
        np.round(np.asarray(fr_patch_size) * low_res_sample_extension)
        / downsample_factor
    )
    lr_coord = np.concatenate([lr_index_ini, lr_index_ini + lr_lengths], axis=-1)

    return lr_coord


def random_patches_from_mask(
    img: torch.Tensor,
    mask: torch.Tensor,
    patch_size: tuple,
    num_patches: int = 1,
    return_mask_patch: bool = False,
    generator=None,
):
    """Select random patches where the center voxels lies in the foreground/mask.

    Parameters
    ----------
    img : torch.Tensor
        Source image, of shape [C x H x W x D]
    mask : torch.Tensor
        Mask that corresponds to the target locations in the img, of shape [H x W x D]
    patch_size : tuple
    num_patches : int, optional
        Number of patches in the batch, by default 1
    return_mask_patch: bool
        Whether or not to return the mask patch at the same location as each sample
        patch. Setting this to True will return a list of tuples of Tensors, so the
        output sequence length can remain equal to the requested number of patches.
    generator : torch.generator, optional
        Pytorch Generator object, if randomization needs to be controlled., by default
        None

    Returns
    -------
    torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]
        Random sample, either as a Tensor of image patches or a sequence of
        (image_patch, mask_patch) tuples.
    """

    if return_mask_patch:
        raise NotImplementedError("Returning mask patches is not yet implemented.")
    patch_size = monai.utils.misc.ensure_tuple_rep(patch_size, 3)
    if mask.ndim == 4:
        mask = mask[0]
    if img.ndim == 3:
        img = img[None, ...]

    patch_centers = torch.stack(torch.where(mask))
    for i_dim, patch_dim_size in enumerate(patch_size):
        offset_lower = int(np.floor(patch_dim_size / 2))
        offset_upper = int(np.ceil(patch_dim_size / 2))
        patch_centers = patch_centers[
            :,
            (patch_centers[i_dim] >= offset_lower)
            & (patch_centers[i_dim] <= img.shape[i_dim + 1] - offset_upper),
        ]

    selected = torch.randperm(patch_centers.shape[1], generator=generator)[:num_patches]
    patch_centers = patch_centers[:, tuple(selected)]
    patch_starts = (
        patch_centers - torch.floor(torch.as_tensor(patch_size)[:, None] / 2).int()
    ).T
    patches = list()
    for start_idx in patch_starts:

        patches.append(
            img[
                :,
                start_idx[0] : start_idx[0] + patch_size[0],
                start_idx[1] : start_idx[1] + patch_size[1],
                start_idx[2] : start_idx[2] + patch_size[2],
            ]
        )

    patches = torch.stack(patches)
    if img.ndim == 3:
        patches = patches[:, 0]
    print(f"Sampled patch batch of size {num_patches}")
    return patches


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
        downsample_factor,
        source_spatial_patch_size: tuple,
        low_res_spatial_patch_size: tuple,
        label_name,
        low_res_sample_extension,
        source_mask_key: str = None,
        subj_keys_to_copy=tuple(),
        **kwargs,
    ):

        super().__init__(
            patch_size=source_spatial_patch_size, label_name=label_name, **kwargs
        )
        self.source_img_key = source_img_key
        self.low_res_key = low_res_key
        self.downsample_factor = downsample_factor
        self.subj_keys_to_copy = subj_keys_to_copy
        self.source_spatial_patch_size = source_spatial_patch_size
        self.low_res_spatial_patch_size = low_res_spatial_patch_size
        self.low_res_sample_extension = low_res_sample_extension
        self.source_mask_key = source_mask_key

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

            if self.source_mask_key is not None:
                # Create a new subject that only contains patches.
                # Add the patch from the full-res image into the subject.
                mask_tensor = extract_patch(
                    subject[self.source_mask_key].data,
                    img_spatial_shape=subject[self.source_mask_key].shape[1:],
                    index_ini=source_index_ini,
                    patch_size=self.source_spatial_patch_size,
                )

                patch_subj[self.source_mask_key] = mask_tensor.bool()
            # else:
            #     mask_tensor = torch.ones_like(source_tensor[0, ...])[None, ...].bool()

            # Include the index in the subject.
            patch_subj["index_ini"] = np.array(source_index_ini).astype(int)

            # Crop low-res image and add to the subject.
            lr_index_ini = map_fr_init_idx_to_lr(
                source_index_ini,
                downsample_factor=self.downsample_factor,
                low_res_sample_extension=self.low_res_sample_extension,
                fr_patch_size=self.source_spatial_patch_size,
            )
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
        low_res_sample_extension,
        source_mask=None,
        mask_patch_filter_fn=lambda p: True,
        subj_keys_to_copy=tuple(),
        **kwargs,
    ):
        """Multi-channel volume sampler for sampling an entire volume in 2 resolutions.

        Parameters
        ----------
        source_img_key : str
        low_res_key : str
        downsample_factor : int
        source_spatial_patch_size : tuple
        low_res_spatial_patch_size : tuple
        low_res_sample_extension : float
        source_mask : torch.BoolTensor, optional
            by default None
        mask_patch_filter_fn: function(patch) -> bool, optional
            Function that indicates whether a volume's patch should be accepted, given
            the source_mask's patch at that same location. Only valid if `source_mask`
            is not None. Should be of the form:
                function(torch.BoolTensor) -> bool
            with the tensor being of size 'W x H x D'. By default 'lambda p: p.any()'
        subj_keys_to_copy : optional
            by default tuple()

        Raises
        ------
        ValueError
            FR patches map to invalid LR patches
        """

        super().__init__(patch_size=source_spatial_patch_size, **kwargs)

        self.source_img_key = source_img_key
        self.low_res_key = low_res_key
        self.downsample_factor = downsample_factor
        self.subj_keys_to_copy = subj_keys_to_copy
        self.source_spatial_patch_size = source_spatial_patch_size
        self.low_res_spatial_patch_size = low_res_spatial_patch_size
        self.low_res_sample_extension = low_res_sample_extension
        self.source_mask = source_mask

        # Filter locations based upon mask, if given.
        if self.source_mask is not None:
            # Remove channel dimension, if found in the mask.
            if len(self.source_mask.shape) > 3:
                self.source_mask = self.source_mask[0]
            self.source_mask = self.source_mask.bool()
            # Sample into mask with all locations to get patches of booleans.
            locs_to_keep = list()
            for loc in self.locations:
                patch = self.source_mask[
                    loc[0] : loc[3], loc[1] : loc[4], loc[2] : loc[5]
                ]
                keep_patch = mask_patch_filter_fn(patch)
                locs_to_keep.append(bool(keep_patch))
            locs_to_keep = torch.as_tensor(locs_to_keep).bool()
            # Keep only those patches with at least 1 voxel overlapping the mask.
            self.locations = self.locations[locs_to_keep]

        # Determine beforehand whether any of the LR locations go out of bounds.
        self.lr_locations = map_fr_coord_to_lr(
            self.locations,
            self.downsample_factor,
            self.low_res_sample_extension,
            self.source_spatial_patch_size,
        )
        if np.min(self.lr_locations) < 0:
            to_remove = np.min(self.lr_locations, axis=1) < 0
            warnings.warn(
                f"Removed {to_remove.sum()}"
                + f" locations out of {self.lr_locations.shape[0]}"
            )
            self.lr_locations = self.lr_locations[~to_remove]
            self.locations = self.locations[~to_remove]
        #             raise ValueError(
        #                 "ERROR: Invalid mapping less than 0 from FR volume to LR volume."
        #             )

        lr_shape = self.subject[self.low_res_key]["data"].shape
        if np.any(np.max(self.lr_locations[:, 3:], axis=0) >= lr_shape[1:]):
            raise ValueError(
                "ERROR: Invalid mapping out of bounds from FR volume to LR volume."
            )

    def __getitem__(self, index):

        # Index into locations to get corresponding patch indices.
        location = self.locations[index]
        lr_location = self.lr_locations[index]
        source_index_ini = location[:3]
        lr_index_ini = lr_location[:3]

        subj_fields_transfer = dict(
            ((k, self.subject[k]) for k in self.subj_keys_to_copy)
        )

        # Create a new subject that only contains patches.
        # Add the patch from the full-res image into the subject.
        source_tensor = extract_patch(
            self.subject[self.source_img_key]["data"],
            img_spatial_shape=self.subject[self.source_img_key]["data"].shape[1:],
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

        # Include the source mask in the return.
        patch_source_mask = extract_patch(
            self.source_mask[None, ...],
            img_spatial_shape=self.source_mask.shape,
            index_ini=source_index_ini,
            patch_size=self.source_spatial_patch_size,
        )
        patch_subj["source_mask"] = patch_source_mask[0]

        # Include the subj ID
        patch_subj["subj_id"] = self.subject["subj_id"]
        # Crop low-res image and add to the subject.
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


class ConcatDatasetBalancedRandomSampler(torch.utils.data.Sampler):
    def __init__(self, datasets, max_samples_per_dataset, generator=None):
        """Sampler that draws a given number of samples from each dataset.

        datasets: List[torch.utils.data.Dataset]
            List of pytorch Datasets to sample.

            TIP If using a ConcatDataset object `cat_ds`, the property `cat_ds.datasets`
            will return a list of Datasets.
        max_samples_per_dataset: int or List[int]
            Give a single integer to make sample amounts the same for all datasets, or
            a list of integers with length equal to the number of datasets to specify
            a sample number particular to each dataset. If a dataset is smaller than
            the requested number of samples, the entire length of the dataset will be
            used instead. Each Dataset *must* be a Map-style Dataset.
        """

        self.ds_lens = [len(ds) for ds in datasets]
        # Expand the max samples into a list of max samples for each dataset.
        if (
            np.isscalar(max_samples_per_dataset)
            and int(max_samples_per_dataset) == max_samples_per_dataset
        ):
            self.sample_sizes = [
                max_samples_per_dataset,
            ] * len(self.ds_lens)
        else:
            self.sample_sizes = max_samples_per_dataset
            if len(self.sample_sizes) != len(self.ds_lens):
                raise ValueError(
                    "Must request sample sizes with length equal to the number of datasets"
                )
        # Make sure we don't assign more samples to a dataset than there are elements
        # in that dataset.
        self.sample_sizes = list(map(min, zip(self.ds_lens, self.sample_sizes)))
        cum_lens = list(itertools.accumulate(self.ds_lens))
        self.start_idx = [
            0,
        ] + cum_lens[:-1]
        self._total_samples = sum(self.sample_sizes)
        self.generator = generator

    def __iter__(self):
        samples = list()
        # Select random indices within each dataset, based on the size of the dataset
        # and the number of requested samples.
        for sample_size, len_i, i_start in zip(
            self.sample_sizes, self.ds_lens, self.start_idx
        ):
            idx_i = (
                i_start + torch.randperm(len_i, generator=self.generator)[:sample_size]
            )
            samples.extend(idx_i.tolist())
        # Return needs to be an iterable object, not just a sequence.
        return (
            samples[i] for i in torch.randperm(len(samples), generator=self.generator)
        )

    def __len__(self):
        return self._total_samples
