# -*- coding: utf-8 -*-
# torchio.Transform functions/objects.

from typing import (
    Dict,
    Hashable,
    Mapping,
)

import numpy as np
import torch
import torch.nn.functional as F
import torchio
from torchio.transforms.preprocessing.label.label_transform import LabelTransform
import monai
from monai.transforms.intensity.array import ThresholdIntensity
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.utils_pytorch_numpy_unification import percentile, clip

import skimage
import skimage.transform
import skimage.morphology
import scipy
import dipy
import dipy.core
import dipy.reconst
import dipy.reconst.dti
import dipy.segment.mask
import nibabel as nib
import joblib


class BValSelectionTransform(torchio.SpatialTransform):
    """Sub-selects scans that are within a certain range of bvals.

    Expects:
    - volumes in canonical (RAS+) format with *channels first.*
    - bvecs to be of shape (N, 3), with N being the number of scans/bvals.

    """

    def __init__(self, bval_range: tuple, bval_key, bvec_key, **kwargs):
        super().__init__(**kwargs)

        self.bval_range = bval_range
        self.bval_key = bval_key
        self.bvec_key = bvec_key

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        print(f"Selecting with bvals: Subject {subject.subj_id}...", flush=True, end="")

        for img in self.get_images(subject):
            bvals = img[self.bval_key]
            scans_to_keep = (self.bval_range[0] <= bvals) & (
                bvals <= self.bval_range[-1]
            )
            img[self.bvec_key] = img[self.bvec_key][scans_to_keep, :]
            img.set_data(img.data[scans_to_keep, ...])
            img[self.bval_key] = img[self.bval_key][scans_to_keep]
        print("Selected", flush=True)
        return subject


class DilateMaskTransform(LabelTransform):
    def __init__(self, dilation_size: int = None, st_elem=None, **kwargs):

        super().__init__(**kwargs)
        if (dilation_size is None and st_elem is None) or (
            dilation_size is not None and st_elem is not None
        ):
            raise TypeError(
                "ERROR: One and only one of dilate_size or st_elem may be set, got "
                + f"dilation_size: {dilation_size} and st_elem: {st_elem}"
            )

        if st_elem is None:
            self.st_elem = skimage.morphology.ball(radius=dilation_size)
        else:
            self.st_elem = st_elem

        if dilation_size is not None and dilation_size == 0:
            self._skip_dilation = True
        else:
            self._skip_dilation = False

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        """Based off code in `torchio.transforms.KeepLargestComponent"""

        if self._skip_dilation:
            return subject

        for image in self.get_images(subject):
            if image.num_channels > 1:
                message = (
                    "The number of input channels must be 1,"
                    f" but it is {image.num_channels}"
                )
                raise RuntimeError(message)
            # Remove channel dimension, which should just be 1.
            mask = image.tensor.detach().cpu().bool().numpy()[0]
            dilated_mask = scipy.ndimage.binary_dilation(mask, self.st_elem)
            dilated_mask = torch.from_numpy(dilated_mask[None, ...]).to(image.tensor)
            image.set_data(dilated_mask)

        return subject


class MeanDownsampleTransform(torchio.SpatialTransform):
    def __init__(self, downsample_factor: int, **kwargs):

        """Mean downsampling transformation.

        Expects volumes in canonical (RAS+) format with *channels first.*
        """
        super().__init__(**kwargs)

        self.downsample_factor = downsample_factor

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        print(f"Downsampling: Subject {subject.subj_id}...", flush=True, end="")
        # Get reference to Image objects that have been included for transformation.

        for img in self.get_images(subject):
            img["downsample_factor"] = self.downsample_factor
            if self.downsample_factor == 1:
                continue
            # Determine dimension-specific downsample factors
            img_ndarray = img.data.numpy()
            dim_factors = np.asarray(
                [
                    self.downsample_factor,
                ]
                * img_ndarray.ndim
            )
            # Only spatial dimensions should be downsampled.
            if img.data.ndim > 3:
                # Don't downsample the channels
                dim_factors[0] = 1
                # Or anything else outside of spatial dims.
                dim_factors[4:] = 1

            downsample_vol = skimage.transform.downscale_local_mean(
                img_ndarray, factors=tuple(dim_factors), cval=0
            )
            # Pad with a small number of 0's to account for sampling at the edge of the
            # full-res image.
            # Don't pad dims that were not scaled.
            # padding_mask = (dim_factors - 1).astype(bool).astype(int)
            # padding = self.spatial_padding * padding_mask
            # padding = [(0, p) for p in padding.tolist()]
            # downsample_vol = np.pad(downsample_vol, pad_width=padding, mode="constant")

            downsample_vol = torch.from_numpy(
                downsample_vol.astype(img_ndarray.dtype)
            ).to(img.data.dtype)
            img.set_data(downsample_vol)

            scaled_affine = img.affine.copy()
            target_voxel_sizes = np.diag(img.affine)[:-1] * dim_factors[1:4]
            scaled_affine = nib.affines.rescale_affine(
                img.affine.copy(),
                shape=img_ndarray.shape[1:4],
                zooms=target_voxel_sizes,
                new_shape=tuple(downsample_vol.shape[1:4]),
            )
            img.affine = scaled_affine
        print("Downsampled", flush=True)
        return subject


class FractionalMeanDownsampleTransform(torchio.SpatialTransform):
    def __init__(self, source_vox_size: float, target_vox_size: float, **kwargs):

        """Mean downsampling by a non-integer factor > 1.

        Expects volumes in canonical (RAS+) format with *channels first.*
        """
        super().__init__(**kwargs)

        self.source_vox_size = source_vox_size
        self.target_vox_size = target_vox_size
        self.downsample_factor = self.target_vox_size / self.source_vox_size

    # @staticmethod
    # def downscale_image(
    #     img: np.ndarray,
    #     source_vox_size: float,
    #     target_vox_size: float,
    #     padding_kwargs={"mode": "constant", "constant_values": 0.0},
    # ):
    #     img_shape = img.shape if img.ndim < 4 else img.shape[-3:]
    #     idx, weights = coords.transform.fraction_downsample_idx_weights(
    #         img_shape, source_vox_size, target_vox_size
    #     )

    #     pad_amt = int(
    #         np.max(
    #             np.clip(
    #                 (idx.max(axis=(1, 2, 3, 4)) + 1) - np.asarray(img_shape), 0, np.inf
    #             )
    #         )
    #     )
    #     weighted = list()
    #     channel_iter = (
    #         iter(img)
    #         if img.ndim == 4
    #         else [
    #             img,
    #         ]
    #     )
    #     for channel_img in channel_iter:

    #         padded_img = np.pad(channel_img, ((0, pad_amt),) * 3, **padding_kwargs)
    #         patches = padded_img[tuple(idx)]
    #         weighted.append((patches * weights).sum(axis=0))

    #     weighted = np.stack(weighted)
    #     if img.ndim < 4:
    #         weighted = weighted[0]

    #     return weighted

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:

        # Get reference to Image objects that have been included for transformation.
        for img in self.get_images(subject):
            img["downsample_factor"] = self.downsample_factor
            if self.downsample_factor == 1:
                continue

            scale_factor = self.downsample_factor ** -1

            img_data = img.data
            # Input image must be a floating point dtype to have a valid average.
            if not img_data.dtype.is_floating_point:
                img_data = img_data.float()
            # Pytorch interpolate() only accepts 5D inputs if we want to interpolate
            # 3D data (B x C x D x H x W).
            if img_data.ndim == 4:
                img_data = img_data[None, ...]

            # Calculate interpolation, chop off the batch dimension.
            downsample_vol = F.interpolate(
                img_data,
                scale_factor=scale_factor,
                mode="area",
                recompute_scale_factor=False,
            )[0]
            # Check if image is binary.
            if len(torch.unique(img.data)) == 2 and torch.unique(img.data).tolist() == [
                0,
                1,
            ]:
                nd_vol = downsample_vol.detach().cpu().numpy()
                nd_vol = np.clip(nd_vol, 0, 1)
                nd_vol = skimage.img_as_bool(nd_vol)
                downsample_vol = torch.from_numpy(nd_vol).to(img.data)

            img.set_data(downsample_vol)

            scaled_affine = nib.affines.rescale_affine(
                img.affine.copy(),
                shape=tuple(img_data.shape[2:5]),
                zooms=self.target_vox_size,
                new_shape=tuple(downsample_vol.shape[1:4]),
            )
            img.affine = scaled_affine
        print("Downsampled", flush=True)
        return subject


def fit_dti(
    dwi: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    fit_method: str,
    mask: np.ndarray = None,
    **tensor_model_kwargs,
) -> np.ndarray:

    gradient_table = dipy.core.gradients.gradient_table_from_bvals_bvecs(
        bvals=bvals,
        bvecs=bvecs,
    )

    tensor_model = dipy.reconst.dti.TensorModel(
        gradient_table, fit_method=fit_method, **tensor_model_kwargs
    )
    # dipy does not like the channels being first, apparently.
    if mask is not None:
        dti = tensor_model.fit(
            np.moveaxis(dwi, 0, -1),
            mask=mask.squeeze().astype(bool),
        )
    else:
        dti = tensor_model.fit(np.moveaxis(dwi, 0, -1))

    # Pull only the lower-triangular part of the DTI (the non-symmetric
    # coefficients.)
    # Do it all in one line to minimize the time that the DTI's have to be
    # duplicated in memory.
    dti = np.moveaxis(dti.lower_triangular().astype(np.float32), -1, 0)

    print("Not cached")

    return dti


class FitDTITransform(torchio.SpatialTransform, torchio.IntensityTransform):
    def __init__(
        self,
        bval_key,
        bvec_key,
        mask_img_key=None,
        cache_dir=None,
        fit_method="WLS",
        tensor_model_kwargs=dict(),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.bval_key = bval_key
        self.bvec_key = bvec_key
        self.mask_img_key = mask_img_key
        self.fit_method = fit_method
        self.tensor_model_kwargs = tensor_model_kwargs
        self._cache_dir = cache_dir
        # If caching should be used, override the _fit_dti method in this instantiation
        # with a cached variant.
        if self._cache_dir is not None:
            self._memory = joblib.Memory(self._cache_dir, verbose=2)
            self._fit_dti = self._memory.cache(fit_dti)
        else:
            self._memory = None

    def _fit_dti(self, *args, **kwargs):
        """Dummy method to wrap the globally-defined `fit_dti` function.

        This method will be overwritten by __init__() if caching is used.
        """
        return fit_dti(*args, **kwargs)

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:

        print(f"Fitting to DTI: Subject {subject.subj_id}...", flush=True, end="")
        mask_img = subject[self.mask_img_key] if self.mask_img_key is not None else None
        if mask_img is not None:
            mask_img = mask_img.tensor.detach().cpu().numpy().squeeze().astype(bool)
        for img in self.get_images(subject):

            dti = self._fit_dti(
                img.numpy(),
                bvals=img[self.bval_key],
                bvecs=img[self.bvec_key],
                fit_method=self.fit_method,
                mask=mask_img,
                **self.tensor_model_kwargs,
            )

            img.set_data(torch.from_numpy(dti).to(img.data))

            # gradient_table = dipy.core.gradients.gradient_table_from_bvals_bvecs(
            #     bvals=img[self.bval_key],
            #     bvecs=img[self.bvec_key],
            # )

            # tensor_model = dipy.reconst.dti.TensorModel(
            #     gradient_table, fit_method=self.fit_method, **self.tensor_model_kwargs
            # )
            # print(f"...DWI shape: {img.data.shape}...", flush=True, end="")
            # # dipy does not like the channels being first, apparently.
            # if mask_img is not None:
            #     dti = tensor_model.fit(
            #         np.moveaxis(img.numpy(), 0, -1),
            #         mask=mask_img.numpy().squeeze().astype(bool),
            #     )
            # else:
            #     dti = tensor_model.fit(np.moveaxis(img.numpy(), 0, -1))

            # # Pull only the lower-triangular part of the DTI (the non-symmetric
            # # coefficients.)
            # # Do it all in one line to minimize the time that the DTI's have to be
            # # duplicated in memory.
            # img.set_data(
            #     torch.from_numpy(
            #         np.moveaxis(dti.lower_triangular().astype(np.float32), -1, 0)
            #     ).to(img.data)
            # )
            print(f"...DTI shape: {img.shape}...", flush=True, end="")
        print(f"Fitted DTI model: {img.data.shape}", flush=True)

        return subject


class RenameImageTransform(torchio.Transform):
    def __init__(self, name_mapping: dict, **kwargs):
        super().__init__(**kwargs)

        self.name_mapping = name_mapping

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        for old_name, new_name in self.name_mapping.items():
            tmp = subject[old_name]
            subject.remove_image(old_name)
            subject.add_image(tmp, new_name)
        subject.update_attributes()
        return subject


class ImageToDictTransform(torchio.Transform):
    """Convert a Subject Image to a simple dict item.

    Removes the `include`ed keys from calculation of the Subject's properties, such as
    `spatial_shape`, `spacing`, etc.
    """

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        for img_name in self.include:

            img_dict = dict(subject[img_name])
            subject.remove_image(img_name)

            subject[img_name] = img_dict

        subject.update_attributes()
        return subject


class ClipPercentileTransformd(monai.transforms.MapTransform):
    """
    Dictionary-based transformation for clipping images at their percentiles.

    Based almost entirely on :py:class:`monai.transforms.ThresholdIntensityd`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        lower: float, lower percentile that determines lower intensity of the input
            image. Must be in range [0, 100] and lower < upper.
        upper: float, upper percentile of input image. Must be in range [0, 100] and
            upper > lower.
        only_nonzero: bool, chooses whether or not to normalize only non-zero intensity
            values.
        channel_wise: bool, chooses whether or not to calculate the percentiles for
            each channel individually, or select percentiles of the whole image.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = ThresholdIntensity.backend

    def __init__(
        self,
        keys: KeysCollection,
        lower: float,
        upper: float,
        only_nonzero: bool = False,
        channel_wise: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        if lower < 0.0 or lower > 100.0:
            raise ValueError("Percentiles must be in the range [0, 100]")
        if upper < 0.0 or upper > 100.0:
            raise ValueError("Percentiles must be in the range [0, 100]")
        if lower > upper:
            raise ValueError("Upper percentile must be greater than lower percentile")

        super().__init__(keys, allow_missing_keys)
        self.lower_percentile = lower
        self.upper_percentile = upper
        self.only_nonzero = only_nonzero
        self.channel_wise = channel_wise

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        # Iterate over images in the dict.
        for key in self.key_iterator(d):
            # Whether or not to handle channels independently.
            if self.channel_wise:
                n_channels = d[key].shape[0]
                for i in range(n_channels):
                    # If only nonzeros are considered, calculate percentiles on only
                    # nonzero values, and only clip nonzero values.
                    if self.only_nonzero:
                        img_slicer = d[key][i] != 0
                    else:
                        img_slicer = slice(None)

                    lower_val = percentile(d[key][i][img_slicer], self.lower_percentile)
                    upper_val = percentile(d[key][i][img_slicer], self.upper_percentile)
                    # Only change the selected pixels/voxels for this channel.
                    d[key][i][img_slicer] = clip(
                        d[key][i][img_slicer], lower_val, upper_val
                    )
            else:
                if self.only_nonzero:
                    img_slicer = d[key] != 0
                else:
                    img_slicer = slice(None)

                lower_val = percentile(d[key][img_slicer], self.lower_percentile)
                upper_val = percentile(d[key][img_slicer], self.upper_percentile)
                d[key][img_slicer] = clip(d[key][img_slicer], lower_val, upper_val)
        return d
