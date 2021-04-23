# torchio.Transform functions/objects.

import numpy as np
import torch
import torchio
import skimage
import skimage.transform
import dipy
import dipy.core
import dipy.reconst
import dipy.reconst.dti
import dipy.segment.mask


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
        print(f"Selecting with bvals: Subject {subject.subj_id}", flush=True)

        for img in self.get_images(subject):
            bvals = img[self.bval_key]
            scans_to_keep = (self.bval_range[0] <= bvals) & (
                bvals <= self.bval_range[-1]
            )
            img[self.bvec_key] = img[self.bvec_key][scans_to_keep, :]
            img.set_data(img.data[scans_to_keep, ...])
            img[self.bval_key] = img[self.bval_key][scans_to_keep]
        print("\tSelected", flush=True)
        return subject


class MeanDownsampleTransform(torchio.SpatialTransform):
    """Mean downsampling transformation.

    sample_extension: the extension of the low-res patch size relative to the full-res
        patch size.

    Ex. sample_extension of 1.5

    Expects volumes in canonical (RAS+) format with *channels first.*
    """

    def __init__(self, downsample_factor: int, spatial_padding: int = 0, **kwargs):
        super().__init__(**kwargs)

        self.downsample_factor = downsample_factor
        self.spatial_padding = spatial_padding

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:
        print(f"Downsampling: Subject {subject.subj_id}", flush=True)
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
            padding_mask = (dim_factors - 1).astype(bool).astype(int)
            padding = self.spatial_padding * padding_mask
            padding = [(0, p) for p in padding.tolist()]
            downsample_vol = np.pad(downsample_vol, pad_width=padding, mode="constant")

            downsample_vol = torch.from_numpy(
                downsample_vol.astype(img_ndarray.dtype)
            ).to(img.data.dtype)
            img.set_data(downsample_vol)

            scaled_affine = img.affine.copy()
            # Scale the XYZ coordinates on the main diagonal.
            scaled_affine[(0, 1, 2), (0, 1, 2)] = (
                scaled_affine[(0, 1, 2), (0, 1, 2)] * self.downsample_factor
            )
            img.affine = scaled_affine
        print("\tDownsampled", flush=True)
        return subject


class FitDTITransform(torchio.SpatialTransform, torchio.IntensityTransform):
    def __init__(
        self,
        bval_key,
        bvec_key,
        mask_img_key=None,
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

    def apply_transform(self, subject: torchio.Subject) -> torchio.Subject:

        print(f"Fitting to DTI: Subject {subject.subj_id}", flush=True)
        mask_img = subject[self.mask_img_key] if self.mask_img_key is not None else None
        for img in self.get_images(subject):

            gradient_table = dipy.core.gradients.gradient_table_from_bvals_bvecs(
                bvals=img[self.bval_key],
                bvecs=img[self.bvec_key],
            )

            tensor_model = dipy.reconst.dti.TensorModel(
                gradient_table, fit_method=self.fit_method, **self.tensor_model_kwargs
            )
            print(f"\tDWI shape: {img.data.shape}", flush=True)
            # dipy does not like the channels being first, apparently.
            if mask_img is not None:
                dti = tensor_model.fit(
                    np.moveaxis(img.numpy(), 0, -1),
                    mask=mask_img.numpy().squeeze().astype(bool),
                )
            else:
                dti = tensor_model.fit(np.moveaxis(img.numpy(), 0, -1))

            # Pull only the lower-triangular part of the DTI (the non-symmetric
            # coefficients.)
            # Do it all in one line to minimize the time that the DTI's have to be
            # duplicated in memory.
            img.set_data(
                torch.from_numpy(
                    np.moveaxis(dti.lower_triangular().astype(np.float32), -1, 0)
                ).to(img.data)
            )
            print(f"\tDTI shape: {img.shape}", flush=True)
        print(f"\tFitted DTI model: {img.data.shape}", flush=True)

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
