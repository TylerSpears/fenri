# -*- coding: utf-8 -*-
import collections
import typing
from pathlib import Path

# Use lazy-loader of slow, unoptimized, or rarely-used module imports.
from pitn._lazy_loader import LazyLoader

import numpy as np
import torch

torchio = LazyLoader("torchio", globals(), "torchio")
ants = LazyLoader("ants", globals(), "ants")
nib = LazyLoader("nib", globals(), "nibabel")

# For more clearly designating the return values of a reader function given to
# the `torchio.Image` object.
ReaderOutput = collections.namedtuple("ReaderOutput", ["dwi", "affine"])


def nifti_reader(
    f_dwi,
) -> ReaderOutput:
    """Reader that reads in NIFTI files quickly.

    Meant for use with the `torchio.Image` object and its sub-classes.
    """

    # Load with nibabel first to get the correct affine matrix. See
    # <https://github.com/ANTsX/ANTsPy/issues/52> for why I don't trust antspy for this.
    # This does not require loading the entire NIFTI file into memory.
    affine = nib.load(f_dwi).affine.copy()
    affine = affine.astype(np.float32)
    print(f"Loading NIFTI image: {f_dwi}", flush=True)
    # Load entire image with antspy, then slice and (possibly) downsample that full image.
    # A float32 is the smallest representation that doesn't lose data.
    dwi = ants.image_read(str(f_dwi), pixeltype="float")
    print("\tLoaded NIFTI image", flush=True)

    # Use `torch.tensor()` to explicitly copy the numpy array. May have issues with
    # underlying memory getting garbage collected when using `torch.from_numpy`.
    # <https://pytorch.org/docs/1.8.0/generated/torch.tensor.html#torch.tensor>
    return ReaderOutput(dwi=torch.tensor(dwi.view()), affine=torch.tensor(affine))


def mgz_to_nifti(
    mgz_in: typing.Union[str, Path, nib.MGHImage], **nib_load_kwargs
) -> nib.Nifti2Image:
    if isinstance(mgz_in, nib.spatialimages.SpatialImage):
        img = mgz_in
    elif isinstance(mgz_in, (Path, str)):
        filename = Path(mgz_in)
        img = nib.load(filename, **nib_load_kwargs)

    nifti_img = nib.Nifti2Image.from_image(img)
    nifti_img = nib.as_closest_canonical(nifti_img, enforce_diag=False)

    return nifti_img
