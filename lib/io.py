# -*- coding: utf-8 -*-
import collections

import numpy as np
import torch
import torchio

import ants
import nibabel as nib

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
