#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import nibabel as nib
import skimage

if __name__ == "__main__":
    assert len(sys.argv) == 3
    mask_f = Path(sys.argv[1]).resolve()
    output_f = Path(sys.argv[2])
    x = nib.load(mask_f)
    # Close holes.
    y = skimage.morphology.binary_closing(
        x.get_fdata().astype(bool), skimage.morphology.ball(2)
    )

    # Slight dilation.
    dilate_footprint = skimage.morphology.ball(1)
    y = skimage.morphology.binary_dilation(y, dilate_footprint)
    y_nib = nib.Nifti1Image(y, x.affine, x.header)

    nib.save(y_nib, output_f)
