#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy
import skimage

if __name__ == "__main__":
    in_file = Path(sys.argv[1])
    out_file = Path(sys.argv[2])

    head_im = nib.load(in_file)
    head = head_im.get_fdata().astype(np.float32)

    # Threshold
    m = head >= 10
    # Remove small points
    m = skimage.morphology.remove_small_objects(m, min_size=6**3)
    # Large binary closing
    m = skimage.morphology.binary_closing(m, skimage.morphology.ball(4))
    # Fill remaining holes
    m = scipy.ndimage.binary_fill_holes(m)

    m_im = nib.Nifti1Image(m.astype(np.uint8), head_im.affine)

    nib.save(m_im, out_file)
