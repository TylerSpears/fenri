#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk


def best_b0_indices(b0s, num_selections=1):
    # Compare each b0 to the median of all b0s.
    median = np.median(b0s, axis=3).astype(b0s.dtype)
    sitk_median = sitk.GetImageFromArray(median)
    l_b0 = np.split(b0s, b0s.shape[-1], axis=3)
    sitk_mi = sitk.ImageRegistrationMethod()
    sitk_mi.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    sitk_mi.SetInterpolator(sitk.sitkNearestNeighbor)
    mis = list()
    print("Mutual Information between b0s and median b0: ")
    sanity_mi = sitk_mi.MetricEvaluate(sitk_median, sitk_median)
    sanity_mi = -sanity_mi
    print("MI between median and itself: ", sanity_mi)
    for b0 in l_b0:
        b0 = np.squeeze(b0)
        sitk_b0 = sitk.GetImageFromArray(b0)
        mi = sitk_mi.MetricEvaluate(sitk_median, sitk_b0)
        # MI is negated to work with a minimization optimization
        # (not neg log-likelihood?)
        mi = -mi
        mis.append(mi)
    print(mis)
    # Sort from max to min for easier indexing.
    mi_order = np.flip(np.argsort(np.asarray(mis)))
    return mi_order[:num_selections]


def main():
    thresh = int(sys.argv[1])
    n_selections = int(sys.argv[2])
    dwi_f = Path(str(sys.argv[3])).resolve()
    bvec_f = Path(str(sys.argv[4])).resolve()
    bval_f = Path(str(sys.argv[5])).resolve()
    out_parent_dir = Path(str(sys.argv[6])).parent.resolve()
    output_basename = Path(str(sys.argv[6])).name

    dwi = nib.load(dwi_f)
    bvec = np.loadtxt(bvec_f)
    bval = np.loadtxt(bval_f).flatten()

    bval_select_idx = bval <= thresh
    b0_dwi = dwi.get_fdata()[..., bval_select_idx]
    b0_bvec = bvec[:, bval_select_idx]
    b0_bval = bval[bval_select_idx]

    best_indices = best_b0_indices(b0_dwi, num_selections=n_selections)
    print("Selecting b0 indices ", best_indices)

    dwi_out = dwi.__class__(
        b0_dwi[..., best_indices], affine=dwi.affine, header=dwi.header
    )
    bvec_out = b0_bvec[:, best_indices]
    bval_out = b0_bval[best_indices].reshape(1, -1)

    nib.save(dwi_out, out_parent_dir / (output_basename + "".join(dwi_f.suffixes)))
    np.savetxt(
        out_parent_dir / (output_basename + ".bvec"),
        bvec_out,
        fmt="%.13f",
        delimiter=" ",
    )
    np.savetxt(
        out_parent_dir / (output_basename + ".bval"), bval_out, fmt="%g", delimiter=" "
    )


if __name__ == "__main__":
    main()
