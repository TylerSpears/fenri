# -*- coding: utf-8 -*-
import numpy as np
import SimpleITK as sitk


def b0_idx(bvals) -> np.ndarray:
    pass


# For use mainly with topup when sub-selecting AP/PA b0s, want the b0s with the lowest
# amount of head motion, i.e. the ones closest to the median b0.
def least_distort_b0_idx(b0s, num_selections=1, seed=None):
    # Compare each b0 to the median of all b0s.
    median = np.median(b0s, axis=3).astype(b0s.dtype)
    sitk_median = sitk.GetImageFromArray(median)
    l_b0 = np.split(b0s, b0s.shape[-1], axis=3)
    sitk_mi = sitk.ImageRegistrationMethod()
    sitk_mi.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    if seed is not None:
        sitk_mi.SetMetricSamplingPercentage(
            float(sitk_mi.GetMetricSamplingPercentagePerLevel()[0]), seed=seed
        )
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
