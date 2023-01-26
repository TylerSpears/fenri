# -*- coding: utf-8 -*-
import collections
from functools import partial
from typing import Optional

import dipy
import jax
import jax.dlpack
import jax.numpy as jnp
import numpy as np
import torch
from jax import lax

import pitn
import pitn.odf

PeaksContainer = collections.namedtuple(
    "PeaksContainer", ["peaks", "theta", "phi", "valid_peak_mask"]
)


def topk_peaks(
    k: int,
    fodf_peaks: torch.Tensor,
    theta_peak: torch.Tensor,
    phi_peak: torch.Tensor,
    valid_peak_mask: torch.Tensor,
) -> PeaksContainer:
    peak_idx = torch.argsort(fodf_peaks, dim=-1, descending=True)
    topk_peak_idx = peak_idx[..., :k]
    topk_peak_valid = torch.take_along_dim(valid_peak_mask, topk_peak_idx, dim=-1)

    topk_peaks = (
        torch.take_along_dim(fodf_peaks, topk_peak_idx, dim=-1) * topk_peak_valid
    )
    topk_theta = torch.take(theta_peak, topk_peak_idx) * topk_peak_valid
    topk_phi = torch.take(phi_peak, topk_peak_idx) * topk_peak_valid

    return PeaksContainer(
        peaks=topk_peaks,
        theta=topk_theta,
        phi=topk_phi,
        valid_peak_mask=topk_peak_valid,
    )


def peaks_from_segment(
    lobe_labels: torch.Tensor,
    sphere_samples: torch.Tensor,
    theta_coord: torch.Tensor,
    phi_coord: torch.Tensor,
    take_topk_peaks: Optional[int] = None,
) -> PeaksContainer:
    unique_labels = lobe_labels.unique(sorted=True)
    unique_labels = unique_labels[unique_labels > 0]

    # The number of peaks (and the shape of the return) may be dependent upon the
    # number of unique labels in the segmentation.
    if take_topk_peaks is not None:
        num_labels = take_topk_peaks
    else:
        num_labels = len(unique_labels)
    batch_size = lobe_labels.shape[0]
    peak_vals = torch.zeros(batch_size, num_labels).to(sphere_samples)
    peak_idx = -torch.ones_like(peak_vals).to(torch.long)

    # Reduce the number of loop iterations to either the k requested peaks, or the total
    # number of unique labels across the batch.
    n_labels_to_process = min(num_labels, len(unique_labels))
    for i in range(n_labels_to_process):
        l = unique_labels[i]
        select_vals = torch.where(lobe_labels == l, sphere_samples, -1)
        l_peak_idx = torch.argmax(select_vals, dim=1)[:, None]
        peak_idx[:, i] = l_peak_idx.flatten()
        peak_idx[:, i] = torch.where(
            select_vals.take_along_dim(l_peak_idx, dim=1) > 0, peak_idx[:, i, None], -1
        ).flatten()

    valid_peak_mask = peak_idx >= 0
    peak_vals = torch.where(
        peak_idx >= 0, sphere_samples.take_along_dim(peak_idx.clamp_min(0), dim=1), -1
    )
    # The invalid indices are set to 0 to avoid subtle indexing errors later on; cuda in
    # particular hates indexing out-of-bounds of a Tensor. Even though it is possible that
    # an index value of 0 is valid, this is the only way to avoid those errors. The valid
    # peak mask must be used to distinguish between real peak indices and those that are
    # actually valued at 0.
    peak_idx.clamp_min_(0)
    peak_theta = torch.take(theta_coord, index=peak_idx) * valid_peak_mask
    peak_phi = torch.take(phi_coord, index=peak_idx) * valid_peak_mask

    return PeaksContainer(
        peaks=peak_vals,
        theta=peak_theta,
        phi=peak_phi,
        valid_peak_mask=valid_peak_mask,
    )


def _t2j(t_tensor: torch.Tensor) -> jax.Array:
    t = t_tensor.contiguous()
    # Dlpack does not handle boolean arrays.
    if t.dtype == torch.bool:
        t = t.to(torch.uint8)
        to_bool = True
    else:
        to_bool = False
    if not jax.config.x64_enabled and t.dtype == torch.float64:
        # Unsafe casting, but it's necessary if jax can only handle 32-bit floats. In
        # some edge cases, like if any dimension size is 1, the conversion will error
        # out.
        t = t.to(torch.float32)

    # 1-dims cause all sorts of problems, so just remove them before conversion, then
    # add them back afterwards.
    if 1 in tuple(t.shape):
        orig_shape = tuple(t.shape)
        t = t.squeeze()
        to_expand = tuple(
            filter(lambda i_d: orig_shape[i_d] == 1, range(len(orig_shape)))
        )
    else:
        to_expand = None
    j = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(t))
    j = j.astype(bool) if to_bool else j

    if to_expand is not None:
        j = lax.expand_dims(j, to_expand)

    return j


def _j2t(j_tensor: jax.Array, delete_from_jax: bool = False) -> torch.Tensor:
    j = j_tensor.block_until_ready()
    if j.dtype == bool:
        j = j.astype(jnp.uint8)
        to_bool = True
    else:
        to_bool = False

    t = torch.utils.dlpack.from_dlpack(
        jax.dlpack.to_dlpack(j, take_ownership=delete_from_jax)
    )
    t = t.bool() if to_bool else t
    return t


def _jax_unbatched_fmls_fodf_seg(
    sphere_sample: jax.Array,
    nearest_sphere_samples_idx: jax.Array,
    nearest_sphere_samples_valid_mask: jax.Array,
    peak_diff_threshold: float,
) -> jax.Array:

    # From
    # R. E. Smith, J.-D. Tournier, F. Calamante, and A. Connelly,
    # "SIFT: Spherical-deconvolution informed filtering of tractograms," NeuroImage,
    # vol. 67, pp. 298–312, Feb. 2013, doi: 10.1016/j.neuroimage.2012.11.049.
    # Algorithm 1. in Appendix A.
    s = sphere_sample

    lobe_labels = jnp.zeros_like(s, dtype=jnp.int16)
    lobe_labels_index_space = jnp.arange(lobe_labels.shape[0])
    # Directions are sorted by absolute value, but the sign of their absolute value
    # will be needed later.
    # Get the indices of the sorted sample values, with each idx indexing into the
    # directions/sample values on the spherical function.
    sort_sample_idx = jnp.flip(jnp.argsort(jnp.abs(s)).astype(jnp.int16))
    sort_sample_val = jnp.take(s, sort_sample_idx)

    curr_max_lobe_label = 0

    def fmls_loop_body_fn(
        i_sample: int,
        state: dict,
        s,
        peak_diff_threshold,
        sort_sample_val,
        sort_sample_idx,
        lobe_labels_index_space,
        nearest_sphere_samples_idx,
        nearest_sphere_samples_valid_mask,
    ) -> dict:

        lobe_labels = state["lobe_labels"]
        curr_max_lobe_label = state["curr_max_lobe_label"]

        # i_sample gives the array index in the *sorted* fod samples, but we also need
        # the current sample's index from the original *unsorted* array of fod samples.
        idx_in_unsort_samples = sort_sample_idx[i_sample]

        # Get the 6 directions closest to the current sample direction by providing the
        # current sample's index over the spherical function samples. These adjacent
        # indices correspond to the unsorted sample values and unsorted theta/phi sphereical
        # coordinates.
        sample_adj_idx = nearest_sphere_samples_idx[idx_in_unsort_samples]

        # Adjacent lobe labels
        adj_lobe_label = jnp.take(lobe_labels, sample_adj_idx)
        # Some adjacent samples should not be considered such as when taking a sample from
        # a hemisphere near the equator.
        sample_adj_valid = nearest_sphere_samples_valid_mask[idx_in_unsort_samples]

        adj_lobe_label = adj_lobe_label * sample_adj_valid
        # Find the number of adjacent lobes in such a way that is compatible with
        # auto vectorization.
        # Sort the lobe labels themselves and and take the diff. The number of non-zero
        # diffs is equal to the number of unique labels, +/- 1. The +/- 1 is determined
        # by whether or not a $label_{l=0} - label_{l\neq0}$ is responsible for one of the
        # diffs being non-zero.
        n_unique_adj_lobe_labels = (
            (jnp.diff(adj_lobe_label.sort()) != 0).sum()
            + 1
            - 1 * (adj_lobe_label == 0).any()  # indicator for any 0s being present.
        )

        # Now handle the 3 conditionals without any explicit 'if' statements!
        # 1. If no lobes are adjacent to this sample.
        def zero_lobes_adj_fn(
            s,
            peak_diff_threshold,
            idx_in_unsort_samples,
            adj_lobe_label,
            lobe_labels,
            curr_max_lobe_label,
        ) -> dict:
            # If 0 lobes are adjacent, create a new lobe label and iterate on it.
            lobe_labels = lobe_labels.at[idx_in_unsort_samples].set(
                curr_max_lobe_label + 1
            )
            curr_max_lobe_label = curr_max_lobe_label + 1

            return dict(
                lobe_labels=lobe_labels, curr_max_lobe_label=curr_max_lobe_label
            )

        # 2. If 1 lobe is adjacent to this sample.
        def one_lobe_adj_fn(
            s,
            peak_diff_threshold,
            idx_in_unsort_samples,
            adj_lobe_label,
            lobe_labels,
            curr_max_lobe_label,
        ) -> dict:
            # If only 1 lobe is adjacent, assign the max (i.e., non-zero) adj lobe label.
            only_adj_label = adj_lobe_label.max()
            lobe_labels = lobe_labels.at[idx_in_unsort_samples].set(only_adj_label)
            return dict(
                lobe_labels=lobe_labels, curr_max_lobe_label=curr_max_lobe_label
            )

        # 3. If > 1 lobe is adjacent to this sample.
        def gt_one_lobe_adj_fn(
            s,
            peak_diff_threshold,
            idx_in_unsort_samples,
            adj_lobe_label,
            lobe_labels,
            curr_max_lobe_label,
        ) -> dict:

            # A 2D mask of n_samples x m_adj_samples can contain a label value/mask of all
            # samples that correspond to a label value, for all labels in adjacent
            # samples.
            max_num_adj_lobes = adj_lobe_label.shape[-1]
            adj_lobe_sample_mask = (
                jnp.tile(lobe_labels, (max_num_adj_lobes, 1)).T == adj_lobe_label
            ) * (adj_lobe_label != 0)
            # For the samples adjacent to sample i, collect the samples that have previously
            # been assigned a valid label.
            samples_of_adj_lobe_labels = jnp.where(
                adj_lobe_sample_mask, jnp.tile(s, (max_num_adj_lobes, 1)).T, jnp.nan
            )
            # Take the max sample value over each adjacent lobe label, even the
            # "invalid" labels (=0).
            adj_peaks = jnp.nanmax(samples_of_adj_lobe_labels, axis=0)
            # The min lobe label must not be 0, so we need to select the first non-zero lobe
            # label.
            min_adj_lobe_label_idx = jnp.nanargmin(adj_peaks)

            # Need both the smallest adjacent lobe's peak.
            min_adj_lobe_peak = adj_peaks[min_adj_lobe_label_idx]

            max_adj_lobe_label = adj_lobe_label[jnp.nanargmax(adj_peaks)]

            fod_sample = s[idx_in_unsort_samples]
            r = fod_sample / jnp.clip(min_adj_lobe_peak, 1e-6, jnp.inf)

            # 3.1 The difference between the current sample and the smallest peak of the
            # adjacent lobes is < the user-defined threshold.
            def _assign_largest(lobe_l, s_idx, adj_s_mask, large_l):
                return lobe_l.at[s_idx].set(large_l)

            # 3.2 Same comparison as 3.1, but >= the user-defined threshold.
            # Merge into the largest adjacent lobe, to keep the lobe indices sorted by peak
            # values.
            def _merge_adj(lobe_l, s_idx, adj_s_mask, large_l):
                return jnp.where(adj_s_mask.at[s_idx].set(True), large_l, lobe_l)

            lobe_labels = lax.cond(
                r < peak_diff_threshold,
                _assign_largest,
                _merge_adj,
                lobe_labels,
                idx_in_unsort_samples,
                jnp.sometrue(adj_lobe_sample_mask, axis=1),  # Collapse adj labels.
                max_adj_lobe_label,
            )

            return dict(
                lobe_labels=lobe_labels, curr_max_lobe_label=curr_max_lobe_label
            )

        segment_state_i = lax.switch(
            jnp.clip(n_unique_adj_lobe_labels, 0, 2),
            (zero_lobes_adj_fn, one_lobe_adj_fn, gt_one_lobe_adj_fn),
            s,
            peak_diff_threshold,
            idx_in_unsort_samples,
            adj_lobe_label,
            lobe_labels,
            curr_max_lobe_label,
        )
        lobe_labels_i = segment_state_i["lobe_labels"]
        curr_max_lobe_label_i = segment_state_i["curr_max_lobe_label"]

        return dict(
            lobe_labels=lobe_labels_i, curr_max_lobe_label=curr_max_lobe_label_i
        )

    state_0 = dict(
        lobe_labels=lobe_labels,
        curr_max_lobe_label=curr_max_lobe_label,
    )
    # Most parameters are constant over the loop, so just fill those in with a partial
    # function call.
    body_fn_with_consts = partial(
        fmls_loop_body_fn,
        s=s,
        peak_diff_threshold=peak_diff_threshold,
        sort_sample_val=sort_sample_val,
        sort_sample_idx=sort_sample_idx,
        lobe_labels_index_space=lobe_labels_index_space,
        nearest_sphere_samples_idx=nearest_sphere_samples_idx,
        nearest_sphere_samples_valid_mask=nearest_sphere_samples_valid_mask,
    )

    # There's no need to assign labels to fodf values of <= 0, so we can reduce the
    # number of iterations in the loop.
    t_max = jnp.where(sort_sample_val <= 0, jnp.arange(1, s.shape[0] + 1), jnp.nan)
    t_max = jnp.nanmin(t_max).astype(int)
    state_t = lax.fori_loop(0, t_max, body_fun=body_fn_with_consts, init_val=state_0)
    lobe_labels = state_t["lobe_labels"]

    # Labels are already in descending order by peak value, but they are not contiguous.
    # We can use the "diff(sort(...)) > 0" trick as above to remap to contiguous, sorted
    # lobe segmentation labels.
    label_sorted_idx = jnp.argsort(lobe_labels)
    unsort_label_idx = jnp.argsort(label_sorted_idx)
    lobe_labels = jnp.cumsum(
        jnp.diff(lobe_labels.at[label_sorted_idx].get(), prepend=0) > 0
    )
    lobe_labels = lobe_labels.at[unsort_label_idx].get()
    lobe_labels = lobe_labels.astype(jnp.int16)

    return lobe_labels


@jax.jit
def _jax_batched_fmls_fodf_seg(
    sphere_samples: jax.Array,
    nearest_sphere_samples_idx: jax.Array,
    nearest_sphere_samples_valid_mask: jax.Array,
    peak_diff_threshold: float,
) -> jax.Array:
    # Only vectorize over the fodf samples themselves; assume the same directional
    # configuration for each fodf to segment.
    vectorized_f = jax.vmap(_jax_unbatched_fmls_fodf_seg, in_axes=(0, None, None, None))

    return vectorized_f(
        sphere_samples,
        nearest_sphere_samples_idx,
        nearest_sphere_samples_valid_mask,
        peak_diff_threshold,
    )


def fmls_fodf_seg(
    sphere_samples: torch.Tensor,
    lobe_merge_ratio: float,
    theta: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    nearest_sphere_points = pitn.odf.adjacent_sphere_points_idx(theta=theta, phi=phi)
    nearest_sphere_samples_idx = nearest_sphere_points[0]
    nearest_sphere_samples_valid_mask = nearest_sphere_points[1]

    s_j = _t2j(sphere_samples)
    s_near_idx_j = _t2j(nearest_sphere_samples_idx)
    s_near_mask_j = _t2j(nearest_sphere_samples_valid_mask)

    lobe_labels_j = _jax_batched_fmls_fodf_seg(
        s_j, s_near_idx_j, s_near_mask_j, lobe_merge_ratio
    )

    lobe_labels = _j2t(lobe_labels_j, delete_from_jax=True)
    del lobe_labels_j

    return lobe_labels


def _contiguify_lobe_labels(
    peak_sorted_noncontiguous_lobe_labels: torch.Tensor,
) -> torch.Tensor:
    # Labels are already in descending order by peak value, but they are not contiguous.
    # We can use the "diff(sort(...)) > 0" trick to remap to contiguous, sorted
    # lobe segmentation labels.
    ll = peak_sorted_noncontiguous_lobe_labels
    label_sorted_idx = torch.argsort(ll, dim=1)
    unsort_label_idx = torch.argsort(label_sorted_idx, dim=1)
    unsorted_contiguous_ll = torch.cumsum(
        torch.diff(
            ll.take_along_dim(label_sorted_idx, dim=1),
            prepend=ll.new_zeros(ll.shape[0], 1),
            dim=1,
        ).to(ll.dtype)
        > 0,
        dim=1,
    )

    ll = unsorted_contiguous_ll.take_along_dim(unsort_label_idx, dim=1)
    return ll


def remove_fodf_labels_by_pdf(
    lobe_labels: torch.Tensor,
    sphere_samples: torch.Tensor,
    pdf_peak_min: float,
    pdf_integral_min: float,
) -> torch.Tensor:
    s = sphere_samples
    s_pdf = s - s.min(1, keepdim=True).values
    s_pdf = s_pdf / s_pdf.sum(1, keepdim=True)

    remapped_ll = torch.clone(lobe_labels)

    for l in lobe_labels.unique():

        select_s_pdf = torch.where(
            (lobe_labels == l) & (lobe_labels != 0), s_pdf, torch.nan
        )

        pdf_peaks = torch.nanquantile(select_s_pdf, 1.0, dim=1, keepdim=True)
        pdf_integrals = torch.nansum(select_s_pdf, dim=1, keepdim=True)
        select_l_mask = ~select_s_pdf.isnan()
        threshold_to_remove_batchwise = (pdf_peaks < pdf_peak_min) | (
            pdf_integrals < pdf_integral_min
        )
        to_remove_mask = select_l_mask & threshold_to_remove_batchwise
        remapped_ll.masked_fill_(to_remove_mask, 0)

    remapped_ll = _contiguify_lobe_labels(remapped_ll)
    return remapped_ll.to(lobe_labels.dtype)
