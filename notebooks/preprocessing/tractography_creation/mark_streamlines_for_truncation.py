#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import dipy
import dipy.io
import dipy.io.streamline
import nibabel as nib
import numpy as np
import scipy


def track_point_lens_weights(arr_track, r):
    amt_in_roa = scipy.ndimage.map_coordinates(r, arr_track.T, order=1)
    weights = (amt_in_roa[:-1] + amt_in_roa[1:]) / 2
    lens = np.linalg.norm(arr_track[:-1] - arr_track[1:], ord=2, axis=1)
    return lens, weights


if __name__ == "__main__":

    assert len(sys.argv) == 6 + 1
    args = sys.argv[1:]

    f_input_streamline = Path(str(args[0])).resolve()
    ref = str(args[1])
    if ref.casefold() != "same":
        ref = Path(ref).resolve()
    else:
        ref = ref.casefold()

    f_roi = Path(str(args[2])).resolve()

    min_proportion_in_roi = float(args[3])
    max_proportion_in_roa = 1 - min_proportion_in_roi

    f_output_unmarked = str(args[4])
    f_output_marked = str(args[5])
    print(args)

    assert f_input_streamline.exists()
    if isinstance(ref, Path):
        assert ref.exists()
    assert f_roi.exists()

    f_input_streamline = str(f_input_streamline)
    ref = str(ref)

    tracks = dipy.io.streamline.load_tractogram(f_input_streamline, reference=ref)
    tracks.to_vox()

    roi = nib.load(f_roi).get_fdata().astype(bool)
    roa = ~roi
    roa_float = roa.astype(np.float32)
    marked_streamlines = list()
    unmarked_streamlines = list()
    for i, s in enumerate(tracks.streamlines):
        l, w = track_point_lens_weights(s, roa_float)
        ratio = (l * w).sum() / l.sum()
        if i % 1000 == 0:
            print(i, end="...", flush=True)

        if ratio > max_proportion_in_roa:
            marked_streamlines.append(s)
        else:
            unmarked_streamlines.append(s)
    print()

    unmarked_tracks = tracks.from_sft(unmarked_streamlines, tracks)
    dipy.io.streamline.save_tractogram(unmarked_tracks, f_output_unmarked)

    if len(marked_streamlines) > 0:
        marked_tracks = tracks.from_sft(marked_streamlines, tracks)
        dipy.io.streamline.save_tractogram(marked_tracks, f_output_marked)
    else:
        print("No streamlines marked for truncation.")
