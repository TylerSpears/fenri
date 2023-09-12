#!/usr/bin/bash

set -eou pipefail

odf_f="$1"
seed_dirs_f="$2"
brain_mask_f="$3"
peaks_out_dir="$4"

SCRIPT_F="$(basename "$0")"
# Default to all processors, if N_PROCS is not set.
N_PROCS=${N_PROCS-"$(nproc)"}

mkdir --parents "$peaks_out_dir"
peak_dirs_f="$peaks_out_dir/peak_dirs_xyz.nii.gz"
peak_amp_f="$peaks_out_dir/peak_amplitudes.nii.gz"

subj_seed_dirs_f=$peaks_out_dir/"$(basename "$seed_dirs_f")"
cp --update --archive --force "$seed_dirs_f" "$subj_seed_dirs_f"

num_peaks=3
thresh_amp=0.1

# Ensure input odf is in RAS orientation.
mrconvert "$odf_f" -strides 1,2,3 - |
    sh2peaks -info -nthreads $N_PROCS \
        - \
        -num $num_peaks \
        -threshold $thresh_amp \
        -seeds "$subj_seed_dirs_f" \
        -mask "$brain_mask_f" \
        "$peak_dirs_f"
# Extract amplitudes.
peaks2amp -info \
    "$peak_dirs_f" \
    "$peak_amp_f"

subj_code_dir=$(dirname "$(dirname "$(realpath "$odf_f")")")/code
mkdir --parents "$subj_code_dir"
cp --update --archive --force "$0" "${subj_code_dir}/${SCRIPT_F}"
