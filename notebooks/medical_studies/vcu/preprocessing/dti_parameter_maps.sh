#!/usr/bin/bash

set -eou pipefail

SUBJ_ID=$1
DWI="$2"
BVAL="$3"
BVEC="$4"
MASK="$5"
DTI_OUT_DIR="$6"

N_THREADS=10

mkdir --parents "$DTI_OUT_DIR"

output_prefix="${DTI_OUT_DIR}/${SUBJ_ID}"
dti="${output_prefix}_dti.nii.gz"

if [ ! -s "$dti" ] || [ "$DWI" -nt "$dti" ]; then
    dwi2tensor --force -info -nthreads $N_THREADS \
        "$DWI" \
        -fslgrad "$BVEC" "$BVAL" \
        -mask "$MASK" \
        "$dti"
fi

# Use FA as a proxy for all metrics.
fa="${output_prefix}_fa.nii.gz"
if [ ! -s "$fa" ] || [ "$dti" -nt "$fa" ]; then
    tensor2metric --force -info -nthreads $N_THREADS \
        "$dti" \
        -mask "$MASK" \
        -adc "${output_prefix}_adc.nii.gz" \
        -fa "$fa" \
        -ad "${output_prefix}_ad.nii.gz" \
        -rd "${output_prefix}_rd.nii.gz" \
        -num 1,2,3 \
        -value "${output_prefix}_eigenvals.nii.gz" \
        -vector "${output_prefix}_eigenvecs.nii.gz" \
        -cl "${output_prefix}_linearity_metric.nii.gz" \
        -cp "${output_prefix}_planarity_metric.nii.gz" \
        -cs "${output_prefix}_sphericity_metric.nii.gz"
fi
