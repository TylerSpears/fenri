#!/usr/bin/bash

set -eou pipefail

nifti_to_fix="$1"
dsistudio_input_dwi_nifti="$2"
output_nifti="$3"
mrtransform_arguments="${@:4}"

mrtransform "$nifti_to_fix" -strides "$dsistudio_input_dwi_nifti" "$output_nifti" $mrtransform_arguments
fslcpgeom "$dsistudio_input_dwi_nifti" "$output_nifti" -d
