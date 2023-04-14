#!/bin/bash

set -eou pipefail

REF_IM="../../reference_density_map.nii.gz"
TCK_SOURCE_DIR="../tck_files"

for f_tck in "${TCK_SOURCE_DIR}"/*.tck; do
        tck_name=$(basename "$f_tck" .tck)
        echo "$tck_name"

        f_unfilled_mask=".tmp.${tck_name}_unfilled.nii.gz"
        mrcalc -info \
                "$REF_IM" 0 -mult 1 -add - |
                tckmap -info \
                        "$f_tck" -image - \
                        -template "$REF_IM" \
                        -contrast scalar_map_count - |
                mrthreshold - -abs 1 "$f_unfilled_mask"
        f_filled_mask="${tck_name}_occupancy_mask.nii.gz"
        ./refine_occupancy_mask_nifti.py "$f_unfilled_mask" "$f_filled_mask"
done

rm .tmp.*.nii.gz
