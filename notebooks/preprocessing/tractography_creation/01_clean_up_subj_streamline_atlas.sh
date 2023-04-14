#!/bin/bash

set -eou pipefail

GM_WM_INTERFACE="/data/srv/outputs/pitn/hcp/7T/fodf/200210/T1w/postproc_gm-wm-interface_seed-mask.nii.gz"
ATLAS_OCCUPANCY_MASK_DIR="/data/srv/data/pitn/atlas/Atlas_30_Bundles/bundles/track_occupancy_segmentations"
ATLAS_TO_SUBJ_PULLBACK_WARP="/data/srv/data/pitn/hcp/200210/T1w/reg_MNI-2009a/reg_tractography/tck_mrtrix_warp_corrected.nii.gz"
STREAMLINE_BUNDLES_SUBJ_SPACE_DIR="/data/srv/outputs/pitn/hcp/7T/fodf/200210/T1w/tractography/recobundles_no-slr/subj_space/tck_files"
SUBJ_OUTPUT_OCCUPANCY_MASK_DIR="/data/srv/outputs/pitn/hcp/7T/fodf/200210/T1w/tractography/recobundles_no-slr/subj_space/track_occupancy_segmentations"
SUBJ_OUTPUT_REFINE_STREAMLINE_DIR="/data/srv/outputs/pitn/hcp/7T/fodf/200210/T1w/tractography/recobundles_no-slr/subj_space/truncated_tck_files"

seed_mask="/tmp/.tmp.gm-wm-interface_mask.nii.gz"
mrthreshold -force "$GM_WM_INTERFACE" -abs 0.001 "$seed_mask"

for f_mask in "${ATLAS_OCCUPANCY_MASK_DIR}"/*.nii.gz; do

    streamline_name=$(basename "$f_mask" .nii.gz)
    streamline_name="${streamline_name/_occupancy_mask/}"
    echo "$streamline_name"
    echo "================================"

    subj_occupancy_mask="${SUBJ_OUTPUT_OCCUPANCY_MASK_DIR}/${streamline_name}_occupancy_mask_subj-space.nii.gz"
    mrtransform -info -force \
        -interp linear \
        "$f_mask" \
        -warp "$ATLAS_TO_SUBJ_PULLBACK_WARP" \
        - |
        mrthreshold -force - -abs 0.001 "$subj_occupancy_mask"

    full_streamline_roi="/tmp/.tmp.${streamline_name}_filter-roi_subj-space.nii.gz"
    mrcalc -info -force \
        "$seed_mask" "$subj_occupancy_mask" -or -datatype uint8 "$full_streamline_roi"

    subj_space_streamlines="$STREAMLINE_BUNDLES_SUBJ_SPACE_DIR"/"${streamline_name}.tck"


    # Some tractograms may not have every streamline parcellated, so skip those that
    # were not found.
    if [[ -s "$subj_space_streamlines" ]]; then

        unmarked_streams="/tmp/.tmp.unmarked_${streamline_name}.tck"
        marked_streams="/tmp/.tmp.marked_${streamline_name}.tck"
        ./mark_streamlines_for_truncation.py \
            "$subj_space_streamlines" \
            "$GM_WM_INTERFACE" \
            "$full_streamline_roi" \
            0.75 \
            "$unmarked_streams" \
            "$marked_streams"

        streams_to_concat="$unmarked_streams"
        if [[ -s "$marked_streams" ]]; then
            surviving_tracks="/tmp/.tmp.surviving_${streamline_name}.tck"
            echo "$full_streamline_roi"
            echo "$marked_streams"
            echo "$surviving_tracks"

            tckedit -force -info -mask "$full_streamline_roi" "$marked_streams" "/tmp/.tmp.tck"
            tckedit -force -info -minlength 20 "/tmp/.tmp.tck" "$surviving_tracks"

            streams_to_concat="$streams_to_concat $surviving_tracks"
        else
            echo "******Streamline file ${subj_space_streamlines} does not exist!******"
            echo "Skipping"
        fi

        output_refine_track="$SUBJ_OUTPUT_REFINE_STREAMLINE_DIR"/"${streamline_name}.tck"
        # Don't quote the variable, we want tckedit to "word split" the potentially > 1
        # file(s).
        tckedit -force -info $streams_to_concat "$output_refine_track"
    fi
    echo "================================"

done

rm /tmp/.tmp.*.nii.gz
rm /tmp/.tmp.*.tck
