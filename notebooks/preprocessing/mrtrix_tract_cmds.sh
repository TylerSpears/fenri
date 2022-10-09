#!/bin/bash
set -euo pipefail

## Convert dwi + bval + bvec file into a more convenient format
if [ ! -s dwi.mif ]; then
    mrconvert -info -fslgrad \
        ../eddy/eddy_full_run.eddy_rotated_bvecs \
        ../eddy/sub-*.bval \
        ../eddy/eddy_full_run.nii* \
        dwi.mif
fi

# Copy this over to deepdream and run on there; it's a bit of a pain to compile and install
# ANTS for just this step.
#copyfiles dwi_mask.mif tyler@deepdream.ece.virginia.edu:/home/tyler/Projects/_sandbox
# Run on deepdream
###dwibiascorrect ants dwi.mif dwi_n4_correct.mif -info -mask dwi_mask.mif
#copyfiles tyler@deepdream.ece.virginia.edu:/home/tyler/Projects/_sandbox/dwi_n4_correct.mif .

if [ ! -s dwi_mask.mif ]; then
    dwi2mask -info -clean_scale 3 dwi.mif dwi_mask.mif
    maskfilter -largest dwi_mask.mif connect dwi_mask.mif -force
    mrcalc dwi.mif dwi_mask.mif -multiply dwi_masked.mif -force
    mrcalc dwi_n4_correct.mif dwi_mask.mif -multiply dwi_n4_correct_masked.mif -force
fi

# Anatomical images
mkdir --parents freesurfer_space
if [ ! -s freesurfer_space/freesurfer_t2w.mif ]; then
    mrconvert -info ../freesurfer/subj_dir/sub-002/mri/T2.masked.mgz freesurfer_space/freesurfer_t2w.mif
fi
if [ ! -s freesurfer_space/freesurfer_mask.mif ]; then
    mrconvert -info ../freesurfer/subj_dir/sub-002/mri/brainmask.mgz freesurfer_space/freesurfer_mask.mif \
        -datatype uint8
fi
if [ ! -s freesurfer_space/freesurfer_t1w.mif ]; then
    mrconvert -info ../freesurfer/subj_dir/sub-002/mri/brain.mgz freesurfer_space/freesurfer_t1w.mif
fi
if [ ! -s freesurfer_space/freesurfer_aparc_a2009s-aseg.mif ]; then
    mrconvert -info \
        ../freesurfer/subj_dir/sub-002/mri/aparc.a2009s+aseg.mgz \
        -datatype int32 \
        freesurfer_space/freesurfer_aparc_a2009s-aseg.mif
fi

if [ ! -s mean_b0_n4_correct.mif ]; then
    dwiextract dwi_n4_correct.mif - -bzero | mrmath - mean mean_b0_n4_correct.mif -axis 3
fi

# Create 5 tissue-type parcellation from freesurfer segmentation.
mkdir --parents dwi_orientation
if [ ! -s 5tt_parcellation_dwi-space.mif ]; then
    5ttgen -info freesurfer -sgm_amyg_hipp -nthreads 4 \
        freesurfer_space/freesurfer_aparc_a2009s-aseg.mif \
        freesurfer_space/5tt_parcellation_fs-space.mif \
        -force

    # Re-sample mean b0 for registration in anatomical space.
    mrgrid mean_b0_n4_correct.mif \
        regrid \
        dwi_orientation/mean_b0_n4_correct_fs-space.mif \
        -template freesurfer_space/freesurfer_t1w.mif \
        -interp sinc \
        -force
    # Register freesurfer T2w to the mean b0 orientation
    mrregister -type rigid \
        freesurfer_space/freesurfer_t2w.mif \
        -transformed dwi_orientation/t2w_fs-space_dwi-orient.mif \
        dwi_orientation/mean_b0_n4_correct_fs-space.mif \
        -rigid dwi_orientation/freesurfer2dwi_orient_matrix.txt \
        -force

    # Apply the same transform to the T1 and fs segmentation, and 5tt parcellation.
    mrtransform \
        -linear dwi_orientation/freesurfer2dwi_orient_matrix.txt \
        -interp sinc \
        -template dwi_orientation/mean_b0_n4_correct_fs-space.mif \
        freesurfer_space/freesurfer_t1w.mif \
        dwi_orientation/t1w_fs-space_dwi-orient.mif \
        -force
    mrtransform \
        -linear dwi_orientation/freesurfer2dwi_orient_matrix.txt \
        -interp nearest \
        -template dwi_orientation/mean_b0_n4_correct_fs-space.mif \
        freesurfer_space/freesurfer_aparc_a2009s-aseg.mif \
        dwi_orientation/aparc_a2009s-aseg_fs-space_dwi-orient.mif \
        -datatype int32 \
        -force
    mrtransform \
        -linear dwi_orientation/freesurfer2dwi_orient_matrix.txt \
        -interp nearest \
        -template dwi_orientation/mean_b0_n4_correct_fs-space.mif \
        freesurfer_space/5tt_parcellation_fs-space.mif \
        dwi_orientation/5tt_parcellation_fs-space_dwi-orient.mif \
        -datatype uint8 \
        -force

    # Also register the dwi-orient/freesurfer-space T2w to the mean b0 orientation
    # back in dwi space.
    mrregister -type rigid \
        dwi_orientation/t2w_fs-space_dwi-orient.mif \
        -transformed /tmp/.t2w_reg.mif \
        mean_b0_n4_correct.mif \
        -rigid dwi_orientation/dwi_orient2dwi_space_matrix.txt \
        -force
    rm /tmp/.t2w_reg.mif
    # We don't care about the registered image here, only the transform matrix.
    transformcompose \
        dwi_orientation/freesurfer2dwi_orient_matrix.txt \
        dwi_orientation/dwi_orient2dwi_space_matrix.txt \
        freesurfer2dwi_space_matrix.txt \
        -force

    # Apply the composed transform to bring freesurfer-space images into dwi space.
    mrtransform \
        -linear freesurfer2dwi_space_matrix.txt \
        -interp sinc \
        -template mean_b0_n4_correct.mif \
        freesurfer_space/freesurfer_t2w.mif \
        t2w_dwi-space.mif \
        -force
    mrtransform \
        -linear freesurfer2dwi_space_matrix.txt \
        -interp sinc \
        -template mean_b0_n4_correct.mif \
        freesurfer_space/freesurfer_t1w.mif \
        t1w_dwi-space.mif \
        -force
    mrtransform \
        -linear freesurfer2dwi_space_matrix.txt \
        -interp nearest \
        -template mean_b0_n4_correct.mif \
        freesurfer_space/freesurfer_aparc_a2009s-aseg.mif \
        aparc_a2009s-aseg_dwi-space.mif \
        -datatype int32 \
        -force
    mrtransform \
        -linear freesurfer2dwi_space_matrix.txt \
        -interp nearest \
        -template mean_b0_n4_correct.mif \
        freesurfer_space/5tt_parcellation_fs-space.mif \
        5tt_parcellation_dwi-space.mif \
        -datatype uint8
fi

## Generate a mask image at the gray matter-white matter interface for seeding tracks
if [ ! -s gm-wm_interface.mif ]; then
    5tt2gmwmi 5tt_parcellation_dwi-space.mif gm-wm_interface.mif
fi

## multi-tissue response estimation using 5tt tissue prior
if [ ! -s wm_response.txt ]; then
    dwi2response msmt_5tt -info \
        -mask dwi_mask.mif \
        dwi_n4_correct_masked.mif \
        5tt_parcellation_dwi-space.mif \
        wm_response.txt gm_response.txt csf_response.txt
fi

# multi-tissue, multi-shell CSD to create odfs
if [ ! -s wm_msmt_csd_fod.mif ]; then
    dwi2fod -info \
        -mask dwi_mask.mif \
        msmt_csd \
        dwi_n4_correct_masked.mif \
        wm_response.txt wm_msmt_csd_fod.mif \
        gm_response.txt gm_msmt_csd.mif \
        csf_response.txt csf_msmt_csd.mif
fi

# multi-shell, whole-brain tractography to be filtered down
tckgen -info -nthreads 6 \
    -act 5tt_parcellation_dwi-space.mif \
    -seed_gmwmi gm-wm_interface.mif \
    -select 100000000 \
    -minlength 20 \
    -maxlength 300 \
    -algorithm iFOD2 \
    wm_msmt_csd_fod.mif \
    ifod2_msmt_whole_brain_tracts.tck

### SIFT filtering
## filtering of multi-shell, whole-brain seeding tracts
tcksift -info -nthreads 6 \
    -term_number 10000000 \
    -act 5tt_parcellation_dwi-space.mif \
    ifod2_msmt_whole_brain_tracts.tck \
    wm_msmt_csd_fod.mif \
    ifod2_msmt_whole_brain_sift_reduce_tracts.tck
