#!/bin/bash
set -euo pipefail

SUBJ_ID="P_07"
EDDY_DIR="../../preproc/${SUBJ_ID}/eddy"
FREESURFER_DIR="../../preproc/${SUBJ_ID}/freesurfer/${SUBJ_ID}"

## Convert dwi + bval + bvec file into a more convenient format
if [ ! -s dwi.mif ]; then
    mrconvert -info -fslgrad \
        "$EDDY_DIR"/"$SUBJ_ID"*.eddy_rotated_bvecs \
        "$EDDY_DIR"/"$SUBJ_ID"*.bval \
        "$EDDY_DIR"/"$SUBJ_ID"_eddy.nii* \
        dwi.mif
fi

if [ ! -s dwi_mask.mif ]; then
    dwi2mask -info -clean_scale 3 dwi.mif dwi_mask.mif
    maskfilter -largest dwi_mask.mif connect dwi_mask.mif -force
    mrcalc dwi.mif dwi_mask.mif -multiply dwi.mif -force
fi

# Anatomical images
mkdir --parents freesurfer_space
if [ ! -s freesurfer_space/freesurfer_mask.mif ]; then
    mrconvert -info "${FREESURFER_DIR}/mri/brainmask.mgz" -datatype uint8 - |
            mrthreshold - -abs 0 -comparison gt \
            freesurfer_space/freesurfer_mask.mif
    # Perform binary closing of small pepper holes.
    maskfilter -info \
            freesurfer_space/freesurfer_mask.mif \
            dilate -npass 5 - |
            maskfilter - \
            erode -npass 5 \
            freesurfer_space/freesurfer_mask.mif -force
fi
if [ ! -s freesurfer_space/freesurfer_t2w.mif ]; then
    mrconvert -info "${FREESURFER_DIR}/mri/T2.mgz" freesurfer_space/freesurfer_unmasked_t2w.mif
    mrcalc freesurfer_space/freesurfer_unmasked_t2w.mif freesurfer_space/freesurfer_mask.mif -multiply \
            freesurfer_space/freesurfer_t2w.mif
fi
if [ ! -s freesurfer_space/freesurfer_t1w.mif ]; then
    mrconvert -info "${FREESURFER_DIR}/mri/brain.mgz" freesurfer_space/freesurfer_t1w.mif
fi
if [ ! -s freesurfer_space/freesurfer_aparc_a2009s-aseg.mif ]; then
    mrconvert -info \
        "${FREESURFER_DIR}/mri/aparc.a2009s+aseg.mgz" \
        -datatype int32 \
        freesurfer_space/freesurfer_aparc_a2009s-aseg.mif
fi

if [ ! -s mean_b0.mif ]; then
    dwiextract dwi.mif - -bzero | mrmath - mean mean_b0.mif -axis 3
fi

# Create 5 tissue-type parcellation from freesurfer segmentation.
mkdir --parents dwi_orientation
if [ ! -s 5tt_parcellation_dwi-space.nii.gz ]; then
    5ttgen -info freesurfer -sgm_amyg_hipp -nthreads 5 \
        freesurfer_space/freesurfer_aparc_a2009s-aseg.mif \
        freesurfer_space/5tt_parcellation_fs-space.mif \
        -force

    # Re-sample mean b0 for registration in anatomical space.
    mrgrid -force \
        mean_b0.mif \
        regrid \
        -template freesurfer_space/freesurfer_t1w.mif \
        -strides mean_b0.mif \
        -interp sinc \
        dwi_orientation/mean_b0_fs-space.mif
    # Clip min and max values to remove ringing artifacts from sinc interpolation. 
    mrcalc -info -force \
       dwi_orientation/mean_b0_fs-space.mif 0 -lt 0 dwi_orientation/mean_b0_fs-space.mif -if \
       dwi_orientation/mean_b0_fs-space.mif
    b0_max=$(mrstats mean_b0.mif -output max)
    mrcalc -info -force \
       dwi_orientation/mean_b0_fs-space.mif $b0_max -gt \
       $b0_max \
       dwi_orientation/mean_b0_fs-space.mif \
       -if \
       dwi_orientation/mean_b0_fs-space.mif

    # Register freesurfer T2w to the mean b0 orientation
    mrtransform -force -info freesurfer_space/freesurfer_t2w.mif \
            -strides dwi_orientation/mean_b0_fs-space.mif \
                dwi_orientation/.tmp.t2w_fs.nii.gz
    mrconvert -force -info \
            dwi_orientation/mean_b0_fs-space.mif \
            dwi_orientation/mean_b0_fs-space.nii.gz
    flirt -v \
            -in dwi_orientation/.tmp.t2w_fs.nii.gz \
            -ref dwi_orientation/mean_b0_fs-space.nii.gz \
            -dof 6 \
            -omat dwi_orientation/freesurfer2dwi_orient_fsl-affine.txt
    transformconvert dwi_orientation/freesurfer2dwi_orient_fsl-affine.txt \
            dwi_orientation/.tmp.t2w_fs.nii.gz \
            dwi_orientation/mean_b0_fs-space.nii.gz \
            flirt_import \
            dwi_orientation/freesurfer2dwi_orient_affine.txt
    rm dwi_orientation/.tmp.t2w_fs.nii.gz

    mrtransform -info -force \
        freesurfer_space/freesurfer_t2w.mif \
        -linear dwi_orientation/freesurfer2dwi_orient_affine.txt \
        -interp sinc \
        -template dwi_orientation/mean_b0_fs-space.mif \
        -strides dwi_orientation/mean_b0_fs-space.mif \
        dwi_orientation/t2w_fs-space_dwi-orient.mif
    # Clip min and max values to remove ringing artifacts from sinc interpolation. 
    mrcalc -info -force \
        dwi_orientation/t2w_fs-space_dwi-orient.mif 0 -lt \
        0 \
        dwi_orientation/t2w_fs-space_dwi-orient.mif \
        -if \
        dwi_orientation/t2w_fs-space_dwi-orient.mif
    t2w_max=$(mrstats freesurfer_space/freesurfer_t2w.mif -output max)
    mrcalc -info -force \
        dwi_orientation/t2w_fs-space_dwi-orient.mif $t2w_max -gt \
        $t2w_max \
        dwi_orientation/t2w_fs-space_dwi-orient.mif \
        -if \
        dwi_orientation/t2w_fs-space_dwi-orient.mif

    # Apply the same transform to the T1 and fs segmentation, and 5tt parcellation.
    mrtransform -info -force \
        freesurfer_space/freesurfer_t1w.mif \
        -linear dwi_orientation/freesurfer2dwi_orient_affine.txt \
        -interp sinc \
        -template dwi_orientation/mean_b0_fs-space.mif \
        -strides dwi_orientation/mean_b0_fs-space.mif \
        dwi_orientation/t1w_fs-space_dwi-orient.mif
    # Clip min and max values to remove ringing artifacts from sinc interpolation. 
    mrcalc -info -force \
        dwi_orientation/t1w_fs-space_dwi-orient.mif 0 -lt \
        0 \
        dwi_orientation/t1w_fs-space_dwi-orient.mif \
        -if \
        dwi_orientation/t1w_fs-space_dwi-orient.mif
    t1w_max=$(mrstats freesurfer_space/freesurfer_t1w.mif -output max)
    mrcalc -info -force \
        dwi_orientation/t1w_fs-space_dwi-orient.mif $t1w_max -gt \
        $t1w_max \
        dwi_orientation/t1w_fs-space_dwi-orient.mif \
        -if \
        dwi_orientation/t1w_fs-space_dwi-orient.mif

    mrtransform -info -force \
        freesurfer_space/freesurfer_aparc_a2009s-aseg.mif \
        -linear dwi_orientation/freesurfer2dwi_orient_affine.txt \
        -interp nearest \
        -template dwi_orientation/mean_b0_fs-space.mif \
        -strides dwi_orientation/mean_b0_fs-space.mif \
        dwi_orientation/aparc_a2009s-aseg_fs-space_dwi-orient.mif \
        -datatype int32

    mrtransform -info -force \
        freesurfer_space/5tt_parcellation_fs-space.mif \
        -linear dwi_orientation/freesurfer2dwi_orient_affine.txt \
        -interp nearest \
        -template dwi_orientation/mean_b0_fs-space.mif \
        -strides dwi_orientation/mean_b0_fs-space.mif \
        dwi_orientation/5tt_parcellation_fs-space_dwi-orient.mif \
        -datatype uint8

    # Re-sample freesurfer images to align with DWI orientation and voxel spacing.
    # T2w
    mrtransform -info -force \
        freesurfer_space/freesurfer_t2w.mif \
        -linear dwi_orientation/freesurfer2dwi_orient_affine.txt \
        -interp sinc \
        -template mean_b0.mif \
        -strides mean_b0.mif \
        t2w_dwi-space.nii.gz
    # Clip min and max values to remove ringing artifacts from sinc interpolation. 
    mrcalc -info -force \
        t2w_dwi-space.nii.gz 0 -lt \
        0 \
        t2w_dwi-space.nii.gz \
        -if \
        t2w_dwi-space.nii.gz
    t2w_max=$(mrstats freesurfer_space/freesurfer_t2w.mif -output max)
    mrcalc -info -force \
        t2w_dwi-space.nii.gz $t2w_max -gt \
        $t2w_max \
        t2w_dwi-space.nii.gz\
        -if \
        t2w_dwi-space.nii.gz 

    # T1w
    mrtransform -info -force \
        freesurfer_space/freesurfer_t1w.mif \
        -linear dwi_orientation/freesurfer2dwi_orient_affine.txt \
        -interp sinc \
        -template mean_b0.mif \
        -strides mean_b0.mif \
        t1w_dwi-space.nii.gz
    # Clip min and max values to remove ringing artifacts from sinc interpolation. 
    mrcalc -info -force \
        t1w_dwi-space.nii.gz 0 -lt 0 t1w_dwi-space.nii.gz -if \
        t1w_dwi-space.nii.gz
    t1w_max=$(mrstats freesurfer_space/freesurfer_t1w.mif -output max)
    mrcalc -info -force \
        t1w_dwi-space.nii.gz $t1w_max -gt \
        $t1w_max \
        t1w_dwi-space.nii.gz \
        -if \
        t1w_dwi-space.nii.gz

    # a2009 segmentation mask
    mrtransform -info -force \
        freesurfer_space/freesurfer_aparc_a2009s-aseg.mif \
        -linear dwi_orientation/freesurfer2dwi_orient_affine.txt \
        -interp nearest \
        -template mean_b0.mif \
        -strides mean_b0.mif \
        -datatype int32 \
        aparc_a2009s-aseg_dwi-space.nii.gz

    # Five-tissue-type parcellation
    mrtransform -info -force \
        freesurfer_space/5tt_parcellation_fs-space.mif \
        -linear dwi_orientation/freesurfer2dwi_orient_affine.txt \
        -interp nearest \
        -template mean_b0.mif \
        -datatype uint8 \
        5tt_parcellation_dwi-space.nii.gz
fi

## Refine DWI mask based on the Freesurfer skull stripping
if [ ! -s mask.nii.gz ]; then
    mrtransform -info -force \
            freesurfer_space/freesurfer_mask.mif \
            -linear dwi_orientation/freesurfer2dwi_orient_affine.txt \
            -interp nearest \
            -template mean_b0.mif \
            -datatype uint8 \
            freesurfer_mask_dwi-space.mif
    # freesurfer excludes voxels that are kept in the DWI mask, so add onto the 
    # dwi-freesurfer intersection mask by thresholding the mean b0 values.
    # Calculate dwi_mask Intersection !freesurfer_mask -> dilate by 2 -> keep largest
    # component -> mask Intersection dwi_mask [to remove dilated voxels outside the
    # original DWI mask, which wouldn't exist in the DWI anyway]
    mrcalc -force \
            dwi_mask.mif \
            freesurfer_mask_dwi-space.mif -not \
            -and \
            mean_b0.mif \
            -mult \
            .tmp.mean_b0-mask-select.mif
    mrthreshold .tmp.mean_b0-mask-select.mif - |
            maskfilter - dilate -npass 2 - |
            maskfilter - \
                connect -largest \
                - |
            mrcalc \
                dwi_mask.mif \
                - \
                -and \
                .tmp.addon_mask_to_fs-mrtrix-intersection.mif
    mrcalc -info -force \
            dwi_mask.mif \
            freesurfer_mask_dwi-space.mif \
            -and \
            .tmp.addon_mask_to_fs-mrtrix-intersection.mif \
            -or \
            mask.nii.gz

    rm .tmp.addon_mask_to_fs-mrtrix-intersection.mif
    
    # Re-mask DWI and all freesurfer-derived segmentations.
    mrcalc -info -force \
            dwi.mif \
            mask.nii.gz \
            -mult \
            dwi.mif

    mrcalc -info -force \
            mean_b0.mif \
            mask.nii.gz \
            -mult \
            mean_b0.mif

    mrcalc -info -force \
            5tt_parcellation_dwi-space.nii.gz \
            mask.nii.gz \
            -mult \
            5tt_parcellation_dwi-space.nii.gz
    mrcalc -info -force \
            aparc_a2009s-aseg_dwi-space.nii.gz \
            mask.nii.gz \
            -mult \
            aparc_a2009s-aseg_dwi-space.nii.gz
fi

## Generate a mask image at the gray matter-white matter interface for seeding tracks
if [ ! -s gm-wm_interface.nii.gz ]; then
    5tt2gmwmi 5tt_parcellation_dwi-space.nii.gz -mask_in mask.nii.gz gm-wm_interface.nii.gz
fi

## multi-tissue response estimation using 5tt tissue prior
if [ ! -s wm_response.txt ]; then
    dwi2response msmt_5tt -info \
        -mask mask.nii.gz \
        dwi.mif \
        5tt_parcellation_dwi-space.nii.gz \
        wm_response.txt gm_response.txt csf_response.txt
fi

# multi-tissue, multi-shell CSD to create odfs
if [ ! -s wm_msmt_csd_fod.nii.gz ]; then
    dwi2fod -info \
        -mask mask.nii.gz \
        msmt_csd \
        dwi.mif \
        wm_response.txt wm_msmt_csd_fod.nii.gz \
        gm_response.txt gm_msmt_csd.nii.gz \
        csf_response.txt csf_msmt_csd.nii.gz
fi

# # multi-shell, whole-brain tractography to be filtered down
# tckgen -info -nthreads 6 \
#     -act 5tt_parcellation_dwi-space.mif \
#     -seed_gmwmi gm-wm_interface.mif \
#     -select 100000000 \
#     -minlength 20 \
#     -maxlength 300 \
#     -algorithm iFOD2 \
#     wm_msmt_csd_fod.mif \
#     ifod2_msmt_whole_brain_tracts.tck
# 
# ### SIFT filtering
# ## filtering of multi-shell, whole-brain seeding tracts
# tcksift -info -nthreads 6 \
#     -term_number 10000000 \
#     -act 5tt_parcellation_dwi-space.mif \
#     ifod2_msmt_whole_brain_tracts.tck \
#     wm_msmt_csd_fod.mif \
#     ifod2_msmt_whole_brain_sift_reduce_tracts.tck
