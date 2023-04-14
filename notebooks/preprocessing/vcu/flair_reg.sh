#!/usr/bin/bash

set -eou pipefail

# Commands used to register the FLAIR image to DWI space
#TODO convert into a usable script, rather than commands on hard-coded files.

# Set the freesurfer-space T1w as the target for registration.
mrconvert ../../../tractography/P_07/freesurfer_space/freesurfer_t1w.mif T1w.nii.gz
# Re-set strides for the FLAIR image to match the target space, otherwise registration
# can mis-behave.
mrtransform flair.nii.gz -strides T1w.nii.gz flair_fs-orient.nii.gz
# Rigid register FLAIR to T1w.
flirt -v -in flair_fs-orient.nii.gz -ref T1w.nii.gz -dof 6 -omat flirt2freesurfer_affine_fsl.txt

# Compose flair->freesurfer and previously estimated freesurfer->dwi transformations 
transformconvert flair2freesurfer_affine_fsl.txt flair_fs-orient.nii.gz T1w.nii.gz flirt_import flair2freesurfer_affine_mrtrix.txt 
transformcompose flair2freesurfer_affine_mrtrix.txt ../../../tractography/P_07/dwi_orientation/freesurfer2dwi_orient_affine.txt flair2dwi_affine.txt
mrtransform \
        flair_fs-orient.nii.gz \
        -linear flair2dwi_affine.txt \
        -template ../../../tractography/P_07/dwi_orientation/mean_b0_fs-space.nii.gz \
        -interp sinc \
        -strides ../../../tractography/P_07/dwi_orientation/mean_b0_fs-space.nii.gz \
        flair_reg-dwi.nii.gz

# Correct ringing artifacts from sinc interpolation.
mrcalc -info -force flair_reg-dwi.nii.gz 0 -lt 0 flair_reg-dwi.nii.gz -if flair_reg-dwi.nii.gz 
flair_max=$(mrstats flair.nii.gz -output max)
mrcalc -force flair_reg-dwi.nii.gz $flair_max -gt $flair_max flair_reg-dwi.nii.gz -if flair_reg-dwi.nii.gz 

# Apply freesurfer's mask to the FLAIR image.
mrtransform \
        ../../../tractography/P_07/freesurfer_space/freesurfer_mask.mif \
        -linear ../../../tractography/P_07/dwi_orientation/freesurfer2dwi_orient_affine.txt \
        -interp nearest \
        -template T1w.nii.gz - |
        mrcalc \
        - \
        flair_reg-dwi.mif \
        -mult \
        flair_reg-dwi_brain.nii.gz
