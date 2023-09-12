#!/usr/bin/bash

set -eou pipefail

# Rough script for bringing freesurfer-derived partial volume maps to the ISMRM 2015
# space.

# ISMRM 2015 is based on HCP subject 100307, so we can use the high-res maps found
# in that subject.

# !NOTE: The freesurfer tool to be used is very sensitive when it comes to orientations
# !and target spaces. So, it's best to leave the moving space (freesurfer) volumes
# !unchanged, and the target space (ISMRM 2015 T1 space) should be changed instead.

# The HCP and ismrm T1 images have their x-axes flipped, so the registration target
# image needs to be flipped before registration. This flips the simulated T1w image
# created for ISMRM.
mrtransform ../lps_ismrm-2015_t1w.nii.gz -flip 0 rps_ismrm-2015_t1w.nii.g

# Register the HCP T1w to the ismrm T1w. They are from the same person, but the sim t1
# is centered differently.
# Make sure the freesurfer outputs for HCP subject 100307 are downloaded! The T1w_hires.nii.gz
# volume is in the same space as the aseg file we will use later, but its content is
# similar to the "fully processed" HCP T1w.
ANTSPATH="/opt/ants/ants-2.4.3/bin/" /opt/ants/ants-2.4.3/bin/antsRegistrationSyN.sh \
    -d 3 \
    -m /data/srv/data/pitn/hcp/100307/T1w/freesurfer/100307/mri/T1w_hires.nii.gz \
    -f rps_ismrm-2015_t1w.nii.gz \
    -x "../lps_ismrm-2015_brain_mask.nii.gz","/data/srv/data/pitn/hcp/100307/T1w/brainmask_fs.nii.gz" \
    -n 20 \
    -t r \
    -p d \
    -j 1 \
    -o hcp-100307_reg_ismrm-2015_
# Convert ANTS affine to a text ITK affine file.
ANTSPATH="/opt/ants/ants-2.4.3/bin/" /opt/ants/ants-2.4.3/bin/ConvertTransformFile \
    3 \
    "hcp-100307_reg_ismrm-2015_0GenericAffine.mat" \
    "hcp-100307_reg_ismrm-2015_0GenericAffine.txt"
# Convert the ITK affine text file to a freesurfer LTA file. This file lets us use
# freesurfer's partial volume estimator.
lta_convert \
    -initk hcp-100307_reg_ismrm-2015_0GenericAffine.txt \
    -outlta hcp-100307_reg_ismrm-2015_0GenericAffine.lta \
    --src /data/srv/data/pitn/hcp/100307/T1w/freesurfer/100307/mri/T1w_hires.nii.gz \
    --trg rps_ismrm-2015_t1w.nii.gz \
    --subject 100307

# Run freesurfer's partial volume fraction estimator.
# Some of the CLI args are incorrect or undocumented, you'll have to see the source code
# <https://github.com/freesurfer/freesurfer/blob/bc5f45fea71dd14738f6a44057e6abd048ddb291/mri_compute_volume_fractions/mri_compute_volume_fractions.cpp>
# for the correct details arguments.
mkdir --parents "hcp-100307_pv_maps"
OMP_NUM_THREADS=20 SUBJECTS_DIR="/data/srv/data/pitn/hcp/100307/T1w/freesurfer" \
    mri_compute_volume_fractions --debug --nii.gz \
    --seg aseg.hires.mgz \
    --ndil 1 \
    --usf 3 \
    --o "/home/tas6hh/Projects/pitn/notebooks/sandbox/ismrm_2015/hcp-100307_reg_ismrm-2015/hcp-100307_pv_maps/hcp-100307_reg_ismrm-2015_pv" \
    --reg "/home/tas6hh/Projects/pitn/notebooks/sandbox/ismrm_2015/hcp-100307_reg_ismrm-2015/hcp-100307_reg_ismrm-2015_0GenericAffine.lta"

# Don't forget to flip the result(s) back to LPS
mrtransform "hcp-100307_pv_maps/hcp-100307_reg_ismrm-2015_pv.csf.nii.gz" -flip 0 \
    "lps_hcp-100307_reg_ismrm-2015_csf_vol_fraction.nii.gz"

# A precise tissue mask can also be created by summing up all pvs, thresholding voxels
# that have a volume fraction of > 0, and using scipy's `scipy.ndimage.binary_fill_holes`
# function.
