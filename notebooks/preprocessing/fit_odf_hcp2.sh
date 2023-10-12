#!/usr/bin/bash

set -eou pipefail

dwi_f="$1"
bvec_f="$2"
bval_f="$3"
mask_f="$4"
fs_aseg_f="$5"
fs_brain_mask_f="$6"
output_root_dir="$7"

# Default to all processors, if N_PROCS is not set.
N_PROCS=${N_PROCS-"$(nproc)"}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS

DWINORMALISE_CSF_PERCENTILE=${DWINORMALISE_CSF_PERCENTILE:-"75"}

mkdir --parents "$output_root_dir"
tmp_dir="${output_root_dir}/tmp"
mkdir --parents "$tmp_dir"
dwi_mif_f="${tmp_dir}/dwi.mif"

diffusion_dir="${output_root_dir}/diffusion"
seg_dir="${output_root_dir}/segmentation"
mkdir --parents "$diffusion_dir" "$seg_dir"
# Convert dwi to .mif and crop by mask for convenience
mrconvert -nthreads $N_PROCS \
    "$dwi_f" \
    -fslgrad "$bvec_f" "$bval_f" \
    - |
    mrconvert \
        - \
        -strides 1,2,3 \
        - |
    mrgrid -nthreads $N_PROCS \
        - \
        crop \
        -mask "$mask_f" -crop_unbound \
        - |
    mrgrid -force -nthreads $N_PROCS \
        - \
        pad \
        -uniform 3 \
        "$dwi_mif_f"
# Crop the brain mask, too.
crop_mask_f="${output_root_dir}/brain_mask.nii.gz"
mrconvert \
    "$mask_f" \
    -strides 1,2,3 \
    - |
    mrgrid -nthreads $N_PROCS \
        - \
        crop \
        -mask "$mask_f" -crop_unbound \
        - |
    mrgrid -force -nthreads $N_PROCS \
        - \
        pad \
        -uniform 3 \
        "$crop_mask_f"
strides_spec_f="$crop_mask_f"
# Normalize the DWI intensities such that some percentile of the CSF has intensity 1000
# in the b=0 volumes.
# We need a mask of CSF voxels, which is most easily found using the fivett map that
# we'll also need later.
fs_fivett_f="${seg_dir}/fivett_anat-space_segmentation.nii.gz"
5ttgen freesurfer -force \
    -nocrop -sgm_amyg_hipp \
    "$fs_aseg_f" \
    "${tmp_dir}/fivett.nii.gz"
mrconvert -force \
    "${tmp_dir}/fivett.nii.gz" \
    -strides "$strides_spec_f" \
    "$fs_fivett_f"
# Create a strict (minimize partial voluming) CSF mask.
csf_mask_f="${seg_dir}/strict_dwinormalise_csf_mask.nii.gz"
mrconvert \
    "$fs_fivett_f" \
    -coord 3 3 -axes 0,1,2 \
    -strides "$strides_spec_f" \
    - |
    mrgrid \
        - \
        regrid \
        -template "$crop_mask_f" -interp linear \
        - |
    mrthreshold \
        - \
        -abs 0.999 \
        - |
    maskfilter -force -nthreads $N_PROCS \
        - \
        erode -npass 1 \
        "$csf_mask_f"

norm_dwi_f="${diffusion_dir}/dwi_norm.nii.gz"
dwinormalise individual -force -nthreads $N_PROCS \
    "$dwi_mif_f" \
    "$csf_mask_f" \
    -intensity 1000 -percentile "$DWINORMALISE_CSF_PERCENTILE" \
    "$norm_dwi_f"
mrcalc -nthreads $N_PROCS \
    "$norm_dwi_f" \
    "$crop_mask_f" \
    -mult \
    - |
    mrcalc -force \
        - "$norm_dwi_f"
# Export diffusion gradients both in fsl and mrtrix formats.
grad_mrtrix_f="${diffusion_dir}/ras_grad_mrtrix.b"
mrinfo -force \
    "$dwi_mif_f" -bvalue_scaling false \
    -export_grad_mrtrix "$grad_mrtrix_f"
mrinfo -force \
    "$dwi_mif_f" -bvalue_scaling false \
    -export_grad_fsl "${diffusion_dir}/bvecs" "${diffusion_dir}/bvals"
# Recreate the mif dwi with normalised dwi intensities.
rm -fv "$dwi_mif_f"
mrconvert -force -nthreads $N_PROCS \
    "$norm_dwi_f" \
    -grad "$grad_mrtrix_f" \
    -strides "$strides_spec_f" \
    "$dwi_mif_f"

# Estimate response functions with multi-shell multi-tissue constrained deconv.
odf_dir="${output_root_dir}/odf"
mkdir --parents "$odf_dir"
dwi2response msmt_5tt -info -force -nthreads $N_PROCS \
    -wm_algo tournier \
    -mask "$crop_mask_f" \
    "$dwi_mif_f" \
    "$fs_fivett_f" \
    "${odf_dir}/wm_response.txt" \
    "${odf_dir}/gm_response.txt" \
    "${odf_dir}/csf_response.txt"
# Estimate the odfs for each tissue type.
non_norm_odf_dir="${odf_dir}/non_norm"
mkdir --parents "$non_norm_odf_dir"
dwi2fod msmt_csd -force -info -nthreads $N_PROCS \
    "$dwi_mif_f" \
    -mask "$crop_mask_f" \
    "${odf_dir}/wm_response.txt" "${non_norm_odf_dir}/wm_msmt_csd_odf.nii.gz" \
    "${odf_dir}/gm_response.txt" "${non_norm_odf_dir}/gm_msmt_csd_odf.nii.gz" \
    "${odf_dir}/csf_response.txt" "${non_norm_odf_dir}/csf_msmt_csd_odf.nii.gz" \
    -niter 50 -lmax 8,0,0 # gm and csf response functions always collapse into order 0

# Normalise the odfs across the volume.
# The normalization can be sensitive to non-brain tissue, so create a strict brain
# mask based on the hcp freesurfer brainmask.
strict_brain_mask_f="${seg_dir}/strict_mtnormalise_brain_mask.nii.gz"
mrgrid \
    "$fs_brain_mask_f" \
    regrid \
    -template "${non_norm_odf_dir}/wm_msmt_csd_odf.nii.gz" -interp linear \
    -strides "$strides_spec_f" \
    - |
    mrthreshold \
        - \
        -abs 0.999 \
        - |
    maskfilter -force \
        - \
        erode -npass 1 \
        "$strict_brain_mask_f"

mtnormalise -force -info -nthreads $N_PROCS \
    -mask "$strict_brain_mask_f" \
    -order 3 \
    -reference 0.282095 \
    -niter 15,7 \
    "${non_norm_odf_dir}/wm_msmt_csd_odf.nii.gz" "${odf_dir}/wm_msmt_csd_norm_odf.nii.gz" \
    "${non_norm_odf_dir}/gm_msmt_csd_odf.nii.gz" "${odf_dir}/gm_msmt_csd_norm_odf.nii.gz" \
    "${non_norm_odf_dir}/csf_msmt_csd_odf.nii.gz" "${odf_dir}/csf_msmt_csd_norm_odf.nii.gz"

# Also resample the segmentations to be in diffusion space.
fivett_f="${seg_dir}/fivett_dwi-space_segmentation.nii.gz"
mrgrid -force \
    "$fs_fivett_f" \
    regrid \
    -template "$crop_mask_f" -interp nearest \
    -strides "$strides_spec_f" \
    "$fivett_f"
aseg_f="${seg_dir}/$(basename "$fs_aseg_f" .nii.gz)_dwi-space_segmentation.nii.gz"
mrgrid -force \
    "$fs_aseg_f" \
    regrid \
    -template "$crop_mask_f" -interp nearest \
    -strides "$strides_spec_f" \
    "$aseg_f"

# Copy this script to the subject directory.
code_dir="${output_root_dir}/code"
mkdir --parents "$code_dir"
cp --update -v "$0" "$code_dir"

# Clear the temp directory.
rm -rfv "$tmp_dir"
