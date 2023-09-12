#!/usr/bin/bash

set -eou pipefail

N_PROCS=${N_PROCS-"$(nproc)"}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS
export OMP_NUM_THREADS=$N_PROCS

ANTSPATH=/opt/ants/ants-2.4.3/bin

moving_im="$1"
moving_reg_mask="$2"
moving_basename="$3"
target_im="$4"
target_reg_mask="$5"
target_basename="$6"
output_dir="."

moving_reg_target_f="${moving_basename}_reg_${target_basename}.nii.gz"
target_reg_moving_f="${target_basename}_reg_${moving_basename}.nii.gz"
transform_prefix="${moving_basename}_reg_${target_basename}_"


# 1. Rigid
transform_type="01_rigid"
"${ANTSPATH}/antsRegistration" --verbose 1 \
    --dimensionality 3 \
    --collapse-output-transforms 1 \
    --output [ "$output_dir/${transform_type}_${transform_prefix}","$output_dir/${transform_type}_${moving_reg_target_f}","$output_dir/${transform_type}_${target_reg_moving_f}" ] \
    --interpolation BSpline[ 3 ] \
    --winsorize-image-intensities [ 0.005,0.995 ] \
    --initial-moving-transform [ "$target_im","$moving_im",1 ] \
    --transform Rigid[ 0.1 ] \
    --metric MI[ "$target_im","$moving_im",1,96,Regular,0.75 ] \
    --convergence [ 1000x600x300x200,1e-6,50 ] \
    --shrink-factors 8x4x2x1 \
    --smoothing-sigmas 3x2x1x0vox \
    --use-histogram-matching \
    --masks [ "$target_reg_mask","$moving_reg_mask" ]
rigid_transform_f="$output_dir/${transform_type}_${transform_prefix}0GenericAffine.mat"

# 2. Similarity
transform_type="02_similarity"
"${ANTSPATH}/antsRegistration" --verbose 1 \
    --dimensionality 3 \
    --collapse-output-transforms 1 \
    --output [ "$output_dir/${transform_type}_${transform_prefix}","$output_dir/${transform_type}_${moving_reg_target_f}","$output_dir/${transform_type}_${target_reg_moving_f}" ] \
    --interpolation BSpline[ 3 ] \
    --winsorize-image-intensities [ 0.005,0.995 ] \
    --initial-moving-transform "$rigid_transform_f" \
    --transform Similarity[ 0.005 ] \
    --metric MI[ "$target_im","$moving_im",1,64,Regular,0.75 ] \
    --convergence [ 1000x600x300,1e-6,40 ] \
    --shrink-factors 4x2x1 \
    --smoothing-sigmas 2x1x0vox \
    --use-histogram-matching \
    --masks [ "$target_reg_mask","$moving_reg_mask" ]
similarity_transform_f="$output_dir/${transform_type}_${transform_prefix}0GenericAffine.mat"

# 3. Affine
transform_type="03_affine"
"${ANTSPATH}/antsRegistration" --verbose 1 \
    --dimensionality 3 \
    --collapse-output-transforms 1 \
    --output [ "$output_dir/${transform_type}_${transform_prefix}","$output_dir/${transform_type}_${moving_reg_target_f}","$output_dir/${transform_type}_${target_reg_moving_f}" ] \
    --interpolation BSpline[ 3 ] \
    --winsorize-image-intensities [ 0.005,0.995 ] \
    --initial-moving-transform "$similarity_transform_f" \
    --transform Affine[ 0.001 ] \
    --metric MI[ "$target_im","$moving_im",1,64,Regular,0.75 ] \
    --convergence [ 600x300,1e-6,30 ] \
    --shrink-factors 2x1 \
    --use-histogram-matching \
    --smoothing-sigmas 1x0vox \
    --masks [ "$target_reg_mask","$moving_reg_mask" ]

# # Correct overshoot/undershoot from registration interpolation.
# im_min=$(mrstats "$moving_im" -output min)
# im_max=$(mrstats "$moving_im" -output max)
# mrcalc -force \
#     "$t2_denoised_reg_t1" $t2_min -max $t2_max -min "$t2_denoised_reg_t1"
