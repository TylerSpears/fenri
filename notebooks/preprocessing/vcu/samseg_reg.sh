#!/usr/bin/bash

set -eou pipefail

# Full registration command with Lanczos-windowed sinc interpolation.
/opt/ants/ants-2.4.3/bin/antsRegistration --verbose 1 \
    --dimensionality 3 \
    --float 0 \
    --collapse-output-transforms 1 \
    --output "[ T2w2T1w_,T2w2T1w_Warped.nii.gz,T2w2T1w_InverseWarped.nii.gz ]" \
    --interpolation LanczosWindowedSinc \
    --use-histogram-matching 0 \
    --winsorize-image-intensities "[ 0.005,0.995 ]" \
    --initial-moving-transform "[ ../T1w.nii.gz,T2w.nii.gz,1 ]" \
    --transform "Rigid[ 0.1 ]" \
    --metric "MI[ ../T1w.nii.gz,T2w.nii.gz,1,32,Regular,0.25 ]" \
    --convergence "[ 1000x500x250x100,1e-6,10 ]" \
    --shrink-factors 8x4x2x1 \
    --smoothing-sigmas 3x2x1x0vox

# Simplified registration with (forced) tri-linear interpolation.
# /opt/ants/ants-2.4.3/bin/antsRegistrationSyN.sh -d 3 -f T1w.nii.gz -m flair.nii.gz -o flair2T1w_ -n 19 -t r

# Apply ants transformation
# /opt/ants/ants-2.4.3/bin/antsApplyTransforms \
#     -d 3 -i ../cluster_mask_zfstat1.nii.gz \
#     -r ../../../../anat/T2w.nii.gz \
#     -o cluster_mask_zfstat1_SyNreg-T2w.nii.gz \
#     -t func2T2w_s-transform_1Warp.nii.gz \
#     -t func2T2w_s-transform_0GenericAffine.mat -n Linear
