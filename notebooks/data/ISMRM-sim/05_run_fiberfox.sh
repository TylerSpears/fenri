#!/usr/bin/bash

set -eou pipefail

N_PROCS=${N_PROCS-"$(nproc)"}
export OMP_NUM_THREADS=$N_PROCS
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS

input_tck="$1"
# Template dwi file must be in the .nrrd format, with the gradient directions embedded
# into the .nrrd file.
template_dwi_f="$2"
# Many fiberfox input files must follow a convention of 
# "{prefix}.ffp[_{volume_type}.nii.gz]", which is how fiberfox discovers these input
# volumes without being explicitly told in the inputs.
# The parameter file must be of the form "{prefix}.ffp" (though we'll strip the .ffp
# in this script because fiberfox will add it in later).
param_f="$3"
# Only one CSF partial volume image is needed, as the white matter pvs will be determined
# by the tracks, and the GM compartment will just fill up the remaining volume.
# The pv volume must be of the form "{prefix}.ffp_VOLUME{compartment_number}.nii.gz",
# where the CSF compartment number is 4.
pv_vol_f="$4"
# The tissue mask must be of the form "{prefix}.ffp_MASK.nii.gz".
mask_vol_f="$5"
output_prefix="$6"
output_dir="$7"

# Output files will typically be of the form "{prefix}[image_type].{file_type}".

mkdir --parents "$output_dir"
tmp_input_dir="${output_dir}/tmp"
mkdir --parents "$tmp_input_dir"

# Make sure the input parameters, partial volume image, and tissue mask are all in the
# same directory.
cp --archive --update "$param_f" "$pv_vol_f" "$mask_vol_f" "$tmp_input_dir"
tmp_param_f="${tmp_input_dir}/$(basename "$param_f")"

MitkFiberfox.sh \
        -i "$input_tck" \
        --parameters "$tmp_param_f" \
        --template "$template_dwi_f" \
        -o "${output_dir}/${output_prefix}" \
        --verbose \
        --dont_apply_direction_matrix \
        --fix_seed
