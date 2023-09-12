#!/usr/bin/bash

set -eou pipefail


SCRIPT_DIR=$(dirname "$(realpath "$0")")
in_nu_noneck="$1"
t1w_template_im="$2"

tmp_mask="tmp_mask.nii.gz"
"${SCRIPT_DIR}/_clean_nu_noneck_mask.py" "$in_nu_noneck" "$tmp_mask"

mask_f="t1w_head_mask.nii.gz"

mrgrid "$tmp_mask" regrid -template "$t1w_template_im" -interp nearest "$mask_f" -strides "$t1w_template_im"
rm -f "$tmp_mask"

t1w_head_f="t1w_head.nii.gz"
mrcalc "$t1w_template_im" "$mask_f" -mult "$t1w_head_f"
