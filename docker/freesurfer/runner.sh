#!/bin/bash
set -e

subjs=(
    "OAS30375_MR_d5792"
    "OAS30558_MR_d2148"
    "OAS30643_MR_d0280"
    "OAS30685_MR_d0032"
    "OAS30762_MR_d0043"
    "OAS30770_MR_d1201"
    "OAS30944_MR_d0089"
    "OAS31018_MR_d0041"
    "OAS31157_MR_d4924"
)

JOBS=2

parallel --halt-on-error now,fail=1 --ungroup --jobs $JOBS ./process_sub.sh ::: "${subjs[@]}"
