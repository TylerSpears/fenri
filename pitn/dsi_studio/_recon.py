# -*- coding: utf-8 -*-
import inspect
import shlex
from pathlib import Path
from typing import Optional, Sequence, Union

from pitn.utils.cli_parse import (
    add_equals_cmd_args,
    convert_seq_for_params,
    is_sequence,
)

RECON_METHOD_MAP = {"DSI": 0, "DTI": 1, "GQI": 4, "QSDR": 7}


def recon_cmd(
    source: Path,
    method: str = "GQI",
    param0: float = 1.25,
    align_acpc: bool = True,
    check_btable: bool = True,
    other_output: Union[str, Sequence[str]] = (
        "fa",
        "ad",
        "rd",
        "md",
        "nqa",
        "iso",
        "rdi",
        "nrdi",
    ),
    record_odf: bool = False,
    qsdr_reso: Optional[float] = None,
    mask: Optional[Path] = None,
    rev_pe: Optional[Path] = None,
    rotate_to: Optional[Path] = None,
    align_to: Optional[Path] = None,
    other_image: Optional[Path] = None,
    save_src: Optional[Path] = None,
    cmd: Optional[str] = None,
    thread_count: Optional[int] = None,
    dti_no_high_b: Optional[bool] = None,
    r2_weighted: bool = False,
) -> str:
    """
    Documentation at <https://dsi-studio.labsolver.org/doc/cli_t2.html>
    # Core Functions
    Parameters 	Default 	Description
    source 	  	specify the .src.gz file for reconstruction.
    method 	4 	specify the reconstruction methods.
    0:DSI, 1:DTI, 4:GQI 7:QSDR.
    param0 	1.25 (in-vivo) or 0.6 (ex-vivo) 	the diffusion sampling length ratio for GQI and QSDR reconstruction.
    align_acpc 	1 	rotate image volume to align ap-pc.
    check_btable 	1 	specify whether the b-table orientation will be checked and automatically flipped
    other_output 	fa,ad,rd,md,nqa,iso,rdi,nrdi 	specify what diffusion metrics to calculate. use all to get all of possible metrics
    record_odf 	0 	specify whether to output the ODF in the fib file (used in connectometry analysis).

    # QSDR Parameters
    Default 	Description
    qsdr_reso 	dwi resolution 	specify output resolution for QSDR reconstruction
    template 	0 	specify the template for QSDR reconstruction:
    0:ICBM152
    1:CIVM_mouse
    2:INDI_rhesus
    3:Neonate
    4:Pitt_Marmoset
    5:WHS_SD_rat

    # Optional Functions
    Parameters 	Description
    mask 	specify the mask file (.nii.gz)
    rev_pe 	assign the NIFTI or SRC file of the reversed-phase encoding images for TOPUP/EDDY
    rotate_to 	specify a T1W or T2W for DSI Studio to rotate DWI to its space. (no scaling or shearing)
    align_to 	specify a T1W or T2W for DSI Studio to use affine transform to its space. (including scaling or shearing)
    other_image 	assign other image volumes (e.g., T1W, T2W image) to be wrapped with QSDR. -other_image=:,:
    save_src 	save preprocessed images to a new SRC file
    cmd 	specify any of the following commands for preprocessing. Use “+” to combine commands, and use “=” to assign value/parameters

    [Step T2][File][Save 4D NIFTI]
    [Step T2][File][Save Src File]
    [Step T2][Edit][Image flip x]
    [Step T2][Edit][Image flip y]
    [Step T2][Edit][Image flip z]
    [Step T2][Edit][Image swap xy]
    [Step T2][Edit][Image swap yz]
    [Step T2][Edit][Image swap xz]
    [Step T2][Edit][Crop Background]
    [Step T2][Edit][Smooth Signals]
    [Step T2][Edit][Align APPC]
    [Step T2][Edit][Change b-table:flip bx]
    [Step T2][Edit][Change b-table:flip by]
    [Step T2][Edit][Change b-table:flip bz]
    [Step T2][Edit][Overwrite Voxel Size]=1.0
    [Step T2][Edit][Resample]=1.0
    [Step T2][Edit][Overwrite Voxel Size]
    [Step T2][B-table][flip bx]
    [Step T2][B-table][flip by
    [Step T2][Corrections][TOPUP EDDY]
    [Step T2][Corrections][EDDY]
    [Step T2][B-table][flip bz]
    [Step T2a][Open]     # specify mask
    [Step T2a][Smoothing]
    [Step T2a][Defragment]
    [Step T2a][Dilation]
    [Step T2a][Erosion]
    [Step T2a][Negate]
    [Step T2a][Remove Background]
    [Step T2a][Threshold]=100

    # Accessory Functions
    Parameters 	Default 	Description
    thread_count 	system thread 	number of multi-thread used to conduct reconstruction
    dti_no_high_b 	1 for human data, 0 for animal data 	specify whether the construction of DTI should ignore high b-value (b>1500)
    r2_weighted 	0 	specify whether GQI and QSDR uses r2-weighted to calculate SDF
    """

    args = locals()
    func_params_list = list(inspect.signature(recon_cmd).parameters.keys())
    call_args = {k: args[k] for k in func_params_list}
    shell_cmd = list()
    shell_cmd.append("dsi_studio")
    shell_cmd.append("--action=rec")

    for k, v in call_args.items():
        if k in {"method"}:
            v = RECON_METHOD_MAP[v.upper().strip()]
        # All bools are converted to int codes.
        if isinstance(v, bool):
            v = int(v)

        # None options won't be set at all.
        if v is None:
            continue

        if k in {"other_output"}:
            if is_sequence(v):
                v = convert_seq_for_params(v, lambda s: s.lower().strip())
            else:
                v = str(v).lower().strip()

        # If non-string Sequence, convert to a comma-separated string.
        if is_sequence(v):
            cval = convert_seq_for_params(v)
        # Everything else should just be a string.
        else:
            cval = str(v)

        shell_cmd.append(f"--{k}")
        shell_cmd.append(cval)

    shell_cmd = shlex.join(shell_cmd)
    eq_sign_cmd = add_equals_cmd_args(shell_cmd)
    return eq_sign_cmd
