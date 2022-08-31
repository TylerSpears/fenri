# -*- coding: utf-8 -*-
import functools
import inspect
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pitn.utils.cli_parse import (
    add_equals_cmd_args,
    append_cmd_stdout_stderr_to_file,
    convert_seq_for_params,
    is_sequence,
)

from . import FSL_OUTPUT_TYPE_SUFFIX_MAP

TOPUP_CONFIG_MAP = {"minmet": {"LM": 0, "SCG": 1}, "bool": {True: 1, False: 0}}
TOPUP_ACQ_DIR_MAP = {"i": (1, 0, 0), "j": (0, 1, 0), "k": (0, 0, 1)}


def topup_cmd(
    imain: Path,
    datain: Path,
    out: Optional[str] = None,
    iout: Optional[Path] = None,
    fout: Optional[Path] = None,
    warpres: Union[int, Sequence[int]] = 10,
    subsamp: Union[int, Sequence[int]] = 1,
    fwhm: Union[int, Sequence[int]] = 8,
    miter: Union[int, Sequence[int]] = 5,
    lambd: Optional[Union[float, Sequence[float]]] = None,
    ssqlambda: Union[bool, Sequence[bool]] = True,
    estmov: Union[bool, Sequence[bool]] = True,
    minmet: Union[str, Sequence[str]] = "LM",
    regmod: str = "bending_energy",
    splineorder: int = 3,
    numprec: str = "double",
    interp: str = "spline",
    scale: Union[bool, Sequence[bool]] = False,
    regrid: Union[bool, Sequence[bool]] = True,
    verbose: bool = True,
    logout: Optional[Path] = None,
    fsl_output_type: str = "NIFTI_GZ",
) -> str:
    """
    # name of 4D file with images
    imain
    # name of text file with PE directions/times
    datain
    #   base-name of output files (spline coefficients (Hz) and movement parameters)
    out
    #   name of image file with field (Hz)
    fout
    #   name of 4D image file with unwarped images
    iout
    # (approximate) resolution (in mm) of warp basis for the different sub-sampling levels, default 10
    warpres=20,16,14,12,10,6,4,4,4
    # sub-sampling scheme, default 1
    subsamp=2,2,2,2,2,1,1,1,1
    #   FWHM (in mm) of gaussian smoothing kernel, default 8
    fwhm=8,6,4,3,3,2,1,0,0
    #   Max # of non-linear iterations, default 5
    miter=5,5,5,5,5,10,10,20,20
    # Weight of regularisation, default depending on ssqlambda and regmod switches. See user documetation.
    lambda
    # If set (=1), lambda is weighted by current ssq, default 1
    ssqlambda=1
    # Estimate movements if set, default 1 (true)
    estmov
    # Minimisation method 0=Levenberg-Marquardt, 1=Scaled Conjugate Gradient, default LM
    Enumerated as strings {'LM', 'SCG'} in the python wrapper.
    minmet
    # Model for regularisation of warp-field [membrane_energy bending_energy], default bending_energy
    regmod=bending_energy
    # Order of spline, 2->Qadratic spline, 3->Cubic spline. Default=3
    splineorder=3
    # Precision for representing Hessian, double or float. Default double
    numprec=double
    # Image interpolation model, linear or spline. Default spline
    interp=spline
    #   If set (=1), the images are individually scaled to a common mean, default 0 (false)
    scale=1
    #   If set (=1), the calculations are done in a different grid, default 1 (true)
    regrid=1
    """
    # Most arg types need to be converted appropriately, so it's easier to loop through
    # and check arg values.
    args = locals()
    func_params_list = list(inspect.signature(topup_cmd).parameters.keys())
    call_args = {k: args[k] for k in func_params_list}
    # Fix lambd/lambda parameter name, constrained due to python lambda token.
    call_args["lambda"] = call_args.pop("lambd")

    cmd = list()
    # Always avoid ambiguity regarding duplicate basenames. This is the default, but
    # it's good to be explicit.
    cmd.append("FSLMULTIFILEQUIT=TRUE")
    cmd.append(f"FSLOUTPUTTYPE={fsl_output_type.upper().strip()}")
    call_args.pop("fsl_output_type")

    cmd.append("topup")

    for k, v in call_args.items():
        # Convert some args from bools to ints.
        if k in {"ssqlambda", "estmov", "scale", "regrid"}:
            if not is_sequence(v):
                v = TOPUP_CONFIG_MAP["bool"][v]
            else:
                v = [TOPUP_CONFIG_MAP["bool"][v_i] for v_i in v]
        elif k in {"minmet"}:
            v = TOPUP_CONFIG_MAP["minmet"][str(v).upper()]

        # None options won't be set at all.
        if v is None:
            continue
        # Despite the fact that these correspond to an individual file, topup only accepts
        # the base name as a valid option. So, just remove the suffixes.
        # Put this after the None-check, because these parameters are truly optional.
        if k in {"fout", "iout"}:
            v = Path(v)
            v = str(v).replace("".join(v.suffixes), "")
        # If non-string Sequence, convert to a comma-separated string.
        if is_sequence(v):
            # If a sequence of numbers, then
            if any(isinstance(v_i, float) for v_i in v):
                cval = convert_seq_for_params(
                    v,
                    element_format_callable=functools.partial(
                        np.format_float_positional, trim="-"
                    ),
                )
            else:
                cval = convert_seq_for_params(v)
        # Everything else should just be a string.
        else:
            cval = str(v)

        cmd.append(f"--{k}")
        if k != "verbose":
            cmd.append(cval)
    cmd = shlex.join(cmd)
    cmd = add_equals_cmd_args(cmd)
    return cmd


def topup_cmd_explicit_in_out_files(
    imain: Path,
    datain: Path,
    out: Optional[str] = None,
    iout: Optional[Path] = None,
    fout: Optional[Path] = None,
    warpres: Union[int, Sequence[int]] = 10,
    subsamp: Union[int, Sequence[int]] = 1,
    fwhm: Union[int, Sequence[int]] = 8,
    miter: Union[int, Sequence[int]] = 5,
    lambd: Optional[Union[float, Sequence[float]]] = None,
    ssqlambda: Union[bool, Sequence[bool]] = True,
    estmov: Union[bool, Sequence[bool]] = True,
    minmet: Union[str, Sequence[str]] = "LM",
    regmod: str = "bending_energy",
    splineorder: int = 3,
    numprec: str = "double",
    interp: str = "spline",
    scale: Union[bool, Sequence[bool]] = False,
    regrid: Union[bool, Sequence[bool]] = True,
    verbose: bool = True,
    builtin_logout: Optional[Path] = None,
    stdout_log_f: Optional[Union[str, Path]] = None,
    fsl_output_type: str = "NIFTI_GZ",
) -> Tuple[str, List[str], Dict[str, Union[List[str], str]]]:

    args = locals()
    # Take the kwargs of this function and intersect them with the parameters of
    # the associated "cmd" function.
    self_params = list(
        inspect.signature(topup_cmd_explicit_in_out_files).parameters.keys()
    )
    func_params_list = list(inspect.signature(topup_cmd).parameters.keys())
    # Params that are not common between these two cmd functions will be inserted later.
    topup_cmd_kwargs = {k: args[k] for k in set(self_params) & set(func_params_list)}
    topup_cmd_kwargs["logout"] = builtin_logout

    in_files = list()
    out_files = dict()

    in_files.append(str(imain))
    in_files.append(str(datain))

    out_base = str(out)
    im_output_suff = FSL_OUTPUT_TYPE_SUFFIX_MAP[fsl_output_type.upper().strip()]
    out_files["fieldcoef"] = "".join([out_base, "fieldcoef", im_output_suff])

    if estmov:
        out_files["movpar"] = out_base + "movpar.txt"
    if iout is not None:
        if len(Path(iout).suffixes) == 0:
            iout_f = str(iout) + im_output_suff
        else:
            iout_f = str(iout)
        out_files["corrected_im"] = iout_f

    if fout is not None:
        if len(Path(fout).suffixes) == 0:
            fout_f = str(fout) + im_output_suff
        else:
            fout_f = str(fout)
        out_files["field_out"] = fout_f

    if builtin_logout is not None:
        out_files["log"] = str(builtin_logout)
    if stdout_log_f is not None:
        out_files["stdout"] = str(stdout_log_f)

    cmd = topup_cmd(**topup_cmd_kwargs)

    if stdout_log_f is not None:
        cmd = append_cmd_stdout_stderr_to_file(
            cmd, str(stdout_log_f), overwrite_log=True
        )

    return cmd, in_files, out_files


def phase_encoding_dirs2acqparams(
    total_readout_time: float, *pe_dir_symbols: Sequence[str]
) -> np.ndarray:

    acqp = list()
    for pe in pe_dir_symbols:
        pe = str(pe).strip().lower().replace(" ", "")
        negate = -1 if "-" in pe else 1
        pe = pe.replace("-", "")
        dir = negate * np.asarray(TOPUP_ACQ_DIR_MAP[pe])
        row = np.concatenate([dir, np.asarray([total_readout_time])])
        acqp.append(row)
    acqp = np.stack(acqp)
    return acqp


if __name__ == "__main__":
    print(topup_cmd("", ""))
