# -*- coding: utf-8 -*-
import functools
import inspect
import shlex
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from . import convert_seq_for_params, is_sequence

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

    return shlex.join(cmd)


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
