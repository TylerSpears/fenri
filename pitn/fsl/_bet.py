# -*- coding: utf-8 -*-
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Union

from . import FSL_OUTPUT_TYPE_SUFFIX_MAP


def bet_cmd(
    in_file: Path,
    out_file_basename: str,
    mask: bool = False,
    skip_brain_output: bool = False,
    robust_iters: bool = False,
    eye_cleanup: bool = False,
    verbose: bool = True,
    fsl_output_type: str = "NIFTI_GZ",
    fractional_intensity_threshold: Optional[float] = None,
    vertical_grad_in_f: float = 0.0,
) -> str:

    cmd = list()
    cmd.append(f"FSLOUTPUTTYPE={fsl_output_type.upper()}")
    cmd.append("bet")

    cmd.append(str(Path(in_file)))
    cmd.append(str(out_file_basename))
    if mask:
        cmd.append("-m")
    if skip_brain_output:
        cmd.append("-n")
    if fractional_intensity_threshold is not None:
        cmd.append("-f")
        cmd.append(str(fractional_intensity_threshold).format("%g"))
    if vertical_grad_in_f != 0:
        cmd.append("-g")
        cmd.append(str(vertical_grad_in_f).format("%g"))

    if robust_iters:
        cmd.append("-R")
    if eye_cleanup:
        cmd.append("-S")

    if verbose:
        cmd.append("-v")

    cmd = shlex.join(cmd)
    return cmd


def _bet_output_files(
    out_file_basename: str,
    mask: bool = False,
    skip_brain_output: bool = False,
    fsl_output_type: str = "NIFTI_GZ",
    **kwargs,
) -> Dict[str, Union[str, List[str]]]:

    base = str(out_file_basename)
    out_files = dict()
    suffix = FSL_OUTPUT_TYPE_SUFFIX_MAP[fsl_output_type.upper()]

    if not skip_brain_output:
        out_files["brain"] = "".join([base, suffix])

    if mask:
        out_files["mask"] = "".join([base, "_mask", suffix])

    return out_files
