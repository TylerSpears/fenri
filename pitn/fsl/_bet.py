# -*- coding: utf-8 -*-
import shlex
from pathlib import Path
from typing import List, Optional, Sequence, Union

from pitn.utils.cli_parse import add_equals_cmd_args


def bet_cmd(
    in_file: Path,
    out_file: Path,
    mask: bool = False,
    skip_brain_output: bool = False,
    robust_iters: bool = False,
    eye_cleanup: bool = False,
    verbose: bool = True,
) -> str:

    cmd = list()
    cmd.append("bet")

    cmd.append(str(Path(in_file)))
    cmd.append(str(Path(out_file)))
    if mask:
        cmd.append("--mask")
    if skip_brain_output:
        cmd.append("--nooutput")
    if robust_iters:
        cmd.append("-R")
    if eye_cleanup:
        cmd.append("-S")

    if verbose:
        cmd.append("--verbose")

    cmd = shlex.join(cmd)
    cmd = add_equals_cmd_args(cmd)
    return cmd
