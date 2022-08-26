# -*- coding: utf-8 -*-
import shlex
from pathlib import Path
from typing import Optional, Sequence, Union

from pitn.utils import cli_parse


def convert_to_src_cmd(
    source: Path,
    output: Path,
    bval: Optional[Path] = None,
    bvec: Optional[Path] = None,
    b_table: Optional[Path] = None,
    other_sources: Optional[Union[Path, Sequence[Path]]] = None,
) -> str:
    """
    Full documentation at <https://dsi-studio.labsolver.org/doc/cli_t1.html>

    # Core Functions
    Parameters 	Description
    source 	Specify a directory storing DICOM files or the path to one 4D NIFTI file

    # Optional Functions
    Parameters 	Description
    other_source 	specify other files to be included in the SRC file. Multiple files can be assigned using comma separator,
                    (e.g. --other_source=1.nii.gz,2.nii.gz)
    output 	assign the output src file name (.src.gz) or the output folder
    b_table 	assign the text file to replace b-table
    bval 	specify the location of the FSL bval file*
    bvec 	specify the location of the FSL bvec file*

    *for most cases, DSI Studio can automatically associate bval and bvec with NIFTI automatically.

    # Accessory Functions
    Parameters 	Default 	Description
    recursive 	0 	search all NIFTI or DICOM files under the directory specified in -source
    up_sampling 	0 	upsampling the DWI, 0:no resampling, 1:upsample by 2, 2:downsample by 2, 3:upsample by 4, 4:downsample by 4
    """

    cmd = list()
    cmd.append("dsi_studio")
    cmd.append("--action=src")

    cmd.append("--source")
    cmd.append(str(Path(source)))

    cmd.append("--output")
    cmd.append(str(Path(output)))

    if bval is not None:
        cmd.append("--bval")
        cmd.append(str(Path(bval)))
    if bvec is not None:
        cmd.append("--bvec")
        cmd.append(str(Path(bvec)))
    if b_table is not None:
        cmd.append("--b_table")
        cmd.append(str(Path(b_table)))

    if other_sources is not None:
        # Try if other_sources is a single Path/string.
        try:
            s = str(Path(other_sources))
        # other_sources is likely a list of strings/Paths.
        except TypeError:
            s = [str(Path(src)) for src in other_sources]
            # Quote the files here, as they need to be joined into one token.
            s = [shlex.quote(src) for src in s]
        cmd.append("--other_source")
        cmd.append(s)

    cmd = shlex.join(cmd)
    eq_sign_cmd = cli_parse.add_equals_cmd_args(cmd)
    return eq_sign_cmd
