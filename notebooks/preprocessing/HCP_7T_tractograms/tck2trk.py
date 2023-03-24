#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import dipy
import dipy.io
import dipy.io.streamline

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert tractography `.tck` files to `.trk` files with a reference space file."
    )

    parser.add_argument(
        "tck_files",
        nargs="+",
        type=Path,
        help=".tck files to convert (required)",
    )
    parser.add_argument(
        "-r",
        "--reference",
        required=True,
        type=Path,
        help="\n".join(
            [
                "File containing the reference space for the input file(s) (required).",
                "May be a .trk or .nii.gz file.",
            ],
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="+",
        required=False,
        default=None,
        type=Path,
        help="\n".join(
            [
                "Output .trk file names, one for each input .tck file name(s) (optional).",
                "If not provided, output name will be the same as the input file name with the file extension `.trk`.",
            ],
        ),
    )

    args = parser.parse_args()
    input_tcks = args.tck_files
    outputs = args.output
    if outputs is not None:
        if len(input_tcks) != len(outputs):
            parser.error(
                f"ERROR: Number of input files {len(input_tcks)} "
                + f"must equal provided number of output files {len(outputs)}"
            )
            raise ValueError()
        output_trks = outputs
    else:
        output_trks = [
            inp.parent / (inp.name.replace(".tck", ".trk")) for inp in input_tcks
        ]
    ref = args.reference
    ref = str(ref.resolve())

    for in_tck, out_trk in zip(input_tcks, output_trks):
        tck_fname = str(in_tck.resolve())
        trk_fname = str(out_trk)
        tck = dipy.io.streamline.load_tractogram(tck_fname, reference=ref)
        tck.to_rasmm()
        print(f"Saving {str(in_tck)} -> {str(out_trk)}")

        dipy.io.streamline.save_trk(tck, filename=trk_fname)

    parser.exit(status=0)
