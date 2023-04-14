#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import dipy
import dipy.io
import dipy.io.streamline

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert tractography `.trk` files to mrtrix `.tck` files with a reference space file."
    )

    parser.add_argument(
        "trk_files",
        nargs="+",
        type=Path,
        help=".trk files to convert (required)",
    )
    parser.add_argument(
        "-r",
        "--reference",
        default="same",
        required=False,
        type=str,
        help="\n".join(
            [
                "String containing the reference space for the input file(s) (optional).",
                "May be the string 'same' to use the input `.trk` file space, or a .trk or .nii.gz file.",
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
                "Output .tck file name(s), one for each input .trk file name(s) (optional).",
                "If not provided, output name will be the same as the input file name with the file extension `.tck`.",
            ],
        ),
    )

    args = parser.parse_args()
    input_trks = args.trk_files
    outputs = args.output
    if outputs is not None:
        if len(input_trks) != len(outputs):
            parser.error(
                f"ERROR: Number of input files {len(input_trks)} "
                + f"must equal provided number of output files {len(outputs)}"
            )
            raise ValueError()
        output_trks = outputs
    else:
        output_trks = [
            inp.parent / (inp.name.replace(".trk", ".tck")) for inp in input_trks
        ]
    if args.reference.casefold() == "same":
        ref = "same"
    else:
        ref_f = Path(args.reference)
        if not ref_f.exists():
            parser.error(f"ERROR: Reference file {str(ref_f)} does not exist.")
        ref = str(ref_f.resolve())

    for in_trk, out_tck in zip(input_trks, output_trks):
        trk_fname = str(in_trk.resolve())
        tck_fname = str(out_tck)
        print(f"Loading {str(in_trk)}", end="...", flush=True)
        trk = dipy.io.streamline.load_trk(trk_fname, reference=ref)
        print("Transforming to RAS mm", end="...", flush=True)
        trk.to_rasmm()
        print(f"Saving {str(in_trk)} -> {str(out_tck)}", end="...", flush=True)

        dipy.io.streamline.save_tck(trk, filename=tck_fname)
        print("Done", flush=True)

    parser.exit(status=0)
