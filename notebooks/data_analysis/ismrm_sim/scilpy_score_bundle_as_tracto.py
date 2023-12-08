#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ipdb.set_trace()
import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import ipdb


def scilpy_score_tractogram_cmd(
    in_tractogram,
    gt_config,
    out_dir,
    gt_dir,
    python_cmd,
    scilpy_script,
    reference=None,
    json_prefix=None,
):
    cmd = f"""
    {str(python_cmd)} {str(scilpy_script)} -v -f \
        {str(in_tractogram)} \
        {str(gt_config)} \
        {str(out_dir)} \
        --gt_dir {str(gt_dir)} \
        --remove_invalid --unique --indent 1"""

    cmd = shlex.split(textwrap.dedent(cmd))
    if reference is not None:
        cmd = cmd + shlex.split(f"--reference {str(reference)}")
    if json_prefix is not None:
        cmd = cmd + shlex.split(f"--json_prefix {str(json_prefix)}")

    return cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score group of bundles as tractograms, one bundle at a time"
    )
    parser.add_argument(
        "-b",
        "--bundles_of_interest",
        type=Path,
        required=True,
        help=".txt file with bundles of interest",
    )
    parser.add_argument("--gt_config", required=True, type=Path)
    parser.add_argument("--gt_dir", required=True, type=Path)
    parser.add_argument("-t", "--test_bundles_dir", required=True, type=Path)
    parser.add_argument(
        "-o",
        "--out_dir",
        required=True,
        help="Output directory for the resulting segmented bundles.",
        type=Path,
    )
    parser.add_argument(
        "--reference",
        "-r",
        required=True,
        help="Reference anatomy for tck/vtk",
        type=Path,
    )
    parser.add_argument(
        "--python_arg",
        help="Path of python executable or python run command",
        default=str(shutil.which("python")),
        type=str,
    )
    parser.add_argument(
        "--scilpy_script",
        help="Path to `scilpy_score_tractogram.py` script",
        default=Path(".") / "scilpy_score_tractogram.py",
        type=Path,
    )
    parser.add_argument(
        "--json_prefix",
        help="Prefix of the two output json files. "
        "Ex: 'study_x_'.Files will be saved inside out_dir. "
        "Suffixes will be 'processing_stats.json' and 'results.json'.",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    bundle_names = list()
    with open(args.bundles_of_interest, "rt") as f:
        for line in f:
            bundle_names.append(str(line).strip())
    with open(args.gt_config, "rt") as f:
        config = json.load(f)
    streamline_fs = list(args.test_bundles_dir.glob("*.tck"))
    streamline_fs = streamline_fs + list(
        (args.test_bundles_dir / "sub_bundles").glob("*.tck")
    )

    # Run the score tractogram script one bundle at a time.
    for bundle in bundle_names:
        bundle_out_dir = Path(args.out_dir).resolve() / str(bundle)
        bundle_out_dir.mkdir(exist_ok=True, parents=True)

        config_k = None
        for k in config.keys():
            if bundle.casefold() in k.casefold():
                config_k = k
                break
        assert config_k is not None
        stream_f = None
        for f in streamline_fs:
            fname = Path(f).name
            if bundle.casefold() in fname.casefold():
                stream_f = f
                break
        assert stream_f is not None

        if args.json_prefix is not None:
            json_prefix = f"{args.json_prefix}bundle-{bundle}_"
        else:
            json_prefix = f"bundle-{bundle}_"

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            bundle_config = {k: config[config_k]}
            tmp_config_f = tmp_dir / "tmp_gt_config.json"
            with open(tmp_config_f, "wt") as f:
                json.dump(bundle_config, f, indent=1)
            # breakpoint()
            score_cmd = scilpy_score_tractogram_cmd(
                in_tractogram=Path(stream_f),
                gt_config=tmp_config_f,
                out_dir=bundle_out_dir,
                gt_dir=Path(args.gt_dir).resolve(),
                python_cmd=args.python_arg,
                scilpy_script=args.scilpy_script,
                reference=args.reference,
                json_prefix=json_prefix,
            )

            subprocess.run(score_cmd, check=True, stderr=subprocess.STDOUT)
