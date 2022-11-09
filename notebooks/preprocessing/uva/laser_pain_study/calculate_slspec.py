#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path

import numpy as np


def main():
    scan_info_f = Path(sys.argv[1]).resolve()
    slspec_f = Path(sys.argv[2])
    # scan_info_f = Path(
    #     "/mnt/storage/data/pitn/uva/liu_laser_pain_study/dry-run/sub-001/ses-002/DRY-RUN_002_20211209081012_3_dMRI_SMS_98-directions_AP.json"
    # )
    # slspec_f = Path("/tmp/slspec.txt")
    with open(scan_info_f, "r") as f:
        scan_info = json.load(f)
    scan_info = dict(scan_info)
    try:
        sl_timing = np.asarray(scan_info["SliceTiming"])
    except KeyError:
        raise RuntimeError(
            f"ERROR: file {scan_info_f} does not have slice timing information!"
        )

    groups = list()
    for time_i in np.unique(sl_timing):
        groups.append(np.where(sl_timing == time_i)[0].astype(int).flatten().tolist())

    with open(slspec_f, "w") as f:
        for g in groups:
            f.write(" ".join(map(str, g)) + "\n")


if __name__ == "__main__":
    main()
