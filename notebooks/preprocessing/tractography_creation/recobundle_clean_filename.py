# -*- coding: utf-8 -*-
from pathlib import Path

atlas_bundles_dir = Path("/data/srv/data/pitn/atlas/Atlas_30_Bundles/bundles/")
curr_recobundles_output_dir = Path(".")
track_names = [p.name.replace(".trk", "") for p in atlas_bundles_dir.glob("*.trk")]

# Grab current directory .trk and/or .npy files.
all_trks = list(curr_recobundles_output_dir.glob("*.trk"))
all_npy = list(curr_recobundles_output_dir.glob("*.npy"))

for t in track_names:
    track_trk = None
    for trk in all_trks:
        if t in trk.name and (t + ".trk") != trk.name:
            track_trk = trk
            break
    track_npy = None
    for npy in all_npy:
        if t in npy.name and (t + ".npy") != npy.name:
            track_npy = npy
            break
    if track_trk is not None:
        track_trk.rename(t + ".trk")
    if track_npy is not None:
        track_npy.rename(t + ".npy")
