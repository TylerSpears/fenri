#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

import dipy
import dipy.io
import dipy.io.streamline

if __name__ == "__main__":
    assert len(sys.argv) == 4
    args = sys.argv[1:]
    input_trk = Path(str(args[0])).resolve()
    ref = str(args[1])
    if ref.casefold() != "same":
        ref = Path(ref).resolve()
    else:
        ref = ref.casefold()
    output_tck = Path(str(args[2]))

    assert input_trk.exists()
    if isinstance(ref, Path):
        assert ref.exists()

    input_trk = str(input_trk)
    ref = str(ref)
    output_tck = str(output_tck)
    trk = dipy.io.streamline.load_trk(input_trk, reference=ref)
    trk.to_rasmm()

    dipy.io.streamline.save_tck(trk, filename=output_tck)
