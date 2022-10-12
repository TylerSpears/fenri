# -*- coding: utf-8 -*-
import collections
from pathlib import Path

import einops
import monai
import nibabel as nib
import numpy as np
import torch

import pitn


class HCPDataset(monai.data.Dataset):
    def __init__(self, subj_ids, root_dir, transform=None):
        self.subj_ids = list(subj_ids)
        self.root_dir = Path(root_dir).resolve()
        data = [self.get_hcp_subj_dict(sid, self.root_dir) for sid in self.subj_ids]
        super().__init__(data, transform=transform)

    @staticmethod
    def get_hcp_subj_dict(subj_id, root_dir):
        sid = str(subj_id)
        d = (
            pitn.utils.system.get_file_glob_unique(Path(root_dir).resolve(), f"*{sid}*")
            / "T1w"
        )
        diff_d = d / "Diffusion"
        data = dict(
            subj_id=sid,
            dwi=diff_d / "data.nii.gz",
            mask=diff_d / "nodif_brain_mask.nii.gz",
            bval=diff_d / "bvals",
            bvec=diff_d / "bvecs",
            t1w=d / "T1w_acpc_dc_restore_brain.nii.gz",
            t2w=d / "T2w_acpc_dc_restore_brain.nii.gz",
            anat_mask=d / "brainmask_fs.nii.gz",
        )

        return data
