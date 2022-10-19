# -*- coding: utf-8 -*-
import collections
from pathlib import Path

import einops
import monai
import nibabel as nib
import numpy as np
import torch

import pitn


class HCPfODFINRDataset(monai.data.Dataset):
    def __init__(
        self,
        subj_ids,
        dwi_root_dir: Path,
        fodf_root_dir: Path,
        lr_dwi_root_dir: Path,
        transform=None,
    ):
        self.subj_ids = list(subj_ids)
        self.dwi_root_dir = Path(dwi_root_dir).resolve()
        self.fodf_root_dir = Path(fodf_root_dir).resolve()
        self.lr_dwi_root_dir = Path(lr_dwi_root_dir).resolve()
        data = list()
        for sid in self.subj_ids:
            hcp_data = self.get_hcp_subj_dict(sid, self.dwi_root_dir)
            fodf_data = self.get_fodf_subj_dict(sid, self.fodf_root_dir)
            lr_data = self.get_lr_hcp_subj_dict(sid, self.lr_dwi_root_dir)
            sid_data = dict(
                subj_id=hcp_data["subj_id"],
                lr_dwi=lr_data["dwi"],
                lr_mask=lr_data["mask"],
                lr_bval=lr_data["bval"],
                lr_bvec=lr_data["bvec"],
                fodf=fodf_data["fodf"],
                mask=fodf_data["mask"],
                fivett=fodf_data["fivett"],
                freesurfer_seg=fodf_data["freesurfer_seg"],
            )
            data.append(sid_data)

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

    @staticmethod
    def get_lr_hcp_subj_dict(subj_id, root_dir):
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
        )

        return data

    @staticmethod
    def get_fodf_subj_dict(subj_id, root_dir):
        sid = str(subj_id)
        d = (
            pitn.utils.system.get_file_glob_unique(Path(root_dir).resolve(), f"*{sid}*")
            / "T1w"
        )
        data = dict(
            subj_id=sid,
            fodf=pitn.utils.system.get_file_glob_unique(
                d, "postproc_*msmt*fod*.nii.gz"
            ),
            mask=d / "postproc_nodif_brain_mask.nii.gz",
            fivett=pitn.utils.system.get_file_glob_unique(d, "postproc*5tt*.nii.gz"),
            freesurfer_seg=pitn.utils.system.get_file_glob_unique(
                d, "postproc*aparc*.nii.gz"
            ),
        )

        return data
