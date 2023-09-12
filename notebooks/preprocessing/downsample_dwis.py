# -*- coding: utf-8 -*-

# %% [markdown]
# # Batch Prediction in Subject Native Resolution for fODF INR
#
# Code by:
#
# Tyler Spears - tas6hh@virginia.edu
#
# Dr. Tom Fletcher

# %% [markdown]
# ## Imports & Setup

# %%
# Imports
# Automatically re-import project-specific modules.
# %load_ext autoreload
# %autoreload 2

# imports
import collections
import copy
import datetime
import functools
import io
import itertools
import math
import os
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import typing
import warnings
from functools import partial
from pathlib import Path
from pprint import pprint as ppr

import dotenv
import einops

# visualization libraries
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import monai

# Data management libraries.
import nibabel as nib
import nibabel.processing

# Computation & ML libraries.
import numpy as np
import pandas as pd
import seaborn as sns
import skimage
import torch
import torch.nn.functional as F
from box import Box
from icecream import ic
from natsort import natsorted

import pitn

# %%
# Update notebook's environment variables with direnv.
# This requires the python-dotenv package, and direnv be installed on the system
# This will not work on Windows.
# NOTE: This is kind of hacky, and not necessarily safe. Be careful...
# Libraries needed on the python side:
# - os
# - subprocess
# - io
# - dotenv

# Form command to be run in direnv's context. This command will print out
# all environment variables defined in the subprocess/sub-shell.
command = "direnv exec {} /usr/bin/env".format(os.getcwd())
# Run command in a new subprocess.
proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, cwd=os.getcwd())
# Store and format the subprocess' output.
proc_out = proc.communicate()[0].strip().decode("utf-8")
# Use python-dotenv to load the environment variables by using the output of
# 'direnv exec ...' as a 'dummy' .env file.
dotenv.load_dotenv(stream=io.StringIO(proc_out), override=True)

# %%
# torch setup
# allow for CUDA usage, if available
#! A GPU for linear interpolation is a bit overkill...
# if torch.cuda.is_available():
if False:
    # Pick only one device for the default, may use multiple GPUs for training later.
    if "CUDA_PYTORCH_DEVICE_IDX" in os.environ.keys():
        dev_idx = int(os.environ["CUDA_PYTORCH_DEVICE_IDX"])
    else:
        dev_idx = 1
    device = torch.device(f"cuda:{dev_idx}")
    print("CUDA Device IDX ", dev_idx)
    torch.cuda.set_device(device)
    print("CUDA Current Device ", torch.cuda.current_device())
    print("CUDA Device properties: ", torch.cuda.get_device_properties(device))
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # See
    # <https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices>
    # for details.

    # Activate cudnn benchmarking to optimize convolution algorithm speed.
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True
        print("CuDNN convolution optimization enabled.")
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True
else:
    device = torch.device("cpu")
# keep device as the cpu
device = torch.device("cpu")
print(device)

# %%
# %% [markdown]
# ## Experiment & Parameters Setup

# %%
p = Box(default_box=True)
# Experiment defaults, can be overridden in a config file.

# General experiment-wide params
###############################################
p.experiment_name = "downsample-dwi_resample-bvec_2.00mm"
p.results_dir = "/data/srv/outputs/pitn/results/runs"
p.tmp_results_dir = "/data/srv/outputs/pitn/results/tmp"
# p.train_val_test_split_file = random.choice(
#     list(Path("./data_splits").glob("HCP*train-val-test_split*.csv"))
# )
###############################################
# kwargs for the sub-selection function to go from full DWI -> low-res DWI.
# See `sub_select_dwi_from_bval` function in `pitn`.
p.bval_sub_sample_fn_kwargs = dict(
    shells_to_remove=[2000],
    within_shell_idx_to_keep={
        0: range(0, 9),
        1000: range(0, 45),
        3000: range(0, 45),
    },
)
# 1.25mm -> 2.0mm
p.baseline_lr_spacing_scale = 1.6
p.scale_prefilter_kwargs = dict(
    sigma_scale_coeff=2.5,
    sigma_truncate=4.0,
)
p.test.subj_ids = list(
    map(
        str,
        [
            106824,
            126628,
            130013,
            134425,
            135225,
            139839,
            144125,
            144933,
            151526,
            153227,
            155231,
            162329,
            163331,
            172433,
            192843,
            193239,
            198451,
            200513,
            213522,
            220721,
            236130,
            245333,
            248339,
            250932,
            255639,
            256540,
            300719,
            314225,
            376247,
            379657,
            453542,
            459453,
            469961,
            480141,
            492754,
            497865,
            500222,
            519647,
            562446,
            566454,
            567961,
            571144,
            580751,
            654754,
            656253,
            677968,
            683256,
            715647,
            727654,
            765056,
            773257,
            872562,
            911849,
            972566,
            991267,
        ],
    )
)

# If a config file exists, override the defaults with those values.
try:
    if "PITN_CONFIG" in os.environ.keys():
        config_fname = Path(os.environ["PITN_CONFIG"])
    else:
        config_fname = pitn.utils.system.get_file_glob_unique(Path("."), r"config.*")
    f_type = config_fname.suffix.casefold()
    if f_type in {".yaml", ".yml"}:
        f_params = Box.from_yaml(filename=config_fname)
    elif f_type == ".json":
        f_params = Box.from_json(filename=config_fname)
    elif f_type == ".toml":
        f_params = Box.from_toml(filename=config_fname)
    else:
        raise RuntimeError()

    p.merge_update(f_params)

except:
    print("WARNING: Config file not loaded")
    pass

# Remove the default_box behavior now that params have been fully read in.
_p = Box(default_box=False)
_p.merge_update(p)
p = _p


# %%
# tvt_split = pd.read_csv(p.train_val_test_split_file)
# p.test.subj_ids = natsorted(tvt_split[tvt_split.split == "test"].subj_id.tolist())

# %%
ic(p.to_dict())

# %% [markdown]
# ## Data Loading

# %%
hcp_full_res_data_dir = Path("/data/srv/data/pitn/hcp")
hcp_full_res_fodf_dir = Path("/data/srv/outputs/pitn/hcp/full-res/fodf")
hcp_low_res_data_dir = Path("/data/srv/outputs/pitn/hcp/downsample/scale-2.00mm/vol")
hcp_low_res_fodf_dir = Path("/data/srv/outputs/pitn/hcp/downsample/scale-2.00mm/fodf")

assert hcp_full_res_data_dir.exists()
assert hcp_full_res_fodf_dir.exists()
assert hcp_low_res_data_dir.exists()
assert hcp_low_res_fodf_dir.exists()

# %%
# Define hte bval/bvec sub-sample scheme according to the parameter dict kwargs.
bval_sub_sample_fn = partial(
    pitn.data.datasets2.sub_select_dwi_from_bval,
    **p.bval_sub_sample_fn_kwargs.to_dict(),
)

# %% [markdown]
# ### Test Dataset

# %%
with warnings.catch_warnings(record=True) as warn_list:

    test_dataset = pitn.data.datasets2.HCPfODFINRDataset(
        subj_ids=p.test.subj_ids,
        dwi_root_dir=hcp_full_res_data_dir,
        fodf_root_dir=hcp_full_res_fodf_dir,
        transform=pitn.data.datasets2.HCPfODFINRDataset.default_pre_sample_tf(
            sample_mask_key="wm_mask",
            bval_sub_sample_fn=bval_sub_sample_fn,
        ),
    )

    test_dataset = pitn.data.datasets2.HCPfODFINRWholeBrainDataset(
        test_dataset,
        transform=pitn.data.datasets2.HCPfODFINRWholeBrainDataset.default_vol_tf(
            baseline_iso_scale_factor_lr_spacing_mm_low_high=p.baseline_lr_spacing_scale,
            scale_prefilter_kwargs=p.scale_prefilter_kwargs,
            keep_metatensor_output=True,
        ),
    )

print("=" * 10)
print("Warnings caught:")
ws = "\n".join(
    [
        warnings.formatwarning(
            w.message, w.category, w.filename, w.lineno, w.file, w.line
        )
        for w in warn_list
    ]
)
ws = "\n".join(filter(lambda s: bool(s.strip()), ws.splitlines()))
print(ws, flush=True)
print("=" * 10)


# %% [markdown]
# ## Testing

# %%
# Utility function for running mrtrix msmt spherical deconvolution on upsampled dwis.
def mrtrix_msmt_sph_deconv(
    dwi_im,
    bval: np.ndarray,
    bvec: np.ndarray,
    brain_mask_im,
    freesurfer_seg_im,
    target_directory: Path,
) -> dict[str, Path]:

    # Copy the function from the preprocessing script that was actually used to fit
    # the ground truth dwis.
    def mrtrix_fit_fodf(
        dwi_f: Path,
        bval_f: Path,
        bvec_f: Path,
        mask_f: Path,
        freesurfer_seg_f: Path,
        target_fodf_f: Path,
        n_threads: int = 4,
    ) -> Path:

        docker_img = "mrtrix3/mrtrix3:3.0.3"

        dwi_f = Path(dwi_f).resolve()
        bval_f = Path(bval_f).resolve()
        bvec_f = Path(bvec_f).resolve()
        freesurfer_seg_f = Path(freesurfer_seg_f).resolve()
        target_dir = Path(target_fodf_f).parent.resolve()

        dwi_mif_f = target_dir / "_tmp.dwi.mif"
        five_tt_parc_f = target_dir / "5tt_parcellation.nii.gz"

        # It's unclear what the default lmax is for dwi2response, and there's a lot of
        # discussion about l_max:
        # <https://mrtrix.readthedocs.io/en/latest/concepts/sh_basis_lmax.html>
        # <https://github.com/MRtrix3/mrtrix3/pull/786> for a discussion about using only
        # zonal harmonics (m=0 for each even l) to estimate the tissue response functions.
        # Looking at other data, it seems that for the actual fod estimation, the
        # white matter gets an l_max of 8, while grey matter and CSF get an l_max of 0.
        # This seems overly restrictive on grey matter, but I'm not the expert here...

        script = rf"""\
        set -eou pipefail
        mrconvert -info -fslgrad \
            {bvec_f} \
            {bval_f} \
            {dwi_f} \
            {dwi_mif_f}
        5ttgen -info freesurfer -nthreads {n_threads} \
            {freesurfer_seg_f} \
            {five_tt_parc_f} \
            -force
        dwi2response msmt_5tt -force \
            -wm_algo tournier \
            -mask {mask_f} \
            {dwi_mif_f} \
            {five_tt_parc_f} \
            {target_dir / "wm_response.txt"} \
            {target_dir / "gm_response.txt"} \
            {target_dir / "csf_response.txt"}
        dwi2fod -info -nthreads {n_threads} \
            -lmax 8,4,0 \
            -niter 100 \
            -mask {mask_f} \
            msmt_csd \
            {dwi_mif_f} \
            {target_dir / "wm_response.txt"} {target_fodf_f} \
            {target_dir / "gm_response.txt"} {target_dir / "gm_msmt_csd_fod.nii.gz"} \
            {target_dir / "csf_response.txt"} {target_dir / "csf_msmt_csd_fod.nii.gz"} \
            -force
        rm {dwi_mif_f}"""

        script = textwrap.dedent(script)
        script = pitn.utils.proc_runner.multiline_script2docker_cmd(script)
        vols = pitn.utils.union_parent_dirs(
            dwi_f, bval_f, bvec_f, freesurfer_seg_f, target_dir
        )
        vols = {v: pitn.utils.proc_runner.get_docker_mount_obj(v) for v in vols}
        docker_config = dict(volumes=vols)

        fodf_run_status = pitn.utils.proc_runner.call_docker_run(
            docker_img, cmd=script, run_config=docker_config
        )

        return target_fodf_f

    # Run everything in a temp directory, then move into the target directory.
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_dir = Path(tmpdirname)

        dwi_f = tmp_dir / "dwi.nii.gz"
        nib.save(dwi_im, dwi_f)
        bval_f = tmp_dir / "bvals"
        np.savetxt(bval_f, bval, fmt="%g")
        bvec_f = tmp_dir / "bvecs"
        if bvec.shape[0] != 3:
            bvec = bvec.T
        np.savetxt(bvec_f, bvec, fmt="%g")

        brain_mask_f = tmp_dir / "nodif_brain_mask.nii.gz"
        nib.save(brain_mask_im, brain_mask_f)

        freesurfer_seg_f = tmp_dir / "aparc.a2009s+aseg.nii.gz"
        nib.save(freesurfer_seg_im, freesurfer_seg_f)

        fodf_f = tmp_dir / "wm_msmt_csd_fod.nii.gz"

        result_wm_fodf = mrtrix_fit_fodf(
            dwi_f,
            bval_f=bval_f,
            bvec_f=bvec_f,
            mask_f=brain_mask_f,
            freesurfer_seg_f=freesurfer_seg_f,
            target_fodf_f=fodf_f,
            n_threads=3,
        )

        fit_fs = dict(
            fivett=pitn.utils.system.get_file_glob_unique(tmp_dir, "*5tt*.nii.gz"),
            wm_response=pitn.utils.system.get_file_glob_unique(
                tmp_dir, "*wm_response*"
            ),
            gm_response=pitn.utils.system.get_file_glob_unique(
                tmp_dir, "*gm_response*"
            ),
            csf_response=pitn.utils.system.get_file_glob_unique(
                tmp_dir, "*csf_response*"
            ),
            wm_odf_coeff=result_wm_fodf,
            gm_odf_coeff=pitn.utils.system.get_file_glob_unique(
                tmp_dir, "*gm*fod*.nii.gz"
            ),
            csf_odf_coeff=pitn.utils.system.get_file_glob_unique(
                tmp_dir, "*csf*fod*.nii.gz"
            ),
        )

        moved_fit_fs = dict()
        for k, f in fit_fs.items():
            moved_f = shutil.move(f, target_directory)
            moved_fit_fs[k] = Path(moved_f)

    return moved_fit_fs


# %%
ts = datetime.datetime.now().replace(microsecond=0).isoformat()
# Break ISO format because many programs don't like having colons ':' in a filename.
# ts = ts.replace(":", "_")
# experiment_name = f"{ts}_{p.experiment_name}"
experiment_name = "trilinear/2023-06-30T17_35_44_downsample-dwi_resample-bvec_2.00mm"
tmp_res_dir = Path(p.results_dir) / experiment_name
# tmp_res_dir = (
#     Path(p.tmp_results_dir) / "2023-06-25T18_28_56_downsample-dwi_fit-odf_scale-2.00mm"
# )
tmp_res_dir.mkdir(parents=True, exist_ok=True)

# %%
model = "trilinear"
model_pred_res_dir = tmp_res_dir / model
model_pred_res_dir.mkdir(parents=True, exist_ok=True)

# Wrap the entire loop in a try...except statement to save out a failure indicator file.
try:
    test_dataloader = monai.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=1,
        persistent_workers=True,
        prefetch_factor=1,
    )

    with torch.no_grad():
        print("Starting inference", flush=True)
        for batch_dict in test_dataloader:
            subj_id = batch_dict["subj_id"]
            if len(subj_id) == 1:
                subj_id = subj_id[0]
            print(f"Starting {subj_id}", flush=True)

            x = batch_dict["lr_dwi"].to(device)
            batch_size = x.shape[0]
            # Much of the processing code below assumes a batch size of 1.
            assert batch_size == 1
            x_mask = batch_dict["lr_brain_mask"].to(torch.bool).to(device)
            x_affine_vox2world = batch_dict["affine_lr_vox2world"].to(device)
            x_vox_size = batch_dict["lr_vox_size"].to(device)
            x_coords = pitn.affine.affine_coordinate_grid(
                x_affine_vox2world, tuple(x.shape[2:])
            )

            bval = batch_dict["bval"]
            bvec = batch_dict["bvec"]

            pred_dwi = x

            # Write out dwi to a .nii.gz file.
            input_vox_size = x_vox_size.flatten().cpu().numpy()[0]

            # Store the DWI in subdirectory, ~~fit the odfs to the prediction,
            # then bring the wm odf to the root directory.~~

            subj_odf_fit_dir = model_pred_res_dir / subj_id
            subj_odf_fit_dir.mkdir(parents=True, exist_ok=True)
            pred_affine = x_affine_vox2world[0].cpu().numpy()
            # Mask the dwi to reduce the file size.
            pred_dwi = pred_dwi.detach().cpu() * x_mask.detach().cpu()
            pred_dwi_vol = einops.rearrange(
                pred_dwi.numpy(), "1 c x y z -> x y z c"
            ).astype(np.float32)
            pred_dwi_im = nib.Nifti1Image(pred_dwi_vol, affine=pred_affine)

            # Final target filenames for the DWI and the fitted wm odf coeffs.
            # Other output filenames will be derived separately.
            pred_dwi_f = (
                subj_odf_fit_dir
                / f"{subj_id}_dwi_downsampled_bvec-resampled_{input_vox_size}mm.nii.gz"
            )
            # Crop/pad prediction to align with the fodf image created directly from
            # mrtrix. This should not change any of the prediction values, only align
            # the images for easier comparison.
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmp_pred_dir = Path(tmpdirname)
                tmp_unresampled_pred_dwi_f = str(
                    (tmp_pred_dir / "_tmp_dwi.nii.gz").resolve()
                )
                tmp_pred_dwi_f = str((tmp_pred_dir / pred_dwi_f.name).resolve())
                nib.save(pred_dwi_im, tmp_unresampled_pred_dwi_f)
                print(f"Aligning and saving prediction image {subj_id}")
                # Do the resampling with mrtrix directly.
                subj_source_files = (
                    pitn.data.datasets2.HCPfODFINRDataset.get_fodf_subj_dict(
                        subj_id, root_dir=hcp_low_res_fodf_dir
                    )
                )
                subj_source_fodf_f = str(subj_source_files["fodf"].resolve())
                subprocess.run(
                    [
                        "mrgrid",
                        tmp_unresampled_pred_dwi_f,
                        "regrid",
                        "-template",
                        subj_source_fodf_f,
                        "-interp",
                        "nearest",
                        "-scale",
                        "1,1,1",
                        "-datatype",
                        "float32",
                        tmp_pred_dwi_f,
                        "-force",
                    ],
                    # env=os.environ,
                    timeout=60,
                    check=True,
                )

                dwi_im = nib.load(tmp_pred_dwi_f)
                brain_mask_im = nib.load(subj_source_files["mask"])
                freesurfer_seg_im = nib.load(subj_source_files["freesurfer_seg"])
                bval_ = bval.squeeze().cpu().numpy().astype(np.float32)
                bval_ = bval_.flatten()[None]
                bval_f = tmp_pred_dir / "bvals"
                np.savetxt(bval_f, bval_, fmt="%g")
                bvec_ = bvec.squeeze().cpu().numpy().astype(np.float32)
                if bvec_.shape[0] != 3:
                    bvec_ = bvec_.T
                bvec_f = tmp_pred_dir / "bvecs"
                np.savetxt(bvec_f, bvec_, fmt="%g")
                # fodf_fit_fs = dict()
                out_fs = dict()
                out_fs["dwi"] = Path(tmp_pred_dwi_f)
                out_fs["bval"] = Path(bval_f)
                out_fs["bvec"] = Path(bvec_f)

                shutil.move(
                    out_fs["dwi"], subj_odf_fit_dir / f"postproc_{out_fs['dwi'].name}"
                )
                shutil.move(out_fs["bval"], subj_odf_fit_dir / out_fs["bval"].name)
                shutil.move(out_fs["bvec"], subj_odf_fit_dir / out_fs["bvec"].name)

            print(f"Finished {subj_id}", flush=True)

except KeyboardInterrupt as e:
    (tmp_res_dir / "STOPPED").touch()
    raise e
except Exception as e:
    (tmp_res_dir / "FAILED").touch()
    raise e
