#!/usr/bin/env python
# -*- coding: utf-8 -*-

# In[1]:


# Automatically re-import project-specific modules.
# imports
import collections
import functools
import io
import itertools
import json
import math
import os
import pathlib
import pprint
import shutil
import subprocess
import tempfile
import time
import typing
import warnings
from pathlib import Path
from pprint import pprint as ppr

import box
import dotenv
import einops
import matplotlib as mpl
import matplotlib.patheffects
import matplotlib.pyplot as plt
import natsort

# Data management libraries.
import nibabel as nib
import nibabel.processing

# Computation & ML libraries.
import numpy as np
import pandas as pd
import redun
import redun.cli
import seaborn as sns
from box import Box
from natsort import natsorted
from redun import File, task

import pitn
from pitn.redun.data.preproc.dwi import top_k_b0s
from pitn.redun.utils import NDArrayValue, join_save_dwis, load_np_txt, save_np_txt

plt.rcParams.update({"figure.autolayout": True})
plt.rcParams.update({"figure.facecolor": [1.0, 1.0, 1.0, 1.0]})

# Set print options for ndarrays/tensors.
np.set_printoptions(suppress=True, threshold=100, linewidth=88)


# In[2]:


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


# ## Directory Setup

# In[3]:


bids_dir = Path("/data/VCU_MS_Study/bids")
assert bids_dir.exists()

output_dir = Path("/data/srv/outputs/pitn/vcu/preproc")
assert output_dir.exists()

selected_subjs = ("P_01", "P_04", "P_08")


# ## Subject File Locations

# In[4]:


subj_files = Box(default_box=True)
for subj in selected_subjs:
    p_dir = bids_dir / subj
    for encode_direct in ("AP", "PA"):
        direct_files = list(p_dir.glob(f"*DKI*{encode_direct}*_[0-9][0-9][0-9].*"))
        dwi = list(filter(lambda p: p.name.endswith(".nii.gz"), direct_files))[0]
        bval = list(filter(lambda p: p.name.endswith("bval"), direct_files))[0]
        bvec = list(filter(lambda p: p.name.endswith("bvec"), direct_files))[0]
        json_info = list(filter(lambda p: p.name.endswith(".json"), direct_files))[0]
        subj_files[subj][encode_direct.lower()] = Box(
            dwi=dwi, bval=bval, bvec=bvec, json=json_info
        )


# ## Data Preprocessing

# In[5]:


# In[6]:


@task(
    hash_includes=[
        pitn.data.utils.least_distort_b0_idx,
        pitn.fsl.phase_encoding_dirs2acqparams,
    ]
)
def vcu_dwi_preproc(
    subj_id: str,
    output_dir: str,
    ap_dwi: File,
    ap_bval: File,
    ap_bvec: File,
    pa_dwi: File,
    pa_bval: File,
    pa_bvec: File,
    ap_json_header_dict: dict,
    pa_json_header_dict: dict,
    eddy_seed: int,
) -> dict:
    assert not isinstance(output_dir, File)

    # Keep track of input files for chaining across tasks.
    input_files = Box()
    input_files["ap"] = dict(dwi=ap_dwi, bvec=ap_bvec, bval=ap_bval)
    input_files["pa"] = dict(dwi=pa_dwi, bvec=pa_bvec, bval=pa_bval)

    # Keep track of all output files for final output.
    outputs = Box(default_box=True)

    # Also track concrete python values for transforming between steps.
    # AP
    ap = Box()
    ap.dwi = nib.load(input_files["ap"]["dwi"].path)
    ap.bvec = np.loadtxt(input_files["ap"]["bvec"].path)
    ap.bval = np.loadtxt(input_files["ap"]["bval"].path)
    ap.json_header = ap_json_header_dict
    # PA
    pa = Box()
    pa.json_header = pa_json_header_dict
    pa.dwi = nib.load(input_files["pa"]["dwi"].path)
    pa.bvec = np.loadtxt(input_files["pa"]["bvec"].path)
    pa.bval = np.loadtxt(input_files["pa"]["bval"].path)

    # Create the output directory in case it doesn't exist.
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # ###### 1. Check bvec orientations
    bvec_flip_correct_path = output_path / "bvec_flip_correct"
    bvec_flip_correct_path.mkdir(exist_ok=True, parents=True)

    tmp_d = bvec_flip_correct_path / "tmp"
    tmp_d.mkdir(exist_ok=True, parents=True)
    for pe_direct, pe_i in zip(("AP", "PA"), (ap, pa)):
        d = str(tmp_d / pe_direct.lower())
        Path(d).mkdir(exist_ok=True, parents=True)
        docker_vols = list(
            set(
                str(Path(f.path).parent)
                for f in (
                    input_files[pe_direct.lower()].dwi,
                    input_files[pe_direct.lower()].bvec,
                    input_files[pe_direct.lower()].bval,
                )
            )
            | {d}
        )
        docker_vols = list((v, v) for v in docker_vols)
        script_exec_config = dict(
            image="dsistudio/dsistudio:chen-2022-08-18",
            executor="docker",
            gpus=0,
            memory=8,
            vcpus=4,
            volumes=docker_vols,
        )
        # with tempfile.TemporaryDirectory(dir=str(bvec_flip_correct_path)) as d:
        correct_bvec = pitn.redun.data.preproc.dwi.bvec_flip_correct_files(
            # dwi_data=NDArrayValue(pe_i.dwi.get_fdata(dtype=np.float32)),
            # dwi_affine=pe_i.dwi.affine,
            dwi_f=input_files[pe_direct.lower()]["dwi"],
            bval_f=input_files[pe_direct.lower()]["bval"],
            bvec_f=input_files[pe_direct.lower()]["bvec"],
            tmp_dir=d,
            script_exec_config=script_exec_config.copy(),
        )
        bvec_file = str(bvec_flip_correct_path / f"{subj_id}_{pe_direct}.bvec")
        bvec_file = save_np_txt(bvec_file, correct_bvec, fmt="%g")
        outputs.bvec_flip_correct[pe_direct.lower()] = bvec_file
        pe_i.bvec = load_np_txt(outputs.bvec_flip_correct[pe_direct.lower()])

    # ###### 2. Run topup
    # Topup really only needs a few b0s in each PE direction, so use image similarity to
    # find the "least distorted" b0s in each PE direction. Then, save out to a file for
    # topup to read.
    topup_out_path = output_path / "topup"
    topup_out_path.mkdir(exist_ok=True, parents=True)
    num_b0s_per_pe = 3
    b0_max = 100
    select_dwi_params = Box(default_box=True)
    for pe_direct, pe_i in zip(("ap", "pa"), (ap, pa)):
        # select_pe = Box()
        top_b0s = top_k_b0s.options(cache=True)(
            NDArrayValue(pe_i.dwi.get_fdata(dtype=np.float32)),
            bval=pe_i.bval,
            bvec=pe_i.bvec,
            n_b0s=num_b0s_per_pe,
            b0_max=b0_max,
        )
        select_dwi_params[pe_direct].dwi = top_b0s["dwi"]
        # Topup doesn't make use of bvec and bval, probably because it only expects to
        # operate over a handful of b0 DWIs.

    combine_b0_files = join_save_dwis(
        dwis=[select_dwi_params["ap"].dwi, select_dwi_params["pa"].dwi],
        affine=ap.dwi.affine,
        dwis_out_f=str(topup_out_path / f"{subj_id}_ap_pa_b0_input.nii.gz"),
    )

    # Create the acquisition parameters file.
    ap_readout_time = float(ap_json_header_dict["EstimatedTotalReadoutTime"])
    ap_pe_direct = ap_json_header_dict["PhaseEncodingAxis"]
    ap_acqparams = pitn.fsl.phase_encoding_dirs2acqparams(
        ap_readout_time, *([ap_pe_direct] * num_b0s_per_pe)
    )
    pa_readout_time = float(pa_json_header_dict["EstimatedTotalReadoutTime"])
    pa_pe_direct = pa_json_header_dict["PhaseEncodingAxis"]
    # The negation of the axis isn't present in these data, for whatever reason.
    if "-" not in pa_pe_direct:
        pa_pe_direct = f"-{pa_pe_direct}"
    pa_acqparams = pitn.fsl.phase_encoding_dirs2acqparams(
        pa_readout_time, *([pa_pe_direct] * num_b0s_per_pe)
    )

    acqparams = np.concatenate([ap_acqparams, pa_acqparams], axis=0)
    acqparams_file = str(topup_out_path / "acqparams.txt")
    topup_acqparams = save_np_txt.options(cache=True)(
        acqparams_file, acqparams, fmt="%g"
    )
    del acqparams

    # Set up docker configuration for running topup.
    docker_vols = {
        str(topup_out_path),
    }
    docker_vols = list((v, v) for v in tuple(docker_vols))
    script_exec_config = dict(
        image="tylerspears/fsl-cuda10.2:6.0.5",
        executor="docker",
        volumes=docker_vols,
        gpus=0,
        memory=12,
        vcpus=4,
    )
    # Finally, run topup.
    # Most parameters were taken from configuration provided by FSL in `b02b0_1.cnf`,
    # which is supposedly optimized for topup runs on b0s when image dimensions are
    # divisible by 1 (a.k.a., all image sizes).
    topup_outputs = pitn.redun.fsl.topup(
        imain=combine_b0_files["dwi"],
        datain=topup_acqparams,
        out=str(topup_out_path / f"{subj_id}_topup"),
        iout=str(topup_out_path / f"{subj_id}_topup_corrected_dwi.nii.gz"),
        fout=str(topup_out_path / f"{subj_id}_topup_displ_field.nii.gz"),
        # Resolution (knot-spacing) of warps in mm
        warpres=(20, 16, 14, 12, 10, 6, 4, 4, 4),
        # Subsampling level (a value of 2 indicates that a 2x2x2 neighbourhood is
        # collapsed to 1 voxel)
        subsamp=(1, 1, 1, 1, 1, 1, 1, 1, 1),
        # FWHM of gaussian smoothing
        fwhm=(8, 6, 4, 3, 3, 2, 1, 0, 0),
        # Maximum number of iterations
        miter=(5, 5, 5, 5, 5, 10, 10, 20, 20),
        # Relative weight of regularisation
        lambd=(
            0.0005,
            0.0001,
            0.00001,
            0.0000015,
            0.0000005,
            0.0000005,
            0.00000005,
            0.0000000005,
            0.00000000001,
        ),
        # If set to 1 lambda is multiplied by the current average squared difference
        ssqlambda=True,
        # Regularisation model
        regmod="bending_energy",
        # If set to 1 movements are estimated along with the field
        estmov=(True, True, True, True, True, False, False, False, False),
        # 0=Levenberg-Marquardt, 1=Scaled Conjugate Gradient
        minmet=("LM", "LM", "LM", "LM", "LM", "SCG", "SCG", "SCG", "SCG"),
        # Quadratic or cubic splines
        splineorder=3,
        # Precision for calculation and storage of Hessian
        numprec="double",
        # Linear or spline interpolation
        interp="spline",
        # If set to 1 the images are individually scaled to a common mean intensity
        scale=True,
        verbose=True,
        log_stdout=True,
        script_exec_config=script_exec_config,
    )
    outputs.topup = topup_outputs

    # ###### 3. Extract mask of unwarped diffusion data.
    bet_out_path = output_path / "bet_topup2eddy"
    bet_out_path.mkdir(exist_ok=True, parents=True)

    # Set up docker configuration for running bet.
    docker_vols = {
        str(bet_out_path),
        str(topup_out_path),
    }
    docker_vols = list((v, v) for v in tuple(docker_vols))
    script_exec_config = dict(
        image="tylerspears/fsl-cuda10.2:6.0.5",
        executor="docker",
        volumes=docker_vols,
        gpus=0,
        memory=8,
        vcpus=4,
    )

    bet_topup2eddy_mask_f = pitn.redun.data.preproc.dwi.bet_mask_median_dwis(
        outputs.topup["corrected_im"],
        out_file=str(bet_out_path / f"{subj_id}_bet_topup2eddy_mask.nii.gz"),
        robust_iters=True,
        script_exec_config=script_exec_config,
    )
    outputs.bet_topup2eddy = bet_topup2eddy_mask_f

    # ###### 4. Run eddy correction
    eddy_out_path = output_path / "eddy"
    eddy_out_path.mkdir(exist_ok=True, parents=True)

    # Create slspec file.
    slspec = pitn.fsl.estimate_slspec(
        ap.json_header, n_slices=ap.dwi.header.get_n_slices()
    )
    slspec_path = eddy_out_path / "slspec.txt"
    slspec_f = save_np_txt.options(cache=True)(str(slspec_path), slspec, fmt="%g")

    # Create index file that relates DWI index to the acquisition params.
    index_path = eddy_out_path / "index.txt"
    ap_acqp_idx = 0
    pa_acqp_idx = len(pa_acqparams)
    index_acqp = np.asarray(
        [ap_acqp_idx] * ap.dwi.shape[-1] + [pa_acqp_idx] * pa.dwi.shape[-1]
    ).reshape(1, -1)
    index_acqp_f = save_np_txt.options(cache=True)(
        str(index_path), index_acqp, fmt="%g"
    )

    # Merge and save both AP and PA DWIs, bvals, and bvecs together.
    ap_pa_basename = f"{subj_id}_ap_pa_uncorrected_dwi"
    ap_pa_dwi_path = eddy_out_path / (ap_pa_basename + ".nii.gz")
    ap_pa_bval_path = eddy_out_path / (ap_pa_basename + ".bval")
    ap_pa_bvec_path = eddy_out_path / (ap_pa_basename + ".bvec")

    ap_pa_files = join_save_dwis(
        dwis=[
            NDArrayValue(ap.dwi.get_fdata(dtype=np.float32)),
            NDArrayValue(pa.dwi.get_fdata(dtype=np.float32)),
        ],
        affine=ap.dwi.affine,
        dwis_out_f=str(ap_pa_dwi_path),
        bvals=[ap.bval, pa.bval],
        bvals_out_f=str(ap_pa_bval_path),
        bvecs=[ap.bvec, pa.bvec],
        bvecs_out_f=str(ap_pa_bvec_path),
    )

    # Set up docker configuration for running eddy with cuda.
    docker_vols = {
        str(eddy_out_path),
        str(bet_out_path),
        str(topup_out_path),
    }
    docker_vols = list((v, v) for v in tuple(docker_vols))
    script_exec_config = dict(
        image="tylerspears/fsl-cuda10.2:6.0.5",
        executor="docker",
        volumes=docker_vols,
        gpus=1,
        memory=16,
        vcpus=12,
    )
    # Run eddy.
    eddy_outputs = pitn.redun.fsl.eddy.options(limits={"gpu": 1})(
        imain=ap_pa_files["dwi"],
        bvecs=ap_pa_files["bvec"],
        bvals=ap_pa_files["bval"],
        mask=outputs.bet_topup2eddy,
        index=index_acqp_f,
        acqp=topup_acqparams,
        slspec=slspec_f,
        topup_fieldcoef=outputs.topup["fieldcoef"],
        topup_movpar=outputs.topup["movpar"],
        niter=10,
        fwhm=[10, 8, 4, 2, 1, 0, 0, 0, 0, 0],
        repol=True,
        rep_noise=True,
        ol_type="both",
        flm="quadratic",
        slm="linear",
        mporder=8,
        s2v_niter=7,
        s2v_lambda=2,
        estimate_move_by_susceptibility=True,
        mbs_niter=10,
        mbs_lambda=10,
        initrand=eddy_seed,
        cnr_maps=True,
        fields=True,
        dfields=True,
        write_predictions=True,
        very_verbose=True,
        log_stdout=True,
        use_cuda=True,
        out=str(eddy_out_path / f"{subj_id}_eddy"),
    )

    outputs.eddy = eddy_outputs

    return outputs.to_dict()


# In[10]:


# Iterate over subjects and run preprocessing task.

# Set up redun scheduler.
repo = "pitn"
redun_conf = redun.cli.setup_config(repo=repo, initialize=True)
sched = redun.Scheduler(config=redun_conf)
sched.load()

eddy_random_seed = 42138

processed_outputs = dict()
for subj_id, files in subj_files.items():

    subj_out_dir = output_dir / subj_id
    subj_out_dir.mkdir(parents=True, exist_ok=True)

    ap_files = files.ap.to_dict()
    ap_json = ap_files.pop("json")
    with open(ap_json, "r") as f:
        ap_json_info = dict(json.load(f))
    ap_files = {k: File(str(ap_files[k])) for k in ap_files.keys()}

    pa_files = files.pa.to_dict()
    pa_json = pa_files.pop("json")
    with open(pa_json, "r") as f:
        pa_json_info = dict(json.load(f))
    pa_files = {k: File(str(pa_files[k])) for k in pa_files.keys()}

    preproc_expr = vcu_dwi_preproc(
        subj_id,
        str(subj_out_dir),
        ap_dwi=ap_files["dwi"],
        ap_bval=ap_files["bval"],
        ap_bvec=ap_files["bvec"],
        ap_json_header_dict=ap_json_info,
        pa_dwi=pa_files["dwi"],
        pa_bval=pa_files["bval"],
        pa_bvec=pa_files["bvec"],
        pa_json_header_dict=pa_json_info,
        eddy_seed=eddy_random_seed,
    )

    processed_outputs[subj_id] = preproc_expr
    # !DEBUG
    break

# Allow all subjects to be scheduled at once.
subj_results = sched.run(processed_outputs)

# In[ ]:
