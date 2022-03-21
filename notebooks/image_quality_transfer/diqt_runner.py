#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Running script for spawning jobs of the diqt.ipynb notebook.
import os
import io
import subprocess
import tempfile
import shutil
from pathlib import Path
import datetime
import atexit

import papermill as pm
import dotenv
from box import Box
import yaml
import torch
import torch.multiprocessing as mp

SUCCESS = 0
FAILURE = 1

def proc_runner(
    param_queue,
    exp_root_name: str,
    nb_path: Path,
    kernel_name: str,
    run_work_dir: Path,
    gpu_idx_queue_bag,
    os_environ: dict,
    results_dirs: list,
    output_queue,
):
    os.setpgrp()
    print("PID ", os.getpid())
    print("GROUP ID ", os.getgid())
    # Update this process' env vars.
    os.environ.update(os_environ)
    # Grab a gpu idx.
    gpu_idx = gpu_idx_queue_bag.get()
    # set gpu idx with CUDA_VISIBLE_DEVICES

    # Grab new params for this run.
    run_params = Box(default_box=True, dot_box=True)
    try:
        new_params = param_queue.get()
    except ValueError:
        print(f"PID {os.getpid()} exiting, queue empty")
        return

    # Merge set and popped params.
    run_params.merge_update(new_params)
    run_params.override_experiment_name = True
    # Determine tmp directory name for this run.
    ts = datetime.datetime.now().replace(microsecond=0).isoformat()
    # Break ISO format because many programs don't like having colons ':' in a filename.
    ts = ts.replace(":", "_")
    experiment_name = ts + "__" + exp_root_name
    run_params.experiment_name = experiment_name

    # Save out the config file
    with tempfile.TemporaryDirectory(prefix=run_params.experiment_Name) as tmpdir_name:
        tmpdir = Path(tmpdir_name)

        conf_fname = tmpdir / "config.yml"
        run_params.to_yaml(conf_fname)
        # add PITN_CONFIG to env vars
        os.environ["PITN_CONFIG"] = str(conf_fname)
        tmp_nb_fname = tmpdir / "temp_nb.ipynb"
        stdout_file = tmpdir / "stdout.log"
        stderr_file = tmpdir / "stderr.log"

        # Run the notebook.
        result = pm.execute_notebook(
            input_path=nb_path,
            output_path=tmp_nb_fname,
            language="python",
            cwd=run_work_dir,
            kernel_name=kernel_name,
            log_output=True,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
        )

        final_result_dir = None
        for d in results_dirs:
            possible_result_dirs = list(d.glob(run_params.experiment_name))
            if len(possible_result_dirs) == 1:
                final_result_dir = possible_result_dirs[0]
                break
            elif len(possible_result_dirs) > 1:
                break
        if final_result_dir is None:
            raise RuntimeError(
                "ERROR: Could not find final result dir "
                + str(run_params.experiment_name)
            )

        # copy stdout and stderr files to final result dir.
        shutil.copyfile(stdout_file, final_result_dir / stdout_file.name)
        shutil.copyfile(stderr_file, final_result_dir / stderr_file.name)

    # If successful, push the filename onto the output queue. Otherwise, push a failure
    # signal.
    output_queue.put(SUCCESS)

    return


def main():
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
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, shell=True, cwd=os.getcwd()
    )
    # Store and format the subprocess' output.
    proc_out = proc.communicate()[0].strip().decode("utf-8")
    # Use python-dotenv to load the environment variables by using the output of
    # 'direnv exec ...' as a 'dummy' .env file.
    dotenv.load_dotenv(stream=io.StringIO(proc_out), override=True)
    # Store global env vars common across all runs.
    env_vars = os.environ.copy()
    for dk in {"DATA_DIR", "WRITE_DATA_DIR", "RESULTS_DIR", "TMP_RESULTS_DIR"}:
        rd = Path(os.environ[dk]).resolve()
        assert rd.exists()
        env_vars[dk] = str(rd)

    # Locate and select source notebook to run.
    source_nb = Path(os.getcwd()).resolve() / "test_runner.ipynb"
    assert source_nb.exists()
    proc_working_dir = source_nb.parent

    # Find number of gpus and put them into a "pool" to be used by child procs.
    n_gpus = torch.cuda.device_count()
    gpu_idx_pool = mp.Queue(maxsize=n_gpus)
    for i in range(n_gpus):
        gpu_idx_pool.put(i)

    # To allow editing of the notebook while experiments are running, copy the source
    # notebook into a temp dir and run with that, while staying in the original
    # notebook's directory.
    with tempfile.TemporaryDirectory(prefix="pitn_diqt_run_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name).resolve()
        tmp_nb = tmp_dir / source_nb.name
        shutil.copyfile(source_nb, tmp_nb)
        assert tmp_nb.exists()

        # Create fixed params for all runs.
        fixed_params = Box(default_box=True, box_dots=True)
        fixed_params.override_experiment_name = True
        fixed_params.n_channels = 6
        fixed_params.progress_bar = False
        fixed_params.num_workers = os.cpu_count() // n_gpus
        fixed_params.hr_center_crop_per_side = 0
        fixed_params.net.kwargs.center_crop_output_side_amt = (
            fixed_params.hr_center_crop_per_side
        )
        fixed_params.train.grad_2norm_clip_val = 0.25
        fixed_params.train.lr_scheduler = None

        # Create iterable of all desired parameter combinations.

        # Create proc pool, one proc for each GPU.

        # Put run params into params queue, block while all child procs are busy.


# Just a list of params, only for human readability. Defaults are set within the
# notebook.
PARAMS_REF = """
###############################################
params.experiment_name = "test_le_anat_stream_comp_tvs_split"
params.override_experiment_name = False
###############################################
# 6 channels for the 6 DTI components
params.n_channels = 6
params.n_subjs = 48
params.lr_vox_size = 2.5
params.fr_vox_size = 1.25
params.use_anat = True
params.use_log_euclid = True
params.use_half_precision_float = True
params.progress_bar = True
params.num_workers = 8

# Data params
params.data.fr_dir = data_dir / f"scale-{params.fr_vox_size:.2f}mm"
params.data.lr_dir = data_dir / f"scale-{params.lr_vox_size:.2f}mm"
params.data.dti_fname_pattern = r"sub-*dti.nii.gz"
params.data.mask_fname_pattern = r"dti/sub-*mask.nii.gz"
params.data.anat_descr = "t2w"
params.data.anat_fname_patterns = [
    f"sub-*t2w.nii.gz",
]
# The data were downsampled artificially by this factor.
params.data.downsampled_by_factor = params.lr_vox_size / params.fr_vox_size
params.data.downsampled_by_factor = (
    int(params.data.downsampled_by_factor)
    if int(params.data.downsampled_by_factor) == params.data.downsampled_by_factor
    else params.data.downsampled_by_factor
)

# This is the number of voxels to remove (read: center crop out) from the network's
# prediction. This allows for an "oversampling" of the low-res voxels to help inform a
# more constrained HR prediction. This value of voxels will be removed from each spatial
# dimension (D, H, and W) starting at the center of the output patches.
# Ex. A size of 1 will remove the 2 outer-most voxels from each dimension in the output,
# while still keeping the corresponding voxels in the LR input.
params.hr_center_crop_per_side = 0

# Maximum allowed eigenvalue for *all* DTIs. This was calculated as the median of the
# eigenvalue thresholds found in the "notebooks/data/dti_thresholding.ipynb" notebook.
# Actual computed value is 0.0033200803422369068, rounded here
# **This counts as outlier removal and will change both the training and test data**
params.data.eigval_clip_cutoff = 0.00332008

# Second data scaling method, where the training data will be scaled and possibly clipped,
# but the testing data will be compared on the originals.
# Scale input data by the valid values of each channel of the DTI.
# I.e., Dx,x in [0, 1], Dx,y in [-1, 1], Dy,y in [0, 1], Dy,z in [-1, 1], etc.
params.data.scale_method = "standard"

# Network params.
# The network's goal is to upsample the input by this factor.
params.net.upscale_factor = params.data.downsampled_by_factor
params.net.kwargs.n_res_units = 3
params.net.kwargs.n_dense_units = 3
params.net.kwargs.interior_channels = 24
params.net.kwargs.anat_in_channels = 1
params.net.kwargs.anat_interior_channels = 14
params.net.kwargs.anat_n_res_units = 2
params.net.kwargs.anat_n_dense_units = 2

params.net.kwargs.activate_fn = "elu"
params.net.kwargs.upsample_activate_fn = "elu"
params.net.kwargs.center_crop_output_side_amt = params.hr_center_crop_per_side

# Adam optimizer kwargs
params.optim.name = "AdamW"
params.optim.kwargs.lr = 2.5e-4
# params.optim.kwargs.lr = 1e-3
params.optim.kwargs.betas = (0.9, 0.999)
params.optim.kwargs.eps = (
    1e-8 if not params.use_half_precision_float else torch.finfo(torch.float16).tiny
)

# # Testing params
# params.test.dataset_subj_percent = 0.4

# # Validation params
# params.val.dataset_subj_percent = 0.2

# Testing params
params.test.subjs = [
    "196952",
    "910241",
    "386250",
    "227432",
    "185947",
    "231928",
    "140117",
    "803240",
    "792867",
    "644246",
    "167238",
    "815247",
    "690152",
    "157437",
    "382242",
    "150019",
    "164030",
    "701535",
    "567961",
    "224022",
    "100408",
    "567759",
    "978578",
    "108828",
    "156637",
    "810439",
    "397154",
    "751348",
    "203923",
    "634748",
    "700634",
    "894774",
    "753251",
    "679770",
]
params.test.dataset_n_subjs = len(params.test.subjs)

# Validation params
params.val.subjs = ["124220", "406432", "141422", "198047"]
params.val.dataset_n_subjs = len(params.val.subjs)

# Training params
params.train.subjs = [
    "307127",
    "118730",
    "175035",
    "297655",
    "103515",
    "303624",
    "135528",
    "103010",
    "189450",
    "118124",
]
params.train.dataset_n_subjs = len(params.train.subjs)

params.train.in_patch_size = (24, 24, 24)
params.train.batch_size = 32
params.train.samples_per_subj_per_epoch = 4000
params.train.max_epochs = 50
params.train.loss_name = "vfro"
params.train.lambda_dti_stream_loss = 0.35
# Percentage of subjs in dataset that go into the training set.
# params.train.dataset_subj_percent = 1 - (
#     params.test.dataset_subj_percent + params.val.dataset_subj_percent
# )
params.train.grad_2norm_clip_val = 0.25
# Learning rate scheduler config.
params.train.lr_scheduler = None
"""

if __name__ == "__main__":
    main()
