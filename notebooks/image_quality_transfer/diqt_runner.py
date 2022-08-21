#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Running script for spawning jobs of the diqt.ipynb notebook.
import datetime
import functools
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from pprint import pprint as ppr

import dotenv
import jupyter_client
import papermill as pm
import torch
import torch.multiprocessing as mp
import yaml
from box import Box

SUCCESS = 0
FAILURE = 1

# fmt: off
splits = [
    {
        "train": {"subjs": ['634748', '386250', '751348', '150019', '910241', '406432', '815247', '690152', '141422', '100408']},
        "val": {"subjs": ['644246', '567759', '231928', '157437']},
        "test": {"subjs": ['701535', '978578', '118124', '894774', '185947', '297655', '135528', '679770', '792867', '567961', '189450', '227432', '108828', '307127', '156637', '803240', '164030', '196952', '753251', '140117', '103515', '198047', '124220', '118730', '303624', '103010', '397154', '700634', '810439', '382242', '203923', '224022', '175035', '167238']},
    },
    {
        "train": {"subjs": ['803240', '118124', '406432', '203923', '224022', '175035', '189450', '198047', '141422', '118730']},
        "val": {"subjs": ['156637', '307127', '894774', '567961']},
        "test": {"subjs": ['701535', '978578', '910241', '185947', '297655', '231928', '690152', '135528', '679770', '792867', '815247', '227432', '108828', '634748', '386250', '751348', '157437', '164030', '150019', '196952', '140117', '753251', '103515', '100408', '124220', '567759', '303624', '103010', '397154', '700634', '810439', '382242', '644246', '167238']},
    },
    {
        "train": {"subjs": ['386250', '679770', '634748', '700634', '978578', '150019', '894774', '406432', '815247', '141422']},
        "val": {"subjs": ['297655', '231928', '140117', '135528']},
        "test": {"subjs": ['701535', '118124', '910241', '185947', '690152', '792867', '189450', '567961', '227432', '108828', '156637', '307127', '803240', '751348', '164030', '157437', '196952', '753251', '103515', '198047', '124220', '100408', '118730', '303624', '567759', '103010', '397154', '810439', '382242', '203923', '224022', '175035', '644246', '167238']},
    },
    # {
    #     "train": {"subjs": ['386250', '803240', '157437', '700634', '978578', '382242', '175035', '753251', '567961', '644246']},
    #     "val": {"subjs": ['224022', '100408', '141422', '397154']},
    #     "test": {"subjs": ['701535', '118124', '894774', '910241', '185947', '297655', '231928', '690152', '135528', '679770', '792867', '815247', '189450', '227432', '108828', '634748', '307127', '751348', '156637', '164030', '150019', '406432', '196952', '140117', '103515', '198047', '124220', '118730', '567759', '303624', '103010', '810439', '203923', '167238']},
    # },
    # {
    #     "train": {"subjs": ['307127', '700634', '150019', '894774', '297655', '203923', '792867', '567961', '303624', '167238']},
    #     "val": {"subjs": ['141422', '382242', '224022', '157437']},
    #     "test": {"subjs": ['701535', '978578', '118124', '910241', '185947', '231928', '690152', '135528', '679770', '815247', '189450', '227432', '108828', '386250', '634748', '156637', '803240', '751348', '164030', '406432', '196952', '140117', '753251', '103515', '198047', '124220', '118730', '567759', '100408', '103010', '397154', '810439', '175035', '644246']},
    # },
]
# fmt: on
split_idx = list(range(1, len(splits) + 1))


def patch_file_stream(file_stream, std_stream, prefix: str):
    def stream_fn_wrapper(fn, fn_key, stdio_stream):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            stdio_stream.__getattribute__(fn_key)(*args, **kwargs)
            return fn(*args, **kwargs)

        return wrapper

    def prefix_write(fn, prefix: str):
        @functools.wraps(fn)
        def wrapper(s: str):
            new_s = s.splitlines(keepends=True)
            new_s = "".join([prefix + " " + str_line for str_line in new_s])
            return fn(new_s)

        return wrapper

    def prefix_writelines(fn, prefix: str):
        @functools.wraps(fn)
        def wrapper(lines: list):
            ls = [prefix + " " + s for s in lines]
            return fn(ls)

        return wrapper

    file_stream.write = stream_fn_wrapper(file_stream.write, "write", std_stream)
    file_stream.write = prefix_write(file_stream.write, prefix)
    file_stream.writelines = stream_fn_wrapper(
        file_stream.writelines, "writelines", std_stream
    )
    file_stream.writelines = prefix_writelines(file_stream.writelines, prefix)
    file_stream.flush = stream_fn_wrapper(file_stream.flush, "flush", std_stream)
    return file_stream


def proc_runner(
    run_params: Box,
    exp_root_name: str,
    gpu_idx: int,
    gpu_idx_queue_bag,
    nb_path: Path,
    kernel_name: str,
    run_work_dir: Path,
    os_environ: dict,
    results_dirs: list,
):
    print("PID ", os.getpid())
    print("GROUP ID ", os.getgid())
    # Update this process' env vars.
    os.environ.update(os_environ)
    # set gpu idx with CUDA_VISIBLE_DEVICES
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    os.environ["CUDA_PYTORCH_DEVICE_IDX"] = str(gpu_idx)

    run_params.override_experiment_name = True
    # Determine tmp directory name for this run.
    ts = datetime.datetime.now().replace(microsecond=0).isoformat()
    # Break ISO format because many programs don't like having colons ':' in a filename.
    ts = ts.replace(":", "_")
    experiment_name = ts + "__" + exp_root_name
    run_params.experiment_name = experiment_name

    # Save out the config file
    with tempfile.TemporaryDirectory(
        prefix=run_params.experiment_name + "__"
    ) as tmpdir_name:
        tmpdir = Path(tmpdir_name)

        conf_fname = tmpdir / "config.yml"
        run_params.to_yaml(conf_fname)
        # add PITN_CONFIG to env vars
        os.environ["PITN_CONFIG"] = str(conf_fname)
        tmp_nb_fname = tmpdir / "tmp_diqt_running.ipynb"
        # Set up stdout and stderr logs.
        stdout_fname = tmpdir / "stdout.log"
        stdout_stream = open(stdout_fname, "w")
        stdout_stream = patch_file_stream(
            stdout_stream, sys.stdout, prefix=f"{os.getpid()} {exp_root_name} |"
        )
        stderr_fname = tmpdir / "stderr.log"
        stderr_stream = open(stderr_fname, "w")
        stderr_stream = patch_file_stream(
            stderr_stream, sys.stderr, prefix=f"{os.getpid()} {exp_root_name} |"
        )

        try:
            # Run the notebook.
            result = pm.execute_notebook(
                input_path=nb_path,
                output_path=tmp_nb_fname,
                language="python",
                cwd=run_work_dir,
                kernel_name=kernel_name,
                log_output=True,
                stdout_file=stdout_stream,
                stderr_file=stderr_stream,
            )

            final_result_dir = None
            for d in results_dirs:
                possible_result_dirs = list(Path(d).glob(run_params.experiment_name))
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
            shutil.copyfile(stdout_fname, final_result_dir / stdout_fname.name)
            shutil.copyfile(stderr_fname, final_result_dir / stderr_fname.name)
        finally:
            gpu_idx_queue_bag.put(gpu_idx)
            stdout_stream.close()
            stderr_stream.close()

    return SUCCESS, result


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
    results_dirs = [env_vars["RESULTS_DIR"], env_vars["TMP_RESULTS_DIR"]]

    # Locate and select source notebook to run.
    source_nb = Path(__file__).parent.resolve() / "diqt.ipynb"
    assert source_nb.exists()
    proc_working_dir = source_nb.parent

    n_gpus = torch.cuda.device_count()
    # n_gpus = 1

    kernel_names = jupyter_client.kernelspec.find_kernel_specs()
    target_kernels = list(filter(lambda kv: "/pitn/" in kv[1], kernel_names.items()))
    print(target_kernels)
    if len(target_kernels) == 1:
        kernel_name = target_kernels[0][0]
    else:
        raise RuntimeError(
            "ERROR: Undetermined kernelspec, got "
            + str(target_kernels)
            + ", expected one valid kernel"
        )
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
        fixed_params.progress_bar = False
        fixed_params.num_workers = os.cpu_count() // n_gpus

        # Create iterable of all desired parameter combinations.
        run_params = list()
        run_basenames = list()
        basename = "uvers_pitn_single_stream"
        for domain in ("le",):
            for i_split, split in zip(reversed(split_idx), reversed(splits)):
                if i_split != 3:
                    continue
                run_p = Box(default_box=False, **fixed_params.copy())
                run_p.merge_update(split)
                run_p.test.dataset_n_subjs = len(run_p.test.subjs)
                run_p.val.dataset_n_subjs = len(run_p.val.subjs)
                run_p.train.dataset_n_subjs = len(run_p.train.subjs)
                run_p.n_subjs = (
                    run_p.test.dataset_n_subjs
                    + run_p.val.dataset_n_subjs
                    + run_p.train.dataset_n_subjs
                )

                if domain == "dti":
                    run_p.use_log_euclid = False
                elif domain == "le":
                    run_p.use_log_euclid = True

                run_p.use_half_precision_float = True
                run_p.use_anat = False
                run_p.train.max_epochs = 50
                run_p.train.grad_2norm_clip_val = 0.25

                run_basenames.append(basename + f"_{domain}_split_{i_split}")
                run_params.append(run_p.copy())

        # Create proc pool, one proc for each GPU.
        with mp.Pool(
            n_gpus,
            maxtasksperchild=1,
            initializer=os.setpgrp,
        ) as pool:

            # Must create a manager to share queues with child processes.
            manager = mp.Manager()
            # Find number of gpus and put them into a "pool" to be used by child procs.
            gpu_idx_pool = manager.Queue(maxsize=n_gpus)
            for i in range(n_gpus):
                gpu_idx_pool.put(i)
            results = list()
            try:
                # Pull gpu indices as they become available, and run a process for each gpu
                for p, basename in zip(run_params, run_basenames):
                    gpu_idx = gpu_idx_pool.get()
                    # Check the status of all "ready" results so far. If any errored out, then
                    # the .get() call will re-raise that exception.
                    list(map(lambda r: r.get(), filter(lambda r: r.ready(), results)))
                    proc_fn = functools.partial(
                        proc_runner,
                        run_params=p,
                        exp_root_name=basename,
                        gpu_idx=gpu_idx,
                        gpu_idx_queue_bag=gpu_idx_pool,
                        nb_path=tmp_nb,
                        run_work_dir=proc_working_dir,
                        kernel_name=kernel_name,
                        os_environ=env_vars,
                        results_dirs=results_dirs,
                    )

                    result = pool.apply_async(proc_fn)
                    # Results won't be "completed" until the pool is join()'ed.
                    results.append(result)

                    # Check the status of all "ready" results so far. If any errored out, then
                    # the .get() call will re-raise that exception.
                    list(map(lambda r: r.get(), filter(lambda r: r.ready(), results)))
                    # Delay next iteration so run names don't have the exact same
                    # starting time.
                    time.sleep(2)
                pool.close()
                pool.join()

            except Exception as e:
                pool.terminate()
                raise e
            finally:
                manager.shutdown()


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
