#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import collections
import datetime
import functools
import itertools
import math
import os
import sys
import textwrap
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

# Disable cuda blocking **for debugging only!**
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Change default behavior of jax GPU memory allocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".20"
# Set any jax optimization flags before import.
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true " "--xla_gpu_triton_gemm_any=True "
)
# NCCL optimization flags.
os.environ["NCCL_LL128_BUFFSIZE"] = "-2"
os.environ["NCCL_LL_BUFFSIZE"] = "-2"
os.environ["NCCL_PROTO"] = "SIMPLE,LL,LL128"

# utility libraries
import pprint
from pprint import pprint as ppr

import dipy
import dipy.core
import dipy.core.geometry
import dipy.data
import einops

# visualization libraries
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

# numerical libraries
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns

try:
    torch
except NameError:
    import jax

    jax.devices()
else:
    raise RuntimeError(
        "ERROR: Must import jax and instantiate devices before importing pytorch"
    )
import jax.numpy as jnp
import torch
import torch.nn.functional as F
from box import Box
from icecream import ic

import pitn

# Add inr_networks module from a different directory
sys.path.append(str(Path("../continuous_sr/").resolve()))
from inr_networks import INREncoder, SimplifiedDecoder

# Update any jax necessary jax configurations
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_default_matmul_precision", 32)


class FenriInterpFn:
    def __init__(
        self,
        net_params: dict,
        net_weights_f: Path,
        input_dwi: torch.Tensor,
        affine_vox2real: torch.Tensor,
        brain_mask: torch.Tensor,
        device,
    ):
        state_dict = torch.load(net_weights_f)
        # Run forward inference with the encoder, then delete and just use the decoder.
        encoder = INREncoder(**net_params["encoder"])
        encoder.load_state_dict(state_dict["encoder"])
        encoder.to(device)
        encoder.eval()
        # Construct encoder inputs.
        x = input_dwi.to(device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim == 4:
            x = x.unsqueeze(0)
        m = brain_mask.to(device)
        if m.ndim == 3:
            m = m.unsqueeze(0)
        if m.ndim == 3:
            m = m.unsqueeze(0)
        # Mask out non-brain voxels in the input DWI.
        x = x * m
        self.ctx_affine_vox2real = affine_vox2real.to(device=device, dtype=x.dtype)
        self.ctx_affine_vox2real = self.ctx_affine_vox2real.unsqueeze(0)
        x_fov = pitn.affine.fov_bb_coords_from_vox_shape(
            self.ctx_affine_vox2real[0], shape=x.shape[2:]
        )
        self.ctx_coords = pitn.transforms.functional.fov_coord_grid(
            x_fov, self.ctx_affine_vox2real[0]
        )
        self.ctx_coords.unsqueeze_(0)

        # Append LR coordinates to the end of the input LR DWIs.
        x = torch.cat(
            [x, einops.rearrange(self.ctx_coords, "b x y z coord -> b coord x y z")],
            dim=1,
        )
        self.ctx_v = encoder(x)
        del encoder, x, m
        # Instantiate the decoder.
        self.decoder = SimplifiedDecoder(**net_params["decoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.decoder.to(device)
        self.decoder.eval()

    def __call__(self, real_coords_xyz_flat: torch.Tensor) -> torch.Tensor:
        # Reshape to be "volume-like"
        query_coords = einops.rearrange(
            real_coords_xyz_flat, "b coord -> 1 b 1 1 coord"
        )
        # All coordinates are to be considered.
        query_coords_mask = torch.ones(
            tuple(query_coords.shape[:-1]) + (1,),
            dtype=torch.bool,
            device=query_coords.device,
        )
        # query_coords_mask = torch.ones_like(query_coords, dtype=torch.bool)

        pred_sh = self.decoder(
            context_v=self.ctx_v,
            context_real_coords=self.ctx_coords,
            query_real_coords=query_coords,
            query_coords_mask=query_coords_mask,
            affine_context_vox2real=self.ctx_affine_vox2real,
            # Spacing values aren't actually used.
            context_spacing=None,
            query_spacing=None,
        )

        # Reshape to move the spatial dimension as the batch dimension.
        pred_sh = einops.rearrange(pred_sh, "1 sh b 1 1 -> b sh")
        return pred_sh


def log_prefix():
    return f"{datetime.datetime.now().replace(microsecond=0)}"


def nib_fdata_loader(nib_im, dtype=np.float32, **get_fdata_kwargs):
    im = nib_im.get_fdata(**get_fdata_kwargs).astype(dtype)
    if len(im.shape) == 4:
        im = np.moveaxis(im, -1, 0)
    if "caching" in get_fdata_kwargs.keys():
        if get_fdata_kwargs["caching"] == "unchanged":
            nib_im.uncache()
    return im


def tractography_batch(
    p: Box,
    seed_coords_table: pd.DataFrame,
    fn_interp_odf_at_coords,
    fn_peak_finding,
    fn_sh_basis,
    device,
    log_stream,
    find_seed_angles: bool = True,
) -> list[np.ndarray]:
    ############# Seed generation #############
    log_stream.write(f"{log_prefix()}| Starting seed generation of batch\n")
    seed_coords = np.stack(
        [
            seed_coords_table.x.to_numpy(),
            seed_coords_table.y.to_numpy(),
            seed_coords_table.z.to_numpy(),
        ],
        -1,
    )
    seed_coords = torch.from_numpy(seed_coords).to(affine_vox2real)
    seed_sh_coeffs = fn_interp_odf_at_coords(seed_coords)
    # If the seed coordinate table does not have starting directions, then estimate
    # starting directions from samples over the whole ODF.
    if find_seed_angles:
        log_stream.write(f"{log_prefix()}| Estimating starting peak directions\n")
        # Calculate seed angles from one of the built-in dipy spheres.
        seed_sphere = dipy.data.get_sphere(p.seed.dipy_sphere)
        seed_angle_x, seed_angle_y, seed_angle_z = dipy.core.geometry.sphere2cart(
            r=1, theta=seed_sphere.theta, phi=seed_sphere.phi
        )
        seed_angle_xyz = torch.from_numpy(
            np.stack([seed_angle_x, seed_angle_y, seed_angle_z], -1)
        )
        seed_theta, seed_phi = pitn.tract.xyz2unit_sphere_theta_phi(seed_angle_xyz)
        seed_theta = seed_theta.to(torch.float32).to(device)
        seed_phi = seed_phi.to(torch.float32).to(device)
        del seed_sphere, seed_angle_x, seed_angle_y, seed_angle_z, seed_angle_xyz
        #! Should RK4 be used to find the tangent t1? Or is just regular peak-finding
        #! sufficient?
        seed_direction_antipodal = pitn.tract.seed.get_topk_starting_peaks(
            seed_sh_coeffs,
            seed_theta=seed_theta,
            seed_phi=seed_phi,
            fn_peak_finding__sh_theta_phi2theta_phi=fn_peak_finding,
            fn_sh_basis=fn_sh_basis,
            max_n_peaks=p.seed.peaks,
            min_odf_height=p.seed.min_odf_amplitude,
            min_peak_arc_len=p.seed.min_peak_separation_arc_len,
        )
        # Flatten the antipodal sides and merge into the overall batch.
        valid_seed_direction_mask = (
            einops.rearrange(
                seed_direction_antipodal.amplitude,
                "b antipodal n_peak -> (b antipodal) n_peak",
            )
            != 0.0
        )

        seed_theta = einops.rearrange(
            seed_direction_antipodal.theta,
            "b antipodal n_peak -> (b antipodal) n_peak",
        )[valid_seed_direction_mask]
        seed_phi = einops.rearrange(
            seed_direction_antipodal.phi,
            "b antipodal n_peak -> (b antipodal) n_peak",
        )[valid_seed_direction_mask]
        # Duplicate the seed spatial coordinates and match to the valid antipodal peaks.
        points_t1 = einops.repeat(
            seed_coords,
            "b coord -> (b antipodal) n_peak coord",
            antipodal=2,
            n_peak=seed_direction_antipodal.amplitude.shape[-1],
        )[valid_seed_direction_mask, :]
        tangent_t1 = pitn.tract.unit_sphere2xyz(seed_theta, seed_phi)
        log_stream.write(
            f"{log_prefix()}| {valid_seed_direction_mask.sum().cpu().item()} valid seeds\n"
        )
    # Otherwise, the seed coordinate table must include 'theta' and 'phi' columns for
    # that indicate starting directions. This will *not* mirror directions in an
    # antipodal symmetry, this will only take those starting directions as they are
    # given!
    else:
        log_stream.write(f"{log_prefix()}| Using given seed directions\n")
        seed_theta = torch.from_numpy(seed_coords_table.theta.to_numpy()).to(
            dtype=torch.float32, device=device
        )
        seed_phi = torch.from_numpy(seed_coords_table.phi.to_numpy()).to(
            dtype=torch.float32, device=device
        )
        seed_theta, seed_phi = fn_peak_finding(seed_sh_coeffs, seed_theta, seed_phi)
        points_t1 = seed_coords
        tangent_t1 = pitn.tract.unit_sphere2xyz(seed_theta, seed_phi)

    log_stream.write(f"{log_prefix()}| Completed seed generation\n")
    ############# Main tractography loop #############
    all_points = list()
    all_streamline_status = list()
    t_max = math.ceil(p.stopping.max_streamline_len / p.tracking.step_size_mm) + 1

    # Initialize all t variables.
    streamline_status_tm1 = (
        torch.ones(points_t1.shape[0], dtype=torch.int8, device=device)
        * pitn.tract.stopping.CONTINUE
    )
    streamline_len_tm1 = streamline_status_tm1.new_zeros(
        streamline_status_tm1.shape[0], dtype=torch.float32
    )
    # streamline_status_t = streamline_status_t.clone()
    points_t = points_t1
    tangent_t = tangent_t1
    sh_coeff_t = fn_interp_odf_at_coords(points_t)
    sh_basis_t = fn_sh_basis(*pitn.tract.xyz2unit_sphere_theta_phi(tangent_t))
    peak_amps_t = (sh_coeff_t * sh_basis_t).sum(-1)

    log_stream.write(f"{log_prefix()}| Starting tractography batch\n")
    log_stream.write(f"{log_prefix()}| Tractography epoch out of {t_max}:\n|")
    # Loop window centers on the streamline point t, but also needs information from
    # points t-1 and t+1.
    for t in range(1, t_max + 1):
        if t % 25 == 0:
            print(f"{t}...", end="", flush=True)
        continue_t_mask = pitn.tract.stopping.to_continue_mask(
            streamline_status_tm1
        ).unsqueeze(-1)

        # To determine some stop conditions, we need the position and tangent at t+1
        points_tp1 = (
            points_t + (p.tracking.step_size_mm * tangent_t)
        ) * continue_t_mask
        # Tangents should be normalized; scaling by the step size should be done when
        # translating the points at position t.
        tangent_tp1 = (
            pitn.tract.gen_tract_step_rk4(
                points_tp1,
                init_direction=tangent_t,
                step_size=p.tracking.step_size_mm,
                fn_xyz_seed_direction2out_direction=fn_xyz_in_direction2out_direction,
            )
            / p.tracking.step_size_mm
        )
        # "Override" tangent at point t+1 with an exponential moving average tangent
        # vector.
        ema_tangent_tp1 = (
            p.tracking.alpha_exponential_moving_avg * tangent_tp1
            + (1 - p.tracking.alpha_exponential_moving_avg) * tangent_t
        )
        norm_ema_tangent_tp1 = torch.linalg.vector_norm(
            ema_tangent_tp1, ord=2, dim=-1, keepdim=True
        )
        # Replace near-0 norms with norms of 1, to avoid division by 0.
        norm_ema_tangent_tp1 = torch.where(
            torch.isclose(norm_ema_tangent_tp1, norm_ema_tangent_tp1.new_zeros(1)),
            norm_ema_tangent_tp1.new_ones(1),
            norm_ema_tangent_tp1,
        )
        # Normalize tangents weighted combination of tangent vectors.
        ema_tangent_tp1 /= norm_ema_tangent_tp1
        tangent_tp1 = ema_tangent_tp1

        tangent_tp1 *= continue_t_mask
        sh_coeff_tp1 = fn_interp_odf_at_coords(points_tp1)
        sh_basis_tp1 = fn_sh_basis(*pitn.tract.xyz2unit_sphere_theta_phi(tangent_tp1))
        peak_amps_tp1 = (sh_coeff_tp1 * sh_basis_tp1).sum(-1)

        # Check stopping criteria
        statuses_t = list()
        # Stop the streamline at point t if the next point goes outside the brain mask.
        statuses_t.append(
            pitn.tract.stopping.vol_sample_threshold(
                streamline_status_tm1,
                brain_mask,
                affine_vox2real=affine_vox2real,
                sample_coords=points_tp1,
                sample_min=0.99,
                mode="nearest",
                align_corners=True,
            )
        )
        # Stop the streamline at point t if t+1 has too low of a GFA.
        statuses_t.append(
            pitn.tract.stopping.gfa_threshold(
                streamline_status_tm1,
                gfa_min_threshold=p.stopping.min_gfa,
                sh_coeff=sh_coeff_tp1,
            )
        )
        # Stop streamline at t if the angular change between t and t+1 is too large.
        statuses_t.append(
            pitn.tract.stopping.angular_threshold(
                streamline_status_tm1,
                angle_x=tangent_t,
                angle_y=tangent_tp1,
                max_radians=p.stopping.max_tangent_angle_diff,
            )
        )
        # Stop the streamline at point t if the t peak amplitude is too low.
        statuses_t.append(
            pitn.tract.stopping.val_threshold(
                streamline_status_tm1,
                peak_amps_t,
                val_min_thresh=p.stopping.min_odf_amplitude,
            )
        )
        # Check if streamline should be stopped at point t due to length.
        statuses_t.append(
            pitn.tract.stopping.streamline_len_mm(
                streamline_status_tm1,
                streamline_len_tm1 + p.tracking.step_size_mm,
                max_len=p.stopping.max_streamline_len,
            )
        )
        streamline_status_t = pitn.tract.stopping.merge_status(
            streamline_status_tm1, *statuses_t
        )
        # Add to the streamline length if it is currently continuing, OR if it was just
        # stopped at this iteration.
        streamline_len_t = streamline_len_tm1 + p.tracking.step_size_mm * (
            (streamline_status_t == pitn.tract.stopping.CONTINUE)
            | (
                (streamline_status_tm1 == pitn.tract.stopping.CONTINUE)
                & (streamline_status_t == pitn.tract.stopping.STOP)
            )
        )
        # Check if streamline is too short to be valid.
        streamline_status_t = pitn.tract.stopping.merge_status(
            streamline_status_t,
            pitn.tract.stopping.streamline_len_mm(
                streamline_status_t,
                streamline_len_t,
                min_len=p.stopping.min_streamline_len,
            ),
        )

        all_streamline_status.append(streamline_status_t.cpu())
        all_points.append(points_t.cpu())
        streamline_status_tm1 = streamline_status_t
        streamline_len_tm1 = streamline_len_t
        points_t = points_tp1
        tangent_t = tangent_tp1
        sh_coeff_t = sh_coeff_tp1
        sh_basis_t = sh_basis_tp1
        peak_amps_t = peak_amps_tp1

        # If no streamlines should continue, break.
        if (~pitn.tract.stopping.to_continue_mask(streamline_status_t)).all():
            break

    log_stream.write(f"\n{log_prefix()}| Finished tractography batch at epoch {t}\n")

    all_points = torch.stack(all_points, dim=1).cpu().numpy()
    all_streamline_status = torch.stack(all_streamline_status, dim=1).cpu().numpy()
    all_valid_streamlines = list()
    for i_streamline in range(all_points.shape[0]):
        s_i = all_points[i_streamline]
        status_i = all_streamline_status[i_streamline]
        if (status_i == pitn.tract.stopping.INVALID).any():
            continue
        if (status_i == pitn.tract.stopping.STOP).any():
            end_idx = np.argwhere((status_i == pitn.tract.stopping.STOP)).min()
        else:
            end_idx = -1
        all_valid_streamlines.append(s_i[:end_idx])

    return all_valid_streamlines


if __name__ == "__main__":
    ############# Initial setup #############
    plt.rcParams.update({"figure.autolayout": True})
    plt.rcParams.update({"figure.facecolor": [1.0, 1.0, 1.0, 1.0]})
    plt.rcParams.update({"image.cmap": "gray"})
    plt.rcParams.update({"image.interpolation": "antialiased"})
    # Set print options for ndarrays/tensors.
    np.set_printoptions(suppress=True, threshold=100, linewidth=88)
    torch.set_printoptions(sci_mode=False, threshold=100, linewidth=88)

    parser = argparse.ArgumentParser(
        description="Perform tractography using trilinear ODF interpolation or FENRI on DWIs"
    )
    parser.add_argument(
        "tracking_interp",
        type=str,
        choices=("trilinear", "fenri"),
        help="Select type of interpolation to perform during tracking",
    )
    parser.add_argument(
        "src_vol",
        type=Path,
        help="Input NIFTI file. "
        + "Should likely be an upsampled ODF if testing trilinear interp tracking, "
        + "or a downsampled DWI if testing FENRI tracking.",
    )
    parser.add_argument(
        "seeds",
        type=Path,
        help="Seeds .csv file that contains real (xyz in mm) coordinates of all seed points",
    )
    parser.add_argument(
        "tracking_config",
        type=Path,
        help="Parameters file to control tractography behavior; JSON, YAML, or TOML",
    )
    parser.add_argument(
        "-m",
        "--mask",
        type=Path,
        required=True,
        help="Brain mask NIFTI file, must be the same size and affine as the input vol",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output .tck file",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=Path,
        help="Pytorch .pt weights file with trained network weights; "
        + "required if using 'fenri' tracking",
    )
    parser.add_argument(
        "-n",
        "--network_config",
        type=Path,
        help="Config file to determine network hyperparams, can be JSON, YAML, or TOML; "
        + "required if using 'fenri' tracking",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device to run tractography calculations (default 'cpu')",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=None,
        help="Device ID, only used when device=cuda (default None)",
    )
    parser.add_argument(
        "-l",
        "--log",
        type=argparse.FileType("at"),
        default="-",
        help="Output log file (optional, default stdout)",
    )
    args = parser.parse_args()

    # torch setup
    # allow for CUDA usage, if selected
    if torch.cuda.is_available() and "cuda" in args.device.casefold():
        # Pick only one device for the default, may use multiple GPUs for training later.
        if args.device_id is not None:
            dev_idx = int(args.device_id)
        else:
            dev_idx = 0
        device = torch.device(f"cuda:{dev_idx}")
        print("CUDA Device IDX ", dev_idx)
        torch.cuda.set_device(device)
        print("CUDA Current Device ", torch.cuda.current_device())
        print("CUDA Device properties: ", torch.cuda.get_device_properties(device))
        # set jax default device
        jax.config.update("jax_default_device", jax.devices()[dev_idx])
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = True
        # See
        # <https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices>
        # for details.
        torch.set_float32_matmul_precision("medium")
        # Activate cudnn benchmarking to optimize convolution algorithm speed.
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = True
            print("CuDNN convolution optimization enabled.")
            # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
            torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    # keep device as the cpu
    # device = torch.device('cpu')
    print(device)

    p = Box(default_box=True, default_box_none_transform=False)
    # Load user config file.
    config_fname = args.tracking_config
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
    # Remove the default_box behavior now that params have been fully read in.
    _p = Box(default_box=False)
    _p.merge_update(p)
    p = Box(_p, default_box=False, frozen_box=True)

    src_vol_f = args.src_vol
    mask_f = args.mask
    seeds_f = args.seeds
    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    log_stream = args.log

    log_stream.write(f"{log_prefix()}| Starting tractography\n")
    log_stream.write(
        textwrap.dedent(
            f"""{log_prefix()}| Tracking parameters:
        Tracking interpolation method {args.tracking_interp}
        Source vol {str(src_vol_f)}
        Brain mask {str(mask_f)}
        Seeds file {str(seeds_f)}
        PyTorch device {str(device)}
        Config file {str(args.tracking_config)} with params:
        {pprint.pformat(p.to_dict())}\n"""
        )
    )
    if args.tracking_interp == "fenri":
        # Load network config file
        net_p = Box(default_box=True, default_box_none_transform=False)
        # Load config file.
        net_config_fname = args.network_config
        f_type = net_config_fname.suffix.casefold()
        if f_type in {".yaml", ".yml"}:
            net_f_params = Box.from_yaml(filename=net_config_fname)
        elif f_type == ".json":
            net_f_params = Box.from_json(filename=net_config_fname)
        elif f_type == ".toml":
            net_f_params = Box.from_toml(filename=net_config_fname)
        else:
            raise RuntimeError()
        net_p.merge_update(net_f_params)

        # Remove the default_box behavior now that params have been fully read in.
        _net_p = Box(default_box=False)
        _net_p.merge_update(net_p)
        net_p = Box(_net_p, default_box=False, frozen_box=True)
        log_stream.write(
            textwrap.dedent(
                f"""Network weights file {str(args.weights)}
                Network config file {str(args.network_config)} with params:
                {pprint.pformat(net_p.to_dict())}\n"""
            )
        )
    else:
        net_p = None
    log_stream.write("\n")

    ############# Data loading #############
    log_stream.write(f"{log_prefix()}| Loading data\n")
    src_vol_im = nib.load(src_vol_f)
    mask_im = nib.load(mask_f)
    # Assert source vol and brain mask have the same affine transform.
    assert np.isclose(src_vol_im.affine, mask_im.affine).all()
    affine_vox2real = torch.from_numpy(src_vol_im.affine).to(
        dtype=torch.float32, device=device
    )
    src_vol = nib_fdata_loader(src_vol_im, dtype=np.float32, caching="unchanged")
    brain_mask = nib_fdata_loader(mask_im, dtype=bool, caching="unchanged")
    src_vol = torch.from_numpy(src_vol).to(device)
    brain_mask = torch.from_numpy(brain_mask).bool().unsqueeze(0).to(device)
    # Assert source vol and brain mask have the same voxel size.
    assert tuple(src_vol.shape[1:]) == tuple(brain_mask.shape[1:])

    # Load seed coordinates.
    seed_coords_table = pd.read_csv(
        seeds_f, comment="#", skip_blank_lines=True, skipinitialspace=True
    )
    log_stream.write(f"{log_prefix()}| Done loading data\n")

    ############# Set up callback functions for tracking #############
    # Accepts sh_coeffs, theta, phi; returns theta, phi
    fn_peak_finding = partial(
        pitn.tract.peak_finding.find_peak_grad_ascent,
        lr=p.peak_finding.lr,
        momentum=p.peak_finding.momentum,
        max_epochs=p.peak_finding.max_epochs,
        l_max=p.l_max,
        tol_angular=p.peak_finding.tol_arc_len,
    )
    # Gives SH basis as a vector
    _broad_l, _broad_m = pitn.tract.peak_finding._get_degree_order_vecs(
        p.l_max, batch_size=1, device=device
    )
    _l = _broad_l.flatten()
    _m = _broad_m.flatten()
    # Accepts theta, phi; returns [B x n_coeffs] matrix.
    fn_sh_basis = partial(pitn.tract.sh_basis_mrtrix3, degree=_l, order=_m)

    ## Main ODF prediction/interpolation function
    # Trilinear interpolation tractography
    if args.tracking_interp == "trilinear":

        def interp_odf_at_coords(
            real_coords_xyz_flat: torch.Tensor,
            odf_vol: torch.Tensor,
            affine_vox2real: torch.Tensor,
        ) -> torch.Tensor:
            # All coordinates are being sampled from the same volume, so the batch size
            # should be 1, and the coordinates should have spatial dimensions.
            real_coords_xyz = einops.rearrange(
                real_coords_xyz_flat, "b coord -> 1 b 1 1 coord"
            )
            affine_vox2real = affine_vox2real.unsqueeze(0)

            pred_sh = pitn.affine.sample_vol(
                odf_vol,
                coords_mm_xyz=real_coords_xyz,
                affine_vox2mm=affine_vox2real,
                mode="trilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            # Reshape to move the spatial dimension as the batch dimension.
            pred_sh = einops.rearrange(pred_sh, "1 sh b 1 1 -> b sh")
            return pred_sh

        fn_interp_odf_at_coords = partial(
            interp_odf_at_coords, odf_vol=src_vol, affine_vox2real=affine_vox2real
        )
    # FENRI tractography
    elif args.tracking_interp == "fenri":
        fn_interp_odf_at_coords = FenriInterpFn(
            net_p.to_dict(),
            net_weights_f=args.weights,
            input_dwi=src_vol,
            affine_vox2real=affine_vox2real,
            brain_mask=brain_mask,
            device=device,
        )

    # Function that composes interpolation and peak finding for a general
    # (spatial coord, incoming direction) -> outgoing direction
    def xyz_in_direction2out_direction(
        coord_xyz: torch.Tensor,
        seed_direction_xyz: torch.Tensor,
        fn_interp_at_coord: Callable,
        fn_peak_finder__coeff_theta_phi2theta_phi: Callable,
    ):
        sh_coeffs = fn_interp_at_coord(coord_xyz)
        seed_theta, seed_phi = pitn.tract.xyz2unit_sphere_theta_phi(seed_direction_xyz)
        out_theta, out_phi = fn_peak_finder__coeff_theta_phi2theta_phi(
            sh_coeffs, seed_theta, seed_phi
        )
        out_direction = pitn.tract.unit_sphere2xyz(out_theta, out_phi)

        return out_direction

    fn_xyz_in_direction2out_direction = partial(
        xyz_in_direction2out_direction,
        fn_interp_at_coord=fn_interp_odf_at_coords,
        fn_peak_finder__coeff_theta_phi2theta_phi=fn_peak_finding,
    )

    # Run tractography in batches
    if ("theta" in seed_coords_table.columns) and ("phi" in seed_coords_table.columns):
        find_seed_angles = False
    else:
        find_seed_angles = True
    n_seed_positions = seed_coords_table.shape[0]
    # If the starting seed angles are given in the seed table, then there is only one
    # starting (coordinate, angle) per row in the seed table.
    n_starts_per_seed = 2 * p.seed.peaks if find_seed_angles else 1
    n_total_seeds = n_seed_positions * n_starts_per_seed

    n_batches = math.ceil(n_total_seeds / p.batch_size)
    batch_idx = list(range(0, n_seed_positions, p.batch_size // n_starts_per_seed))

    if batch_idx[-1] != (n_seed_positions - 1):
        batch_idx.append(n_seed_positions - 1)

    all_streamlines = list()
    total_completed_seeds = 0
    for i_batch, (batch_start_idx, batch_end_idx) in enumerate(
        itertools.pairwise(batch_idx)
    ):
        log_stream.write(f"{log_prefix()}| Starting batch {i_batch+1}/{n_batches}\n")
        seeds_this_batch = batch_end_idx - batch_start_idx
        batch_seed_coords_table = seed_coords_table.iloc[batch_start_idx:batch_end_idx]

        streamlines_batch_i = tractography_batch(
            p=p,
            seed_coords_table=batch_seed_coords_table,
            fn_interp_odf_at_coords=fn_interp_odf_at_coords,
            fn_peak_finding=fn_peak_finding,
            fn_sh_basis=fn_sh_basis,
            device=device,
            log_stream=log_stream,
            find_seed_angles=find_seed_angles,
        )
        all_streamlines.extend(streamlines_batch_i)
        total_completed_seeds += seeds_this_batch
        log_stream.write(f"{log_prefix()}| Completed batch {i_batch+1}/{n_batches}\n")
        log_stream.write(
            f"{log_prefix()}| Batch contained {len(streamlines_batch_i)} streamlines\n"
        )

        log_stream.write(
            f"{log_prefix()}| Seeds completed {total_completed_seeds}/{n_seed_positions}\n"
        )

    # Merge streamlines that may share the same starting point, but should be one
    # streamline.
    merged_streamlines = pitn.tract.join_streamlines(
        all_streamlines,
        min_match_arc_len=90 * (np.pi / 180),
        antipodal_arc_len_thresh=p.stopping.max_tangent_angle_diff,
    )
    log_stream.write(f"{log_prefix()}| Finished tractography\n")

    log_stream.write(f"{log_prefix()}| Saving streamlines\n")
    tracto = dipy.io.streamline.StatefulTractogram(
        merged_streamlines,
        space=dipy.io.stateful_tractogram.Space.RASMM,
        reference=src_vol_im.header,
    )
    tractogram_fname = args.output
    dipy.io.streamline.save_tck(tracto, str(tractogram_fname))
    log_stream.write(f"{log_prefix()}| Finished saving streamlines\n")
