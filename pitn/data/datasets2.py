# -*- coding: utf-8 -*-
import collections
import copy
import functools
import itertools
import math
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import einops
import monai
import nibabel as nib
import numpy as np
import scipy
import skimage
import torch
from monai.utils import (
    convert_to_tensor,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
)

import pitn


class DWIDataDict(TypedDict):
    dwi: torch.Tensor
    bval: torch.Tensor
    bvec: torch.Tensor


def resample_dwi_to_grad_directions(
    dwi: torch.Tensor,
    src_bvec: torch.Tensor,
    src_bval: torch.Tensor,
    target_bvec: torch.Tensor,
):
    K = 5
    bval_round_decimals = -2
    # Assume that the src and target bvecs are referring to the same gradient strengths
    # (bvals), just different orientations.
    x_g = src_bvec.detach().cpu().numpy().T
    y_g = target_bvec.detach().cpu().numpy().T
    bval = src_bval.detach().cpu().numpy()
    shells = np.round(bval, decimals=bval_round_decimals).astype(int)
    # If target and source are b0s, then no re-weighting should be done, as there is no
    # gradient.

    d_cos = scipy.spatial.distance.cdist(y_g, x_g, "cosine")
    sim = 1 - d_cos
    sim = np.clip(np.abs(sim), a_min=None, a_max=1 - 1e-5)
    l = np.arccos(sim)
    # For each shell (excluding b=0), restrict the available dwis to only the matching
    # shell.
    unique_shells = set(np.unique(shells).tolist()) - {0}
    for s in unique_shells:
        s_mask = np.isclose(shells, s)
        shell_intersection_mask = np.logical_and(s_mask[:, None], s_mask[None, :])
        shell_dissimilar_mask = ~shell_intersection_mask
        l[shell_dissimilar_mask * s_mask[:, None]] = np.inf

    top_k_idx = np.argsort(l, axis=1, kind="stable")[:, :K]

    w = np.take_along_axis(1 / np.clip(l, a_min=1e-5, a_max=None), top_k_idx, axis=1)
    w = w / w.sum(axis=1, keepdims=True)

    w = torch.from_numpy(w).to(dwi)
    top_k_idx = torch.from_numpy(top_k_idx).to(dwi).long()
    # Start with identity convolution, which will be left for the b0s.
    w_conv = torch.eye(dwi.shape[0]).to(dwi)
    shells = torch.from_numpy(shells).to(dwi.device)
    for i_y in range(dwi.shape[0]):
        shell_i = shells[i_y]
        if shell_i == 0:
            continue
        top_k_i = top_k_idx[i_y]
        w_i = w[i_y]
        # zero-out the row for y_i
        w_conv[i_y] = 0
        w_conv[i_y, top_k_i] = w_i

    w_conv = w_conv[:, :, None, None, None]
    dwi_target = torch.nn.functional.conv3d(dwi[None], w_conv)
    dwi_target = dwi_target[0]

    # w = torch.from_numpy(w).to(dwi)

    # top_k_idx = torch.from_numpy(top_k_idx).to(src_bval).long()
    # # dwi_np = dwi.detach().cpu().numpy()
    # dwi_target = monai.inferers.sliding_window_inference(
    #     dwi[None],
    #     roi_size=(72, 72, 72),
    #     sw_batch_size=1,
    #     predictor=lambda dw: torch.from_numpy(
    #         einops.einsum(
    #             np.take(dw[0].cpu().numpy(), top_k_idx, axis=0),
    #             w,
    #             "i j x y z,i j -> i x y z",
    #         )
    #     ).to(dw)[None],
    #     overlap=0,
    #     padding_mode="replicate",
    # )[0]
    # # Use ein. notation to weight and sum over K closest gradient directions.
    # dwi_target = einops.einsum(
    #     torch.take(dwi_np, top_k_idx, axis=0), w, "i j x y z,i j -> i x y z"
    # )
    # Re-assign the b0s
    # b0_mask = torch.from_numpy(np.isclose(y_g, 0.0).all(1)).to(src_bval).bool()
    # dwi_target[b0_mask] = dwi[b0_mask]

    return dwi_target, target_bvec.to(src_bvec)


class ResampleDWItoBvecd(monai.transforms.MapTransform):
    backend = [
        monai.utils.enums.TransformBackends.TORCH,
        monai.utils.enums.TransformBackends.NUMPY,
    ]

    def __init__(
        self,
        dwi_key,
        src_bvec_key,
        src_bval_key,
        target_bvec: torch.Tensor,
        allow_missing_keys: bool = False,
    ):
        keys = [dwi_key]
        super().__init__(keys, allow_missing_keys)
        self.dwi_key = dwi_key
        self.src_bvec_key = src_bvec_key
        self.src_bval_key = src_bval_key
        self.target_bvec = target_bvec

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        src_bvec = d[self.src_bvec_key]
        src_bval = d[self.src_bval_key]
        for k_i, dwi_key in enumerate(self.key_iterator(d)):
            # We only expect one dwi key to be changed.
            assert k_i == 0
            new_dwi, new_bvec = resample_dwi_to_grad_directions(
                d[dwi_key],
                src_bvec=src_bvec,
                src_bval=src_bval,
                target_bvec=self.target_bvec,
            )

            d[dwi_key] = new_dwi
            d[self.src_bvec_key] = new_bvec

        return d


def sub_select_dwi_from_bval(
    dwi: torch.Tensor,
    bval: torch.Tensor,
    bvec: torch.Tensor,
    shells_to_remove: list[float] = list(),
    within_shell_idx_to_keep: dict[float, tuple[int]] = dict(),
    bval_round_decimals: int = -2,
) -> DWIDataDict:

    keep_mask = torch.ones_like(bval).bool()

    shells = torch.round(bval, decimals=bval_round_decimals)

    for s in shells_to_remove:
        keep_mask[torch.isclose(shells, shells.new_tensor(float(s)))] = False

    for s, idx_to_keep in within_shell_idx_to_keep.items():
        # Sub-select only bvals in this shell.
        shell_mask = torch.isclose(shells, shells.new_tensor(float(s))).bool()
        within_shell_idx_to_keep = bval.new_tensor(tuple(idx_to_keep)).long()
        within_shell_mask = shell_mask[shell_mask]
        within_shell_mask[within_shell_idx_to_keep] = False
        within_shell_mask_to_keep = ~within_shell_mask
        # Merge current running mask with the sub-selected shell.
        keep_mask[shell_mask] = keep_mask[shell_mask] * within_shell_mask_to_keep

    return {"dwi": dwi[keep_mask], "bval": bval[keep_mask], "bvec": bvec[:, keep_mask]}


class SubSelectDWIfromBvald(monai.transforms.MapTransform):
    backend = [
        monai.utils.enums.TransformBackends.TORCH,
        monai.utils.enums.TransformBackends.NUMPY,
    ]

    def __init__(
        self,
        dwi_key,
        bval_key,
        bvec_key,
        bval_sub_sample_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], DWIDataDict
        ],
        allow_missing_keys: bool = False,
    ):
        keys = [dwi_key]
        super().__init__(keys, allow_missing_keys)
        self.dwi_key = dwi_key
        self.bval_key = bval_key
        self.bvec_key = bvec_key
        self.bval_sub_sample_fn = bval_sub_sample_fn

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        bval = d[self.bval_key]
        bvec = d[self.bvec_key]
        for k_i, dwi_key in enumerate(self.key_iterator(d)):
            # We only expect one dwi key to be changed.
            assert k_i == 0
            overwrite_dict = self.bval_sub_sample_fn(
                dwi=d[dwi_key],
                bval=bval,
                bvec=bvec,
            )
            d[dwi_key] = overwrite_dict["dwi"]

            d[self.bval_key] = overwrite_dict["bval"]
            d[self.bvec_key] = overwrite_dict["bvec"]

        return d


# Save the affine as its own field.
def _extract_affine(src_vol):
    aff = src_vol.affine
    if torch.is_tensor(aff):
        aff = torch.clone(aff).to(torch.float32)
    else:
        aff = np.array(np.asarray(aff), copy=True, dtype=np.float32)
    return aff


class HCPfODFINRDataset(monai.data.Dataset):

    _SAMPLE_KEYS = (
        "subj_id",
        # "lr_dwi",
        # "lr_mask",
        # "lr_bval",
        # "lr_bvec",
        # "lr_fodf",
        # "lr_fivett",
        # "lr_wm_mask",
        # "lr_gm_mask",
        # "lr_csf_mask",
        # "affine_lr_vox2world",
        "dwi",
        "bval",
        "bvec",
        "fodf",
        "brain_mask",
        "fivett",
        "wm_mask",
        "gm_mask",
        "csf_mask",
        "sampling_mask",
        "affine_vox2world",
    )

    def __init__(
        self,
        subj_ids,
        dwi_root_dir: Path,
        fodf_root_dir: Path,
        transform=None,
    ):
        self.subj_ids = list(subj_ids)
        self.dwi_root_dir = Path(dwi_root_dir).resolve()
        self.fodf_root_dir = Path(fodf_root_dir).resolve()
        data = list()
        for sid in self.subj_ids:
            hcp_data = self.get_hcp_subj_dict(sid, self.dwi_root_dir)
            fodf_data = self.get_fodf_subj_dict(sid, self.fodf_root_dir)
            sid_data = dict(
                subj_id=hcp_data["subj_id"],
                dwi=hcp_data["dwi"],
                bval=hcp_data["bval"],
                bvec=hcp_data["bvec"],
                fodf=fodf_data["fodf"],
                brain_mask=fodf_data["mask"],
                fivett=fodf_data["fivett"],
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
                d, "postproc_*wm*csd*fod*.nii.gz"
            ),
            mask=d / "postproc_nodif_brain_mask.nii.gz",
            fivett=pitn.utils.system.get_file_glob_unique(d, "postproc*5tt*.nii.gz"),
            freesurfer_seg=pitn.utils.system.get_file_glob_unique(
                d, "postproc*aparc*.nii.gz"
            ),
        )

        return data

    @staticmethod
    def default_pre_sample_tf(
        sample_mask_key="wm_mask",
        mask_dilate_radius: int = 0,
        bval_sub_sample_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], DWIDataDict]
        ] = None,
    ):

        VOL_KEYS = (
            "dwi",
            "fodf",
            "brain_mask",
            "fivett",
        )

        tfs = list()
        # Load images
        vol_reader = monai.data.NibabelReader(as_closest_canonical=True)
        tfs.append(
            monai.transforms.LoadImaged(
                VOL_KEYS,
                reader=vol_reader,
                dtype=np.float32,
                meta_key_postfix="meta",
                image_only=False,
                ensure_channel_first=True,
                simple_keys=True,
            )
        )
        tfs.append(
            monai.transforms.Lambdad(
                ("bval", "bvec"), lambda f: np.loadtxt(str(f)), overwrite=True
            )
        )

        # Data conversion
        tfs.append(
            monai.transforms.ToTensord(
                VOL_KEYS,
                track_meta=True,
            )
        )
        # Crop all volumes to match the fodf, as the fodf has been mask-cropped.
        tfs.append(
            monai.transforms.ResampleToMatchd(
                ["dwi", "brain_mask", "fivett"],
                key_dst="fodf",
                align_corners=True,
                mode="nearest",
                dtype=[torch.float32, torch.uint8, torch.uint8],
            )
        )

        tfs.append(monai.transforms.ToTensord(("bval", "bvec"), track_meta=False))

        # Utility function to flip the X-axis of the bvecs, which is what happens to
        # the DWIs themselves when loading HCP data (LAS) with nibabel's "canonical"
        # orientation setting (RAS).
        def bvec_las2ras(bvec):
            aff = torch.eye(4).to(bvec)
            aff[0, 0] *= -1
            new_bvec = pitn.affine.transform_coords(bvec.T, aff).T
            return new_bvec

        # At least with mrtrix, the bvecs do not need to be flipped, even though
        # the DWIs are reoriented from LAS to RAS. I don't know why, but the odf lobes
        # end up being flipped incorrectly (L-R) if the bvec_las2ras function is used.
        # tfs.append(monai.transforms.Lambdad("bvec", bvec_las2ras, overwrite=True))

        tfs.append(
            monai.transforms.CastToTyped(("brain_mask", "fivett"), dtype=torch.uint8)
        )
        tfs.append(
            ResampleDWItoBvecd(
                "dwi",
                src_bvec_key="bvec",
                src_bval_key="bval",
                target_bvec=pitn.data.HCP_STANDARD_3T_BVEC,
            )
        )

        if bval_sub_sample_fn is not None:
            tfs.append(
                SubSelectDWIfromBvald(
                    dwi_key="dwi",
                    bval_key="bval",
                    bvec_key="bvec",
                    bval_sub_sample_fn=bval_sub_sample_fn,
                )
            )
        # Re-cast the bvec and bval into non meta-tensors.
        tfs.append(monai.transforms.ToTensord(("bval", "bvec"), track_meta=False))

        # Extract different tissue masks according to the 5tt designations.
        tfs.append(
            monai.transforms.SplitDimd(
                keys=("fivett",),
                output_postfixes=(
                    "cort_gm",
                    "sub_cort_gm",
                    "wm",
                    "csf",
                    "pathologic_tissue",
                ),
                dim=0,
                keepdim=True,
                update_meta=True,
                list_output=False,
            )
        )
        # Merge the two gm masks back together.
        tfs.append(
            monai.transforms.ConcatItemsd(
                keys=("fivett_cort_gm", "fivett_sub_cort_gm"),
                name="gm_mask",
                dim=0,
            )
        )
        tfs.append(
            monai.transforms.Lambdad(
                keys=("gm_mask",),
                func=lambda m: torch.amax(m, dim=0, keepdim=True),
                overwrite=True,
            )
        )
        # Rename wm and csf masks.
        tfs.append(
            monai.transforms.CopyItemsd(
                keys=("fivett_wm", "fivett_csf"),
                names=("wm_mask", "csf_mask"),
            )
        )
        # Delete leftover tissue mask items.
        tfs.append(
            monai.transforms.DeleteItemsd(
                keys=(
                    "fivett_wm",
                    "fivett_csf",
                    "fivett_cort_gm",
                    "fivett_sub_cort_gm",
                    "fivett_pathologic_tissue",
                ),
            )
        )

        # Handle patch sampling mask.
        tfs.append(
            monai.transforms.CopyItemsd(
                keys=sample_mask_key,
                names="sampling_mask",
            )
        )
        # Dilate the sampling mask, if requested.
        if mask_dilate_radius != 0:
            # Dilate the lr mask to allow for uniform sampling of the patch centers.
            tfs.append(
                pitn.transforms.BinaryDilated(
                    ["sampling_mask"],
                    footprint=skimage.morphology.ball(mask_dilate_radius),
                    write_to_keys=["sampling_mask"],
                )
            )
        # Apply distance transform to sampling mask, to focus sampling on the more
        # concentrated regions.
        def distance_transform_mask(m: torch.Tensor):
            m_np = m.cpu().numpy()[0]
            dt = scipy.ndimage.distance_transform_edt(m_np)
            return (
                torch.from_numpy(dt)
                .to(dtype=torch.float32, device=m.device)
                .expand_as(m)
            )

        tfs.append(
            monai.transforms.Lambdad(
                "sampling_mask",
                distance_transform_mask,
            )
        )
        # Rescale the sampling mask to add up to 1.0
        tfs.append(
            monai.transforms.Lambdad(
                "sampling_mask", lambda m: m / torch.sum(m, (1, 2, 3), keepdim=True)
            )
        )

        tfs.append(
            monai.transforms.Lambdad(
                keys="fodf", func=_extract_affine, overwrite="affine_vox2world"
            )
        )
        tfs.append(
            monai.transforms.Lambdad(
                keys="affine_vox2world",
                func=monai.data.utils.affine_to_spacing,
                overwrite="vox_size",
            )
        )

        tfs.append(
            monai.transforms.CastToTyped(
                ("affine_vox2world", "vox_size", "bval", "bvec"), dtype=torch.float32
            )
        )
        return monai.transforms.Compose(tfs)


class _CallablePromisesList(collections.UserList):
    """Utility class that calls a callable when using indexing/using __getitem__.

    Used for lazily accessing items in a (potentially large) list of (potentially
    large) objects/containers.
    """

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            ret = list()
            for i in idx:
                ret.append(self.data[i]() if callable(self.data[i]) else self.data[i])
        else:
            ret = self.data[idx]() if callable(self.data[idx]) else self.data[idx]
        return ret


class _EfficientRandWeightedCropd(monai.transforms.RandWeightedCropd):
    """Efficient replacement for monai RandWeightedCropd.

    Functionality should be equivalent, while computation time should scale linearly
    with the number of samples requested (rather than quadratically). Memory
    requirements should be equal or better.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._spatial_size = self.cropper.spatial_size

        self._num_output_samples = self.cropper.num_samples
        del self.cropper
        self.cropper = monai.transforms.RandWeightedCrop(
            self._spatial_size, num_samples=1
        )

    @staticmethod
    def _crop_promise(
        data,
        cropper,
        keys,
        weight_map,
        cropper_random_state,
        nontransform_addon_data_dict,
    ):
        ret_dict = dict()
        # Initialize the random state that should correspond to this given selection.
        cropper.set_random_state(cropper_random_state)
        for key in keys:
            ret_dict[key] = list(
                cropper(data[key], weight_map=weight_map, randomize=False)
            )[0]
        return {**nontransform_addon_data_dict, **ret_dict}

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> _CallablePromisesList:
        ret = _CallablePromisesList()
        nontransformed_data = dict()
        # deep copy all the unmodified data
        for key in set(data.keys()).difference(set(self.keys)):
            nontransformed_data[key] = copy.deepcopy(data[key])

        self.randomize(weight_map=data[self.w_key])
        keys_to_sample = list(self.key_iterator(data))
        # Create a new callable "promise" for each output sample, with a designated
        # random state for the cropper object.
        for _ in range(self._num_output_samples):
            # Set a new random state for the cropper.
            self.cropper.randomize(weight_map=data[self.w_key])
            cropper_random_state = copy.deepcopy(self.cropper.R)
            promise_callable = partial(
                self._crop_promise,
                data=data,
                cropper=copy.copy(self.cropper),
                keys=keys_to_sample,
                weight_map=data[self.w_key],
                nontransform_addon_data_dict=nontransformed_data,
                cropper_random_state=cropper_random_state,
            )
            ret.append(promise_callable)
        # Return lazily-computed cropped samples.
        return ret


def _get_extent_world(x_pre_img: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    """Calculates the world coordinates that enumerate each voxel in the FOV.

    Parameters
    ----------
    x_pre_img : torch.Tensor
        Tensor with its last N-1 dimensions as the spatial extent in voxels.
    affine : torch.Tensor
        Affine, homogenious matrix that defines the voxel -> world coordinate transform.

    Returns
    -------
    torch.Tensor
        Tensor of world coordinates that exhaustively span all spatial dimensions.
    """
    spatial_shape = tuple(x_pre_img.shape[1:])
    vox_extent = [torch.arange(0, dim).to(affine) for dim in spatial_shape]
    vox_grid = torch.stack(torch.meshgrid(*vox_extent, indexing="ij"), dim=-1).reshape(
        -1, len(spatial_shape)
    )
    # Calculate world coordinates of the vox extent.
    world_extent = (affine[:3, :3] @ vox_grid.T) + (affine[:3, 3:4])
    # Reshape to be 3 x X x Y x Z, to match the shape found in the patch dataset.
    world_extent = world_extent.reshape(len(spatial_shape), *spatial_shape)
    # world_extent = world_extent.T
    # world_extent = world_extent.reshape(*spatial_shape, len(spatial_shape))
    return world_extent


def _random_iso_center_scale_affine(
    src_affine: torch.Tensor,
    src_spatial_sample: torch.Tensor,
    scale_low: float,
    scale_high: float,
    n_delta_buffer_scaled_vox: int = 1,
) -> torch.Tensor:
    # Randomly increase the spacing of src affine isotropically, while adding
    # translations to center the resamples into the src FoV.
    scale = np.random.uniform(scale_low, scale_high)
    scaling_affine = monai.transforms.utils.create_scale(
        3,
        [scale] * 3,
    )
    scaling_affine = torch.from_numpy(scaling_affine).to(src_affine)
    # Calculate the offset in target space voxels such that the target FoV will be
    # centered in the src FoV, and have 1 target voxel buffer between the LR fov and src
    # fov.
    src_spatial_shape = np.array(tuple(src_spatial_sample.shape[1:]))
    src_spacing = monai.data.utils.affine_to_spacing(src_affine, r=3).cpu().numpy()
    src_spatial_extent = src_spacing * src_spatial_shape
    target_n_vox_in_src_fov = src_spatial_extent / (scale * src_spacing)
    fov_delta = target_n_vox_in_src_fov - np.floor(target_n_vox_in_src_fov)
    evenly_distribute_fov_delta = fov_delta / 2
    # Add 1 scaled voxel between the inner (scaled) fov and the outer (src) fov, while
    # also keeping the delta spacing to keep the scaled fov centered wrt the src fov.
    evenly_distribute_fov_delta = (
        evenly_distribute_fov_delta + n_delta_buffer_scaled_vox
    )
    translate_aff = torch.eye(4).to(scaling_affine)
    translate_aff[:-1, -1] = torch.from_numpy(evenly_distribute_fov_delta).to(
        translate_aff
    )

    # Delta is in target space voxels, so we need to scale first, then translate.
    target_affine = src_affine @ (translate_aff @ scaling_affine)
    return target_affine


class CropFRInsideFoVNLRVox(monai.transforms.MapTransform):

    backend = monai.transforms.Crop.backend
    SPATIAL_COORD_PRECISION = 5

    def __init__(
        self,
        keys,
        affine_fr_vox2world_key,
        affine_lr_patch_vox2world_key,
        lr_vox_extent_fov_im_key,
        fr_vox_extent_fov_im_key,
        n_lr_vox_fr_interior_buffer=1,
        allow_missing_keys=False,
    ):
        monai.transforms.MapTransform.__init__(self, keys, allow_missing_keys)
        self.cropper = monai.transforms.Crop()
        self.n_lr_vox_fr_interior_buffer = n_lr_vox_fr_interior_buffer
        self.affine_fr_vox2world_key = affine_fr_vox2world_key
        self.affine_lr_patch_vox2world_key = affine_lr_patch_vox2world_key
        self.lr_vox_extent_fov_im_key = lr_vox_extent_fov_im_key
        self.fr_vox_extent_fov_im_key = fr_vox_extent_fov_im_key

    def __call__(self, data) -> dict:
        d = dict(data)
        lr_fov_vox = tuple(d[self.lr_vox_extent_fov_im_key].shape[-3:])
        fr_fov_vox = tuple(d[self.fr_vox_extent_fov_im_key].shape[-3:])
        lr_affine_vox2world = torch.as_tensor(d[self.affine_lr_patch_vox2world_key])
        fr_affine_vox2world = torch.as_tensor(d[self.affine_fr_vox2world_key])
        shared_dtype = torch.result_type(lr_affine_vox2world, fr_affine_vox2world)
        lr_affine_vox2world = lr_affine_vox2world.to(shared_dtype)
        fr_affine_vox2world = fr_affine_vox2world.to(shared_dtype)

        # Transform the desired LR FoV bounds into FR voxel coordinates, then "shrink"
        # the FR bbox to fit within those bounds.
        # Map the LR fov bounds to FR vox coordinates.
        affine_lr_vox2fr_vox = (
            torch.linalg.inv(fr_affine_vox2world) @ lr_affine_vox2world
        )
        # Account for any floating point errors that may cause the bottom row in the
        # homogeneous matrix to be invalid.
        affine_lr_vox2fr_vox[-1] = torch.round(torch.abs(affine_lr_vox2fr_vox[-1]))

        # Reduce the LR fov by the desired buffer size, and find the lower corner of
        # the (reduced) bounding box.
        lr_fov_vox_low = (0,) * len(lr_fov_vox)
        lr_fov_vox_low = affine_lr_vox2fr_vox.new_tensor((0,) * len(lr_fov_vox))
        lr_fov_vox_low = lr_fov_vox_low + self.n_lr_vox_fr_interior_buffer
        lr_fov_vox_high = affine_lr_vox2fr_vox.new_tensor(lr_fov_vox)
        lr_fov_vox_high = lr_fov_vox_high - self.n_lr_vox_fr_interior_buffer

        # Calculate the bbox coordinates of the reduced LR fov in FR voxels.
        bbox_lr_fov_in_fr_vox = pitn.affine.transform_coords(
            torch.stack([lr_fov_vox_low, lr_fov_vox_high], dim=0),
            affine_lr_vox2fr_vox,
        ).round(decimals=self.SPATIAL_COORD_PRECISION)
        # To find the desired slices that make the FR fov be contained within the LR fov,
        # round the bbox surrounding the LR fov, in FR voxels, to the next "interior" FR
        # voxel. Also, limit the slices to be within the FR fov.
        fr_fov_vox_low = (0,) * len(fr_fov_vox)
        fr_fov_vox_low = fr_affine_vox2world.new_tensor((0,) * len(fr_fov_vox))
        fr_fov_vox_high = fr_affine_vox2world.new_tensor(fr_fov_vox)
        new_fr_bbox_inside_lr_buffer = (
            torch.maximum(
                bbox_lr_fov_in_fr_vox[0].ceil().to(torch.int), fr_fov_vox_low
            ),
            torch.minimum(
                bbox_lr_fov_in_fr_vox[1].floor().to(torch.int), fr_fov_vox_high
            ),
        )

        fr_slices = self.cropper.compute_slices(
            roi_start=new_fr_bbox_inside_lr_buffer[0],
            roi_end=new_fr_bbox_inside_lr_buffer[1],
        )

        for key in self.key_iterator(d):
            d[key] = self.cropper(d[key], fr_slices)  # type: ignore
        return d


class ContainedFoVSpatialResampled(monai.transforms.SpatialResampled):
    def __call__(
        self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None
    ) -> dict[Hashable, torch.Tensor]:
        """
        Args:
            data: a dictionary containing the tensor-like data to be processed. The ``keys`` specified
                in this dictionary must be tensor like arrays that are channel first and have at most
                three spatial dimensions
            lazy: a flag to indicate whether this transform should execute lazily or not
                during this call. Setting this to False or True overrides the ``lazy`` flag set
                during initialization for this call. Defaults to None.

        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """

        n_vox_edge_buffer = 1
        lazy_ = self.lazy if lazy is None else lazy
        d: dict = dict(data)
        # Add a quick-and-dirty cache in case all key-ed volumes are the same shape,
        # and all have the same source affines.
        sp_srcaff_dstaff2dst_sp = list()
        for key, mode, padding_mode, align_corners, dtype, dst_key in self.key_iterator(
            d,
            self.mode,
            self.padding_mode,
            self.align_corners,
            self.dtype,
            self.dst_keys,
        ):
            # Override the spatial size such that the interpolation does not
            # need to be padded.
            src_affine = torch.as_tensor(d[key].affine)
            dst_affine = torch.as_tensor(d[dst_key])
            shared_dtype = torch.result_type(src_affine, dst_affine)
            src_affine = src_affine.to(shared_dtype)
            dst_affine = dst_affine.to(shared_dtype)
            src_vox_upper_bound = torch.as_tensor(d[key].shape[1:])[None, :]
            if (
                sp_srcaff_dstaff2dst_sp
                and torch.isclose(src_vox_upper_bound, sp_srcaff_dstaff2dst_sp[0]).all()
                and torch.isclose(src_affine, sp_srcaff_dstaff2dst_sp[1]).all()
                and torch.isclose(dst_affine, sp_srcaff_dstaff2dst_sp[2]).all()
            ):
                dst_vox_spatial_size_rounded = sp_srcaff_dstaff2dst_sp[-1]
            else:
                affine_src_vox2dst_vox = torch.linalg.inv(dst_affine) @ src_affine
                # Add vox buffer to the higher edge of the fov.
                dst_vox_upper_bound = (
                    pitn.affine.transform_coords(
                        src_vox_upper_bound, affine_src_vox2dst_vox
                    )
                    - n_vox_edge_buffer
                )
                dst_vox_spatial_size_rounded = tuple(
                    dst_vox_upper_bound.floor().int().cpu().flatten().tolist()
                )
                if not sp_srcaff_dstaff2dst_sp:
                    sp_srcaff_dstaff2dst_sp.append(src_vox_upper_bound)
                    sp_srcaff_dstaff2dst_sp.append(src_affine)
                    sp_srcaff_dstaff2dst_sp.append(dst_affine)
                    sp_srcaff_dstaff2dst_sp.append(dst_vox_spatial_size_rounded)

            d[key] = self.sp_transform(
                img=d[key],
                dst_affine=d[dst_key],
                spatial_size=dst_vox_spatial_size_rounded,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
                dtype=dtype,
                lazy=lazy_,
            )
        return d


def _crop_lr_inside_fr_spatial_surface(
    d: dict,
    keys_to_crop: tuple,
    output_keys: tuple,
    affine_fr_vox2world_key,
    lr_vox_extent_fov_im_key,
    fr_vox_extent_fov_im_key,
    affine_lr_patch_vox2world_key,
    im_cropper,
):
    SPATIAL_COORD_PRECISION = 5
    lr_fov_vox = tuple(d[lr_vox_extent_fov_im_key].shape[-3:])
    fr_fov_vox = tuple(d[fr_vox_extent_fov_im_key].shape[-3:])
    lr_affine_vox2world = torch.as_tensor(d[affine_lr_patch_vox2world_key])
    fr_affine_vox2world = torch.as_tensor(d[affine_fr_vox2world_key])
    shared_dtype = torch.result_type(lr_affine_vox2world, fr_affine_vox2world)
    lr_affine_vox2world = lr_affine_vox2world.to(shared_dtype)
    fr_affine_vox2world = fr_affine_vox2world.to(shared_dtype)

    # Get the bounding box of the FR fov in LR voxel coordinates. If the FR bbox has
    # voxel coordinates > 0 or < the LR fov shape, then the LR fov is out of bounds
    # w.r.t. the FR spatial extent, and must be cropped to be entirely contained
    # within the FR fov.
    # Map the FR fov bounds to LR vox coordinates.
    affine_fr_vox2lr_vox = torch.linalg.inv(lr_affine_vox2world) @ fr_affine_vox2world
    # Account for any floating point errors that may cause the bottom row in the
    # homogeneous matrix to be invalid.
    affine_fr_vox2lr_vox[-1] = torch.round(torch.abs(affine_fr_vox2lr_vox[-1]))

    # Find the lower corner of the bounding box.
    fr_fov_vox_low = (0,) * len(fr_fov_vox)
    fr_fov_vox_low = affine_fr_vox2lr_vox.new_tensor((0,) * len(fr_fov_vox))
    fr_fov_vox_high = affine_fr_vox2lr_vox.new_tensor(fr_fov_vox)

    # Calculate the bbox coordinates of the FR fov in LR voxels.
    bbox_fr_fov_in_lr_vox = pitn.affine.transform_coords(
        torch.stack([fr_fov_vox_low, fr_fov_vox_high], dim=0),
        affine_fr_vox2lr_vox,
    ).round(decimals=SPATIAL_COORD_PRECISION)
    # To find the desired slices that make the LR fov be contained within the FR fov,
    # round the bbox surrounding the FR fov, in LR voxels, to the next "interior" LR
    # voxel. Also, limit the slices to be within the LR fov.
    lr_fov_vox_low = (0,) * len(lr_fov_vox)
    lr_fov_vox_low = lr_affine_vox2world.new_tensor((0,) * len(lr_fov_vox))
    lr_fov_vox_high = lr_affine_vox2world.new_tensor(lr_fov_vox)
    bbox_fr_fov_in_lr_vox_rounded_inside_fr_fov = (
        torch.maximum(bbox_fr_fov_in_lr_vox[0].ceil().to(torch.int), lr_fov_vox_low),
        torch.minimum(bbox_fr_fov_in_lr_vox[1].floor().to(torch.int), lr_fov_vox_high),
    )

    lr_slices = monai.transforms.Crop.compute_slices(
        roi_start=bbox_fr_fov_in_lr_vox_rounded_inside_fr_fov[0],
        roi_end=bbox_fr_fov_in_lr_vox_rounded_inside_fr_fov[1],
    )

    if isinstance(keys_to_crop, str):
        keys_to_crop = (keys_to_crop,)
    if isinstance(output_keys, str):
        output_keys = (output_keys,)
    assert len(keys_to_crop) == len(output_keys)
    for k_in, k_out in zip(keys_to_crop, output_keys):
        im = d[k_in]
        im_patch = im_cropper(im, lr_slices)
        d[k_out] = im_patch

    return d


def prefilter_gaussian_blur(
    vol: torch.Tensor,
    src_affine: torch.Tensor,
    target_affine: torch.Tensor,
    sigma_scale_coeff: float = 2.5,
    sigma_truncate=4.0,
):

    v = vol.detach().cpu().numpy()
    src_size = monai.data.utils.affine_to_spacing(src_affine.detach().cpu())
    target_size = monai.data.utils.affine_to_spacing(target_affine.detach().cpu())
    scale_ratio_high_to_low = (torch.mean(src_size) / torch.mean(target_size)).item()
    sigma = 1 / (sigma_scale_coeff * scale_ratio_high_to_low)
    if len(v.shape) == 4:
        sigma = (0, sigma, sigma, sigma)
    else:
        sigma = (sigma,) * 3
    v_filter = scipy.ndimage.gaussian_filter(
        v, sigma=sigma, order=0, mode="nearest", truncate=sigma_truncate
    )

    vol_blur = torch.from_numpy(v_filter).to(vol)

    return vol_blur


def pad_list_data_collate_update_affines_to_tensor(
    d, meta_tensor_key_write_aff_key_pairs: tuple[tuple[str, str]] = tuple(), **kwargs
):
    # The datasets will usually produce volumes of different shapes due to the possible
    # random re-sampling, so the batch must be padded, and the padded masks must be
    # used to calculate the loss.
    ret = monai.data.utils.pad_list_data_collate(d, **kwargs)
    for meta_tensor_key, write_affine_key in meta_tensor_key_write_aff_key_pairs:
        ret[write_affine_key] = ret[meta_tensor_key].affine.to(torch.float32)
    return {
        k: monai.utils.convert_to_tensor(v, track_meta=False)
        if isinstance(v, monai.data.MetaObj)
        else v
        for k, v in ret.items()
    }


class HCPfODFINRPatchDataset(monai.data.PatchDataset):

    _SAMPLE_KEYS = (
        "subj_id",
        "fodf",
        "brain_mask",
        "fivett",
        "wm_mask",
        "gm_mask",
        "csf_mask",
        "vox_size",
        "affine_vox2world",
        "lr_dwi",
        "lr_brain_mask",
        "lr_vox_size",
        "affine_lr_vox2world",
        "lr_bval",
        "lr_bvec",
    )

    def __init__(
        self,
        base_dataset,
        patch_func,
        samples_per_image: int = 1,
        transform=None,
    ):

        self.base_dataset = base_dataset
        super().__init__(
            self.base_dataset,
            patch_func=patch_func,
            samples_per_image=samples_per_image,
            transform=transform,
        )

    @staticmethod
    def default_patch_func(
        keys=("dwi", "fodf", "brain_mask", "fivett", "wm_mask", "gm_mask", "csf_mask"),
        w_key="sampling_mask",
        **sample_tf_kwargs,
    ):
        return _EfficientRandWeightedCropd(keys=keys, w_key=w_key, **sample_tf_kwargs)

    @staticmethod
    def default_feature_tf(
        baseline_iso_scale_factor_lr_spacing_mm_low_high: float,
        augmentation_prob: float,
        augment_iso_scale_factor_lr_spacing_mm_low_high: Tuple[float] = (1.0, 1.0),
        scale_prefilter_kwargs: dict = dict(),
        augment_spatial_resample_kwargs: dict = dict(),
        augment_rand_rician_noise_kwargs: dict = dict(),
        augment_rand_rotate_90_kwargs: dict = dict(),
        augment_rand_flip_kwargs: dict = dict(),
        keep_metatensor_output: bool = False,
    ):

        VOL_KEYS = (
            "dwi",
            "fodf",
            "brain_mask",
            "fivett",
            "wm_mask",
            "gm_mask",
            "csf_mask",
        )
        LR_VOL_KEYS = (
            "lr_dwi",
            "lr_brain_mask",
            "lr_fivett",
            "lr_wm_mask",
            "lr_gm_mask",
            "lr_csf_mask",
        )

        # Transforms for extracting features for the network.
        feat_tfs = list()

        # Extract the patch affine matrix.
        feat_tfs.append(
            monai.transforms.adaptor(
                _extract_affine, outputs="affine_vox2world", inputs={"dwi": "src_vol"}
            )
        )
        feat_tfs.append(
            monai.transforms.CopyItemsd(
                keys=["affine_vox2world"],
                times=1,
                names=["target_affine_lr_vox2world"],
            )
        )
        # Make copies of the patches that will be downsampled.
        feat_tfs.append(
            monai.transforms.CopyItemsd(
                keys=(
                    "dwi",
                    "brain_mask",
                    "fivett",
                    "wm_mask",
                    "gm_mask",
                    "csf_mask",
                    "bval",
                    "bvec",
                ),
                times=1,
                names=(
                    "lr_dwi",
                    "lr_brain_mask",
                    "lr_fivett",
                    "lr_wm_mask",
                    "lr_gm_mask",
                    "lr_csf_mask",
                    "lr_bval",
                    "lr_bvec",
                ),
            )
        )

        #### Augmentation transforms, subject to `augmentation prob`.
        # Random iso-scaling
        low, high = augment_iso_scale_factor_lr_spacing_mm_low_high
        # Randomly choose a scale, apply to an affine matrix, then resample from that.
        scale_shift_aff_tf = monai.transforms.adaptor(
            partial(
                _random_iso_center_scale_affine,
                scale_low=low,
                scale_high=high,
                n_delta_buffer_scaled_vox=1,
            ),
            inputs={"affine_vox2world": "src_affine", "dwi": "src_spatial_sample"},
            outputs="target_affine_lr_vox2world",
        )
        # Prefilter/blur DWI with a small gaussian filter. Sigma is chosen by a slightly
        # tweaked rule-of-thumb by default:
        # sigma = 1 / 2.5 * (high_res_spacing / low_res_spacing)
        # This function is common for both the augmentation path, and the "baseline"
        # path.
        prefilter_fn = partial(
            prefilter_gaussian_blur,
            **{
                **dict(sigma_scale_coeff=2.5, sigma_truncate=4.0),
                **scale_prefilter_kwargs,
            },
        )
        # Blur, then downsample the full-res DWI patch to create the LR patch
        prefilter_tf = monai.transforms.adaptor(
            prefilter_fn,
            outputs="lr_dwi",
            inputs={
                "lr_dwi": "vol",
                "affine_vox2world": "src_affine",
                "target_affine_lr_vox2world": "target_affine",
            },
        )
        # Then downscale with bilinear (for the dwi) or nearest neighbor (masks/labels)
        resample_lr_patch_tf = ContainedFoVSpatialResampled(
            **{
                **dict(
                    keys=(
                        "lr_dwi",
                        "lr_brain_mask",
                        "lr_fivett",
                        "lr_wm_mask",
                        "lr_gm_mask",
                        "lr_csf_mask",
                    ),
                    mode=["bilinear"] + (["nearest"] * 5),
                    padding_mode="zeros",
                    align_corners=True,
                    dst_keys="target_affine_lr_vox2world",
                ),
                **augment_spatial_resample_kwargs,
            }
        )

        # Crop the full-resolution patch such that it lies 1 LR voxel within the
        # LR spatial extent on all sides.
        fr_crop_inside_lr_buffer_fov_tf = CropFRInsideFoVNLRVox(
            keys=VOL_KEYS,
            n_lr_vox_fr_interior_buffer=1,
            affine_fr_vox2world_key="affine_vox2world",
            affine_lr_patch_vox2world_key="target_affine_lr_vox2world",
            fr_vox_extent_fov_im_key="dwi",
            lr_vox_extent_fov_im_key="lr_dwi",
        )

        # Random noise injection
        # Dipy constructs the std as `SNR = ref_signal / sigma`, with the ref being
        # the max value in the b0s. In HCP, this value is ~10,000 mm/s^2, so with an snr
        # of ~40, a reasonable amount of noise is added.
        s_0 = 10000.0
        target_snr = 41
        noise_tf = monai.transforms.RandRicianNoised(
            **{
                **dict(
                    keys="lr_dwi",
                    prob=0.4,
                    mean=0.0,
                    std=s_0 / target_snr,
                    relative=False,
                    channel_wise=False,
                    sample_std=True,
                ),
                **augment_rand_rician_noise_kwargs,
            }
        )
        # Apply random flips and rotations to both the lr patch and the fr patch.
        # Random 90 degree rotations
        # These won't produce a uniform distribution of final orientations, but that
        # probably isn't a big deal at this time.
        rotate_xy_tf = monai.transforms.RandRotate90d(
            keys=VOL_KEYS + LR_VOL_KEYS,
            max_k=3,
            spatial_axes=(0, 1),
            **augment_rand_rotate_90_kwargs,
        )
        rotate_yz_tf = monai.transforms.RandRotate90d(
            keys=VOL_KEYS + LR_VOL_KEYS,
            max_k=3,
            spatial_axes=(1, 2),
            **augment_rand_rotate_90_kwargs,
        )
        # Random flips
        flip_x_tf = monai.transforms.RandFlipd(
            keys=VOL_KEYS + LR_VOL_KEYS, spatial_axis=0, **augment_rand_flip_kwargs
        )
        flip_y_tf = monai.transforms.RandFlipd(
            keys=VOL_KEYS + LR_VOL_KEYS, spatial_axis=1, **augment_rand_flip_kwargs
        )
        flip_z_tf = monai.transforms.RandFlipd(
            keys=VOL_KEYS + LR_VOL_KEYS, spatial_axis=2, **augment_rand_flip_kwargs
        )
        augment_tfs = [
            scale_shift_aff_tf,
            prefilter_tf,
            resample_lr_patch_tf,
            fr_crop_inside_lr_buffer_fov_tf,
            noise_tf,
            rotate_xy_tf,
            rotate_yz_tf,
            flip_x_tf,
            flip_y_tf,
            flip_z_tf,
        ]
        augment_tfs = monai.transforms.Compose(augment_tfs)

        #### Baseline transforms, which reuse some of the augment transforms.
        # The scaling is "random," but there's only one value that can be selected.
        baseline_scale_shift_aff_tf = monai.transforms.adaptor(
            partial(
                _random_iso_center_scale_affine,
                scale_low=baseline_iso_scale_factor_lr_spacing_mm_low_high,
                scale_high=baseline_iso_scale_factor_lr_spacing_mm_low_high,
            ),
            inputs={"affine_vox2world": "src_affine", "dwi": "src_spatial_sample"},
            outputs="target_affine_lr_vox2world",
        )
        # Remove augmentation kwargs from the baseline downsampler.
        baseline_resample_lr_patch_tf = monai.transforms.SpatialResampled(
            keys=(
                "lr_dwi",
                "lr_brain_mask",
                "lr_fivett",
                "lr_wm_mask",
                "lr_gm_mask",
                "lr_csf_mask",
            ),
            mode=["bilinear"] + (["nearest"] * 5),
            padding_mode="zeros",
            align_corners=True,
            dst_keys="target_affine_lr_vox2world",
        )
        baseline_non_augment_tfs = [
            baseline_scale_shift_aff_tf,
            copy.deepcopy(prefilter_tf),
            baseline_resample_lr_patch_tf,
            copy.deepcopy(fr_crop_inside_lr_buffer_fov_tf),
        ]
        baseline_non_augment_tfs = monai.transforms.Compose(baseline_non_augment_tfs)

        # Create a transformation branch conditioned on the chance of whether or not
        # to use augmentation.
        augment_branch_tf = monai.transforms.OneOf(
            transforms=(augment_tfs, baseline_non_augment_tfs),
            weights=(augmentation_prob, 1.0 - augmentation_prob),
        )
        feat_tfs.append(augment_branch_tf)

        # Remove the full resolution DWI as it is not used, for now.
        feat_tfs.append(
            monai.transforms.DeleteItemsd(
                keys=(
                    "dwi",
                    "bval",
                    "bvec",
                    "target_affine_lr_vox2world",
                )
            )
        )
        curr_vol_keys = tuple(set(VOL_KEYS) - {"dwi"})

        # Extract the new LR patch affine matrix.
        feat_tfs.append(
            monai.transforms.adaptor(
                _extract_affine,
                outputs="affine_lr_vox2world",
                inputs={"lr_dwi": "src_vol"},
            )
        )
        feat_tfs.append(
            monai.transforms.adaptor(
                monai.data.utils.affine_to_spacing,
                outputs="lr_vox_size",
                inputs={"affine_lr_vox2world": "affine"},
            )
        )
        # Similarly for the FR patch.
        feat_tfs.append(
            monai.transforms.adaptor(
                _extract_affine, outputs="affine_vox2world", inputs={"fodf": "src_vol"}
            )
        )
        feat_tfs.append(
            monai.transforms.adaptor(
                monai.data.utils.affine_to_spacing,
                outputs="vox_size",
                inputs={"affine_vox2world": "affine"},
            )
        )

        curr_vol_keys = tuple(set(curr_vol_keys) - {"fivett"})
        curr_lr_vol_keys = tuple(
            set(LR_VOL_KEYS) - {"lr_fivett", "lr_csf_mask", "lr_wm_mask", "lr_gm_mask"}
        )
        # Remove unnecessary items from the data dict.
        # Sub-select keys to free memory.
        feat_tfs.append(
            monai.transforms.SelectItemsd(
                (
                    "subj_id",
                    "affine_vox2world",
                    "affine_lr_vox2world",
                    "vox_size",
                    "lr_vox_size",
                    "lr_bval",
                    "lr_bvec",
                )
                + curr_vol_keys
                + curr_lr_vol_keys,
            )
        )

        # Convert all MetaTensors to regular Tensors.
        all_tensor_keys = (
            (
                "affine_vox2world",
                "affine_lr_vox2world",
                "vox_size",
                "lr_vox_size",
                "lr_bval",
                "lr_bvec",
            )
            + curr_vol_keys
            + curr_lr_vol_keys
        )
        feat_tfs.append(
            monai.transforms.CastToTyped(
                (
                    "affine_vox2world",
                    "affine_lr_vox2world",
                    "vox_size",
                    "lr_vox_size",
                    "fodf",
                    "lr_dwi",
                    "lr_bval",
                    "lr_bvec",
                )
                + (
                    "brain_mask",
                    "wm_mask",
                    "gm_mask",
                    "csf_mask",
                    "lr_brain_mask",
                ),
                dtype=[torch.float32] * 8 + [torch.bool] * 5,
            ),
        )
        if not keep_metatensor_output:
            feat_tfs.append(
                monai.transforms.FromMetaTensord(
                    curr_vol_keys + curr_lr_vol_keys, data_type="tensor"
                )
            )
        feat_tfs.append(monai.transforms.SelectItemsd(all_tensor_keys + ("subj_id",)))

        return monai.transforms.Compose(feat_tfs)


class HCPfODFINRWholeBrainDataset(monai.data.Dataset):

    _SAMPLE_KEYS = (
        "subj_id",
        "bval",
        "bvec",
        "fodf",
        "brain_mask",
        "fivett",
        "wm_mask",
        "gm_mask",
        "csf_mask",
        "vox_size",
        "affine_vox2world",
        "lr_dwi",
        "lr_brain_mask",
        "lr_vox_size",
        "affine_lr_vox2world",
        "bval",
        "bvec",
    )

    def __init__(
        self,
        base_dataset,
        transform=None,
    ):
        self.base_dataset = base_dataset
        super().__init__(
            self.base_dataset,
            transform=transform,
        )

    @staticmethod
    def default_vol_tf(
        baseline_iso_scale_factor_lr_spacing_mm_low_high: float,
        scale_prefilter_kwargs: dict = dict(),
        keep_metatensor_output: bool = False,
    ):

        VOL_KEYS = (
            "dwi",
            "fodf",
            "brain_mask",
            "fivett",
            "wm_mask",
            "gm_mask",
            "csf_mask",
        )
        LR_VOL_KEYS = (
            "lr_dwi",
            "lr_brain_mask",
            "lr_fivett",
            "lr_wm_mask",
            "lr_gm_mask",
            "lr_csf_mask",
        )

        # Transforms for whole-brain volumes as input to the network.
        feat_tfs = list()

        # Pad the full-resolution volumes, because various transformations will require
        # cropping, and we can safely 0-pad here.
        feat_tfs.append(
            monai.transforms.BorderPadd(
                keys=VOL_KEYS, spatial_border=4, mode="constant", value=0
            )
        )

        # Extract the vol affine matrix.
        feat_tfs.append(
            monai.transforms.Lambdad(
                keys="dwi", func=_extract_affine, overwrite="affine_vox2world"
            )
        )
        # feat_tfs.append(
        #     monai.transforms.adaptor(
        #         _extract_affine, outputs="affine_vox2world", inputs={"dwi": "src_vol"}
        #     )
        # )
        feat_tfs.append(
            monai.transforms.CopyItemsd(
                keys=["affine_vox2world"],
                times=1,
                names=["target_affine_lr_vox2world"],
            )
        )
        # Make copies of the vols that will be downsampled.
        feat_tfs.append(
            monai.transforms.CopyItemsd(
                keys=(
                    "dwi",
                    "brain_mask",
                    "fivett",
                    "wm_mask",
                    "gm_mask",
                    "csf_mask",
                    "bval",
                    "bvec",
                ),
                times=1,
                names=(
                    "lr_dwi",
                    "lr_brain_mask",
                    "lr_fivett",
                    "lr_wm_mask",
                    "lr_gm_mask",
                    "lr_csf_mask",
                    "lr_bval",
                    "lr_bvec",
                ),
            )
        )

        # Scale and center the LR target affine matrix for later resampling.
        # The scaling is "random," but there's only one value that can be selected.
        feat_tfs.append(
            monai.transforms.adaptor(
                partial(
                    _random_iso_center_scale_affine,
                    scale_low=baseline_iso_scale_factor_lr_spacing_mm_low_high,
                    scale_high=baseline_iso_scale_factor_lr_spacing_mm_low_high,
                ),
                inputs={"affine_vox2world": "src_affine", "dwi": "src_spatial_sample"},
                outputs="target_affine_lr_vox2world",
            )
        )

        # Prefilter/blur DWI with a small gaussian filter. Sigma is chosen by a slightly
        # tweaked rule-of-thumb by default:
        # sigma = 1 / 2.5 * (high_res_spacing / low_res_spacing)
        # NOTE: The given values should match those used in training!
        prefilter_fn = partial(
            prefilter_gaussian_blur,
            **{
                **dict(sigma_scale_coeff=2.5, sigma_truncate=4.0),
                **scale_prefilter_kwargs,
            },
        )
        # Blur, then downsample the full-res DWI vol to create the LR vol
        feat_tfs.append(
            monai.transforms.adaptor(
                prefilter_fn,
                outputs="lr_dwi",
                inputs={
                    "lr_dwi": "vol",
                    "affine_vox2world": "src_affine",
                    "target_affine_lr_vox2world": "target_affine",
                },
            )
        )

        feat_tfs.append(
            monai.transforms.SpatialResampled(
                keys=(
                    "lr_dwi",
                    "lr_brain_mask",
                    "lr_fivett",
                    "lr_wm_mask",
                    "lr_gm_mask",
                    "lr_csf_mask",
                ),
                mode=["bilinear"] + (["nearest"] * 5),
                padding_mode="zeros",
                align_corners=True,
                dst_keys="target_affine_lr_vox2world",
            )
        )

        # Crop the full-resolution patch such that it lies 1 LR voxel within the
        # LR spatial extent on all sides.
        feat_tfs.append(
            CropFRInsideFoVNLRVox(
                keys=VOL_KEYS,
                n_lr_vox_fr_interior_buffer=1,
                affine_fr_vox2world_key="affine_vox2world",
                affine_lr_patch_vox2world_key="target_affine_lr_vox2world",
                fr_vox_extent_fov_im_key="dwi",
                lr_vox_extent_fov_im_key="lr_dwi",
            )
        )

        # Remove the full resolution DWI as it is not used, for now.
        feat_tfs.append(
            monai.transforms.DeleteItemsd(
                keys=(
                    "dwi",
                    "target_affine_lr_vox2world",
                )
            )
        )
        curr_vol_keys = tuple(set(VOL_KEYS) - {"dwi"})

        # Extract the new LR vol affine matrix.
        feat_tfs.append(
            monai.transforms.adaptor(
                _extract_affine,
                outputs="affine_lr_vox2world",
                inputs={"lr_dwi": "src_vol"},
            )
        )
        feat_tfs.append(
            monai.transforms.adaptor(
                monai.data.utils.affine_to_spacing,
                outputs="lr_vox_size",
                inputs={"affine_lr_vox2world": "affine"},
            )
        )
        # Similarly for the FR vol.
        feat_tfs.append(
            monai.transforms.adaptor(
                _extract_affine, outputs="affine_vox2world", inputs={"fodf": "src_vol"}
            )
        )
        feat_tfs.append(
            monai.transforms.adaptor(
                monai.data.utils.affine_to_spacing,
                outputs="vox_size",
                inputs={"affine_vox2world": "affine"},
            )
        )

        curr_vol_keys = tuple(set(curr_vol_keys) - {"fivett"})
        curr_lr_vol_keys = tuple(
            set(LR_VOL_KEYS) - {"lr_fivett", "lr_csf_mask", "lr_wm_mask", "lr_gm_mask"}
        )
        # Remove unnecessary items from the data dict.
        # Sub-select keys to free memory.
        select_k_tf = monai.transforms.SelectItemsd(
            (
                "subj_id",
                "bval",
                "bvec",
                "affine_vox2world",
                "affine_lr_vox2world",
                "vox_size",
                "lr_vox_size",
                "lr_bval",
                "lr_bvec",
            )
            + curr_vol_keys
            + curr_lr_vol_keys,
        )
        feat_tfs.append(select_k_tf)

        # Convert all MetaTensors to regular Tensors.
        if not keep_metatensor_output:
            to_tensor_tf = monai.transforms.ToTensord(
                (
                    "bval",
                    "bvec",
                    "affine_vox2world",
                    "affine_lr_vox2world",
                    "vox_size",
                    "lr_vox_size",
                    "lr_bval",
                    "lr_bvec",
                )
                + curr_vol_keys
                + curr_lr_vol_keys,
                track_meta=False,
            )
            feat_tfs.append(to_tensor_tf)

        return monai.transforms.Compose(feat_tfs)
