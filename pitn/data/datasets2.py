# -*- coding: utf-8 -*-
import collections
import copy
import functools
import itertools
import math
from functools import partial
from pathlib import Path
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import einops
import monai
import nibabel as nib
import numpy as np
import skimage
import torch
from monai.config import DtypeLike, KeysCollection, SequenceStr
from monai.data.meta_obj import get_track_meta
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    convert_to_tensor,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
)

import pitn


class HCPfODFINRDataset(monai.data.Dataset):

    _SAMPLE_KEYS = (
        "subj_id",
        "lr_dwi",
        "lr_mask",
        "lr_bval",
        "lr_bvec",
        "lr_fodf",
        "lr_fivett",
        "lr_wm_mask",
        "lr_gm_mask",
        "lr_csf_mask",
        "affine_lr_vox2world",
        "fodf",
        "mask",
        "fivett",
        "wm_mask",
        "gm_mask",
        "csf_mask",
        "affine_vox2world",
    )

    def __init__(
        self,
        subj_ids,
        dwi_root_dir: Path,
        fodf_root_dir: Path,
        lr_dwi_root_dir: Path,
        lr_fodf_root_dir: Path,
        transform=None,
    ):
        self.subj_ids = list(subj_ids)
        self.dwi_root_dir = Path(dwi_root_dir).resolve()
        self.fodf_root_dir = Path(fodf_root_dir).resolve()
        self.lr_dwi_root_dir = Path(lr_dwi_root_dir).resolve()
        self.lr_fodf_root_dir = Path(lr_fodf_root_dir).resolve()
        data = list()
        for sid in self.subj_ids:
            hcp_data = self.get_hcp_subj_dict(sid, self.dwi_root_dir)
            fodf_data = self.get_fodf_subj_dict(sid, self.fodf_root_dir)
            lr_data = self.get_lr_hcp_subj_dict(sid, self.lr_dwi_root_dir)
            lr_fodf_data = self.get_fodf_subj_dict(sid, self.lr_fodf_root_dir)
            sid_data = dict(
                subj_id=hcp_data["subj_id"],
                lr_dwi=lr_data["dwi"],
                lr_fodf=lr_fodf_data["fodf"],
                lr_mask=lr_data["mask"],
                lr_fivett=lr_fodf_data["fivett"],
                lr_bval=lr_data["bval"],
                lr_bvec=lr_data["bvec"],
                fodf=fodf_data["fodf"],
                mask=fodf_data["mask"],
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
    def default_pre_sample_tf(mask_dilate_radius: int, skip_sample_mask=False):

        VOL_KEYS = (
            "lr_dwi",
            "lr_mask",
            "lr_fodf",
            "lr_fivett",
            "fodf",
            "mask",
            "fivett",
        )

        tfs = list()
        # Load images
        vol_reader = monai.data.NibabelReader(
            as_closest_canonical=True, dtype=np.float32
        )
        tfs.append(
            monai.transforms.LoadImaged(
                VOL_KEYS,
                reader=vol_reader,
                dtype=np.float32,
                meta_key_postfix="meta",
                ensure_channel_first=True,
                simple_keys=True,
            )
        )

        grad_file_reader = monai.transforms.Lambdad(
            ("lr_bval", "lr_bvec"), lambda f: np.loadtxt(str(f)), overwrite=True
        )
        tfs.append(grad_file_reader)

        # Data conversion
        tfs.append(
            monai.transforms.ToTensord(
                VOL_KEYS,
                track_meta=True,
            )
        )
        tfs.append(monai.transforms.ToTensord(("lr_bval", "lr_bvec"), track_meta=False))
        tfs.append(
            monai.transforms.CastToTyped(
                ("lr_mask", "lr_fivett", "mask", "fivett"), dtype=torch.uint8
            )
        )

        # Extract different tissue masks according to the 5tt designations.
        tfs.append(
            monai.transforms.SplitDimd(
                keys=("lr_fivett", "fivett"),
                output_postfixes=(
                    "cort_gm",
                    "sub_cort_gm",
                    "wm",
                    "csf",
                    "pathologic_tissue",
                ),
                dim=0,
                keepdim=False,
                update_meta=True,
                list_output=False,
            )
        )
        # Merge the two gm masks back together.
        tfs.append(
            monai.transforms.AddChanneld(
                keys=(
                    "lr_fivett_cort_gm",
                    "lr_fivett_sub_cort_gm",
                    "fivett_cort_gm",
                    "fivett_sub_cort_gm",
                ),
            )
        )
        tfs.append(
            monai.transforms.ConcatItemsd(
                keys=("lr_fivett_cort_gm", "lr_fivett_sub_cort_gm"),
                name="lr_gm_mask",
                dim=0,
            )
        )
        tfs.append(
            monai.transforms.ConcatItemsd(
                keys=("fivett_cort_gm", "fivett_sub_cort_gm"),
                name="gm_mask",
                dim=0,
            )
        )
        tfs.append(
            monai.transforms.Lambdad(
                keys=("lr_gm_mask", "gm_mask"),
                func=lambda m: torch.amax(m, dim=0, keepdim=False),
                overwrite=True,
            )
        )
        # Rename wm and csf masks.
        tfs.append(
            monai.transforms.CopyItemsd(
                keys=("lr_fivett_wm", "lr_fivett_csf", "fivett_wm", "fivett_csf"),
                names=("lr_wm_mask", "lr_csf_mask", "wm_mask", "csf_mask"),
            )
        )
        # Delete leftover tissue mask items.
        tfs.append(
            monai.transforms.DeleteItemsd(
                keys=(
                    "lr_fivett_wm",
                    "lr_fivett_csf",
                    "lr_fivett_pathologic_tissue",
                    "lr_fivett_cort_gm",
                    "lr_fivett_sub_cort_gm",
                    "fivett_wm",
                    "fivett_csf",
                    "fivett_cort_gm",
                    "fivett_sub_cort_gm",
                    "fivett_pathologic_tissue",
                ),
            )
        )
        tfs.append(
            monai.transforms.AddChanneld(
                keys=(
                    "lr_wm_mask",
                    "lr_gm_mask",
                    "lr_csf_mask",
                    "wm_mask",
                    "gm_mask",
                    "csf_mask",
                ),
            )
        )

        # Handle patch sampling mask.
        if not skip_sample_mask:
            # Dilate the lr mask to allow for uniform sampling of the patch centers.
            dilate_tf = pitn.transforms.BinaryDilated(
                ["lr_mask"],
                footprint=skimage.morphology.ball(mask_dilate_radius),
                write_to_keys=["lr_sampling_mask"],
            )
            rescale_tf = monai.transforms.Lambdad(
                "lr_sampling_mask", lambda m: m / torch.sum(m, (1, 2, 3), keepdim=True)
            )
            tfs.append(dilate_tf)
            tfs.append(rescale_tf)

        tfs.append(
            functools.partial(
                _extract_affine, src_vol_key="lr_dwi", write_key="affine_lr_vox2world"
            )
        )
        tfs.append(
            functools.partial(
                _extract_affine, src_vol_key="fodf", write_key="affine_vox2world"
            )
        )
        vox_size_tf = monai.transforms.adaptor(
            monai.data.utils.affine_to_spacing,
            outputs="vox_size",
            inputs={"affine_vox2world": "affine"},
        )
        tfs.append(vox_size_tf)
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


# Save the affine as its own field.
def _extract_affine(d: dict, src_vol_key, write_key: str):
    aff = d[src_vol_key].affine
    if torch.is_tensor(aff):
        d[write_key] = torch.clone(aff).to(torch.float32)
    else:
        d[write_key] = np.array(np.asarray(aff), copy=True, dtype=np.float32)
    return d


def _extract_lr_patch_info(lr_dwi, affine_lr_vox2world, patch_size: tuple):
    # Extract low-resolution input information
    patch_center_lr_vox = torch.clone(
        lr_dwi.meta["crop_center"].as_tensor().to(torch.int)
    )
    affine_lr_vox2world = torch.as_tensor(affine_lr_vox2world).to(torch.float32)
    vox_extent = list()
    for patch_dim_len, patch_center in zip(patch_size, patch_center_lr_vox):
        half_patch_start = math.ceil(patch_dim_len // 2)
        half_patch_end = math.floor(patch_dim_len // 2)
        vox_extent.append(
            torch.arange(
                patch_center - half_patch_start, patch_center + half_patch_end
            ).to(patch_center_lr_vox)
        )
    vox_extent = torch.stack(vox_extent, dim=-1)
    patch_extent_lr_vox = vox_extent
    # Calculate world-space coordinates of the vox extent.
    world_extent = (
        affine_lr_vox2world[:3, :3] @ vox_extent.T.to(affine_lr_vox2world)
    ) + (affine_lr_vox2world[:3, 3:4])
    world_extent = world_extent.T
    lr_patch_extent_world = world_extent

    return dict(
        patch_center_lr_vox=patch_center_lr_vox,
        lr_patch_extent_world=lr_patch_extent_world,
        patch_extent_lr_vox=patch_extent_lr_vox,
    )


def _crop_fr_patch(
    d,
    vols_to_crop_key_map: dict,
    affine_vox2world_key="affine_vox2world",
    fodf_key="fodf",
    vox_size_key="vox_size",
    lr_patch_extent_world_key="lr_patch_extent_world",
    lr_vox_size_key="lr_vox_size",
    write_fr_patch_extent_world_key="fr_patch_extent_world",
):
    # Calculate patch voxel coordinates from LR patch real/spatial coordinates.
    affine_vox2world = d[affine_vox2world_key]
    affine_world2vox = torch.inverse(affine_vox2world)
    lr_patch_extent_world = d[lr_patch_extent_world_key]
    lr_patch_extent_world = lr_patch_extent_world.to(affine_world2vox)
    lr_patch_extent_in_fr_vox = (affine_world2vox[:3, :3] @ lr_patch_extent_world.T) + (
        affine_world2vox[:3, 3:4]
    )
    lr_patch_extent_in_fr_vox = lr_patch_extent_in_fr_vox.T

    # Calculate the spatial bounds of the full-res patch to be *within* the coordinates
    # of the low-res input, otherwise the network cannot be given distance-weighted
    # inputs for the borders of the full-res patch.
    # Patch shapes in FR space should be consistently-sized, so set an upper bound
    # based on the raw vox size, minus some buffer space.
    lr_vox_size = d[lr_vox_size_key]
    vox_size = d[vox_size_key]
    # Assume vox sizes are isotropic.
    EPSILON_SPACE = 1e-6
    patch_len = torch.floor(
        ((lr_vox_size[0] * (lr_patch_extent_world.shape[0] - 4)) / vox_size[0])
        - EPSILON_SPACE
    )
    # patch_len = torch.floor(
    #     ((lr_patch_extent_world.shape[0] * lr_vox_size[0]) - (2 * 2 * lr_vox_size[0]))
    #     / vox_size[0]
    # )
    patch_len = patch_len.to(torch.int).cpu().item()
    patch_center_vox_idx = (
        torch.round(torch.quantile(lr_patch_extent_in_fr_vox, q=0.5, dim=0))
        .to(torch.int)
        .cpu()
    )
    l_bound = patch_center_vox_idx - math.floor(patch_len / 2)
    u_bound = patch_center_vox_idx + math.ceil(patch_len / 2)

    # Calculate the spatial coordinates of the patch's voxel indices.
    # First, get the span of vox indices according to the bounds.
    extent_vox_l = list()
    for l, u in zip(l_bound.cpu().tolist(), u_bound.cpu().tolist()):
        extent_vox_l.append(torch.arange(l, u).to(torch.int))
    patch_extent_vox = torch.stack(extent_vox_l, dim=-1).to(torch.int32)

    fr_vol_low_limits = torch.zeros(3).to(torch.int)
    example_fr_vol = d[fodf_key]
    fr_vol_up_limits = torch.as_tensor(example_fr_vol.shape[1:]).to(torch.int)

    # Construct transforms for FR sampling.
    tfs = list()
    # Check for FR sampling out of bounds.
    # Handle under out of bounds indices.
    if (l_bound < fr_vol_low_limits).any():
        pad_pre = torch.zeros_like(l_bound)
        pad_pre[l_bound < fr_vol_low_limits] = torch.abs(
            (l_bound - fr_vol_low_limits)[l_bound < fr_vol_low_limits]
        )
        padder = monai.transforms.BorderPadd(
            keys=list(vols_to_crop_key_map.keys()),
            spatial_border=list(
                itertools.chain.from_iterable(
                    zip(pad_pre.tolist(), itertools.repeat(0, len(pad_pre)))
                )
            ),
            mode="constant",
            value=0,
        )
        tfs.append(padder)
        # Adjust the patch's center vox according to the new padding.
        patch_center_vox_idx = patch_center_vox_idx + pad_pre
        l_bound = l_bound + pad_pre

    # Handle upper out of bounds indices.
    if (u_bound > fr_vol_up_limits).any():
        pad_post = torch.zeros_like(u_bound)
        pad_post[u_bound > fr_vol_up_limits] = torch.abs(u_bound - fr_vol_up_limits)[
            u_bound > fr_vol_up_limits
        ]
        padder = monai.transforms.BorderPadd(
            keys=list(vols_to_crop_key_map.keys()),
            spatial_border=list(
                itertools.chain.from_iterable(
                    zip(
                        itertools.repeat(0, len(pad_post)),
                        pad_post.tolist(),
                    )
                )
            ),
            mode="constant",
            value=0,
        )
        tfs.append(padder)

    cropper = monai.transforms.SpatialCropd(
        list(vols_to_crop_key_map.keys()),
        roi_start=l_bound.tolist(),
        roi_end=u_bound.tolist(),
        # roi_center=patch_center_vox_idx.tolist(),
        # roi_size=(patch_len,) * 3,
        # roi_start=roi_start, roi_end=roi_end
    )
    tfs.append(cropper)
    to_crop = {v: d[v] for v in vols_to_crop_key_map.keys()}
    cropped = monai.transforms.Compose(tfs)(to_crop)
    # plt.imshow(d['fodf'][0, 51:99, 45:93, 51:99][:, 25])
    # plt.imshow(cropped['fodf'][0, :, 25])
    # Store the cropped vols into the data dict with the (possibly) new keys.
    for old_v in cropped.keys():
        d[vols_to_crop_key_map[old_v]] = cropped[old_v]
        if (torch.as_tensor(cropped[old_v].shape[1:]).to(torch.int) != patch_len).any():
            raise RuntimeError()

    fr_patch_extent_world = (
        affine_vox2world[:3, :3] @ patch_extent_vox.T.to(affine_vox2world)
    ) + (affine_vox2world[:3, 3:4])
    fr_patch_extent_world = fr_patch_extent_world.T
    assert (
        torch.amin(fr_patch_extent_world, 0) >= torch.amin(lr_patch_extent_world, 0)
    ).all()
    assert (
        torch.amax(fr_patch_extent_world, 0) <= torch.amax(lr_patch_extent_world, 0)
    ).all()
    d[write_fr_patch_extent_world_key] = fr_patch_extent_world

    return d


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


class RandIsotropicResampleAffineInteriord(
    monai.transforms.RandomizableTransform,
    monai.transforms.MapTransform,
    monai.transforms.InvertibleTransform,
):
    backend = monai.transforms.Affined.backend

    def __init__(
        self,
        keys,
        prob: float,
        isotropic_scale_range: Tuple[float, float],
        # rotate_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        # shear_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        # translate_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        # scale_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        mode: SequenceStr = GridSampleMode.BILINEAR,
        padding_mode: SequenceStr = GridSamplePadMode.REFLECTION,
        device: Optional[torch.device] = None,
        allow_missing_keys: bool = False,
    ) -> None:

        monai.transforms.MapTransform.__init__(self, keys, allow_missing_keys)
        monai.transforms.RandomizableTransform.__init__(self, prob)

        self._rand_iso_scale_param = None
        self.isotopic_scale_range = ensure_tuple(isotropic_scale_range)
        self.mode = monai.utils.ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = monai.utils.ensure_tuple_rep(padding_mode, len(self.keys))

    def randomize(self, data: Optional[Any] = None) -> None:
        self._rand_iso_scale_param = self.R.uniform(
            self.isotopic_scale_range[0], self.isotopic_scale_range[1]
        )
        super().randomize(None)

    @staticmethod
    def interior_spatial_shape(input_to_target_scale: float, fov_shape) -> Tuple[float]:
        target_space_in_input_space = np.floor(input_to_target_scale / 2)
        target_shape = tuple(np.asarray(2 * target_space_in_input_space).tolist())
        return target_shape

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            out = convert_to_tensor(d, track_meta=get_track_meta())
            return out

        self.randomize(None)

        # do random transform?
        do_resampling = self._do_transform

        # do the transform
        if do_resampling:
            spatial_size = d[first_key].shape[1:]

            scale_params = (self._rand_iso_scale_param,) * len(spatial_size)

            target_shape = self.interior_spatial_shape(
                self._rand_iso_scale_param, fov_shape=np.asarray(spatial_size)
            )
            affine_tf = monai.transforms.Affined(
                self.keys,
                scale_params=scale_params,
                spatial_size=target_shape,
                mode=self.mode,
                padding_mode=self.padding_mode,
                device=self.device,
                dtype=self.dtype,
                allow_missing_keys=self.allow_missing_keys,
            )

            d = affine_tf(d)

        for key in self.key_iterator(d):
            if not do_resampling:
                d[key] = convert_to_tensor(
                    d[key], track_meta=get_track_meta(), dtype=torch.float32
                )
            if get_track_meta():
                for key in self.key_iterator(d):
                    xform = (
                        self.pop_transform(d[key], check=False) if do_resampling else {}
                    )
                    self.push_transform(
                        d[key],
                        extra_info={
                            "do_resampling": do_resampling,
                            "rand_affine_info": xform,
                        },
                    )

        return d


def _random_iso_scale_affine(
    src_affine: np.ndarray, scale_low: float, scale_high: float
) -> np.ndarray:
    scale = np.random.uniform(scale_low, scale_high)
    scaling_affine = monai.transforms.utils.create_scale(
        3,
        [scale] * 3,
    )
    scaled_affine = scaling_affine @ src_affine
    return scaled_affine


def _crop_fr_min_hull_by_lr_world_coords_extent(
    d: dict,
    keys_to_crop: tuple,
    output_keys: tuple,
    affine_fr_vox2world_key,
    lr_vox_extent_fov_im_key,
    affine_lr_patch_vox2world_key,
    im_cropper,
):
    SPATIAL_COORD_PRECISION = 5
    lr_fov_vox = tuple(d[lr_vox_extent_fov_im_key].shape[-3:])
    lr_affine_vox2world = torch.as_tensor(d[affine_lr_patch_vox2world_key])
    fr_affine_vox2world = torch.as_tensor(d[affine_fr_vox2world_key])
    shared_dtype = torch.result_type(lr_affine_vox2world, fr_affine_vox2world)
    lr_affine_vox2world = lr_affine_vox2world.to(shared_dtype)
    fr_affine_vox2world = fr_affine_vox2world.to(shared_dtype)

    affine_lr_patch_vox2fr_vox = (
        torch.linalg.inv(fr_affine_vox2world) @ lr_affine_vox2world
    )
    # Account for any floating point errors that may cause the bottom row in the
    # homogeneous matrix to be invalid.
    affine_lr_patch_vox2fr_vox[-1] = torch.round(
        torch.abs(affine_lr_patch_vox2fr_vox[-1])
    )
    lr_fov_vox_low = (0,) * len(lr_fov_vox)
    fr_in_lr_fov_vox_low = (
        pitn.affine.transform_coords(
            lr_affine_vox2world.new_tensor(lr_fov_vox_low), affine_lr_patch_vox2fr_vox
        )
        .round_(decimals=SPATIAL_COORD_PRECISION)
        .floor_()
        .to(torch.int)
    )
    lr_fov_vox_high = lr_fov_vox
    fr_in_lr_fov_vox_high = (
        pitn.affine.transform_coords(
            lr_affine_vox2world.new_tensor(lr_fov_vox_high), affine_lr_patch_vox2fr_vox
        )
        .round_(decimals=SPATIAL_COORD_PRECISION)
        .ceil_()
        .to(torch.int)
    )
    fr_patch_slices = monai.transforms.Crop.compute_slices(
        roi_start=fr_in_lr_fov_vox_low, roi_end=fr_in_lr_fov_vox_high
    )

    if isinstance(keys_to_crop, str):
        keys_to_crop = (keys_to_crop,)
    if isinstance(output_keys, str):
        output_keys = (output_keys,)
    assert len(keys_to_crop) == len(output_keys)
    for k_in, k_out in zip(keys_to_crop, output_keys):
        im = d[k_in]
        im_patch = im_cropper(im, fr_patch_slices)
        d[k_out] = im_patch

    return d


class HCPfODFINRPatchDataset(monai.data.PatchDataset):

    _SAMPLE_KEYS = (
        "subj_id",
        "lr_dwi",
        "lr_mask",
        "lr_bval",
        "lr_bvec",
        "lr_fodf",
        "lr_vox_size",
        "affine_lr_patch_vox2world",
        "fodf",
        "mask",
        "vox_size",
        "affine_fr_patch_vox2world",
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
        keys=("lr_dwi", "lr_mask", "lr_fodf"),
        w_key="lr_sampling_mask",
        **sample_tf_kwargs,
    ):
        return _EfficientRandWeightedCropd(keys=keys, w_key=w_key, **sample_tf_kwargs)

    @staticmethod
    def default_feature_tf(
        augmentation_prob: float,
        iso_scale_factor_lr_spacing_mm_low_high: Tuple[float] = (1.0, 1.0),
        augment_spatial_resample_kwargs: dict = dict(),
        augment_rand_rician_noise_kwargs: dict = dict(),
        augment_rand_rotate_90_kwargs: dict = dict(),
        augment_rand_flip_kwargs: dict = dict(),
    ):
        VOL_KEYS = (
            "lr_dwi",
            "lr_mask",
            "lr_fodf",
            "lr_fivett",
            "fodf",
            "mask",
            "fivett",
        )
        # Transforms for extracting features for the network.
        feat_tfs = list()

        # Extract the new LR patch affine matrix.
        feat_tfs.append(
            functools.partial(
                _extract_affine,
                src_vol_key="lr_dwi",
                write_key="_orig_affine_lr_patch_vox2world",
            )
        )
        # If augmenting the lr patch, then the fov of the fr patch must be adjusted to
        # match.
        feat_tfs.append(
            monai.transforms.CopyItemsd(
                keys=["_orig_affine_lr_patch_vox2world"],
                times=1,
                names=["target_affine_lr_patch"],
            )
        )

        augment_tfs = list()
        non_augment_tfs = list()

        # Augmentation transforms, subject to `augmentation prob`.
        # Random iso-scaling
        low, high = iso_scale_factor_lr_spacing_mm_low_high
        scale_aff_tf = monai.transforms.RandLambdad(
            "target_affine_lr_patch",
            partial(_random_iso_scale_affine, low=low, high=high),
            overwrite=True,
        )
        resample_lr_patch_tf = monai.transforms.SpatialResampled(
            **{
                **dict(
                    keys=["lr_dwi", "lr_fodf", "lr_mask", "lr_fivett"],
                    mode=["bilinear", "bilinear", "nearest", "nearest"],
                    align_corners=True,
                    dst_keys="target_affine_lr_patch",
                ),
                **augment_spatial_resample_kwargs,
            }
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
        augment_tfs.append(scale_aff_tf)
        augment_tfs.append(resample_lr_patch_tf)
        augment_tfs.append(noise_tf)

        # Crop the target patch according to the center and fov of the lr patch,
        # agnostic as to whether the lr patch was randomly augmented.
        # Crop the "minimum hull" such that the fr patch fully encompasses all points
        # in the lr patch, extending beyond the lr coordinates by a maximum of 1 voxel.
        fr_cropper = monai.transforms.Crop()
        fr_crop_min_hull_tf = partial(
            _crop_fr_min_hull_by_lr_world_coords_extent,
            keys_to_crop=("fodf", "mask", "fivett"),
            output_keys=("fodf", "mask", "fivett"),
            affine_fr_vox2world_key="affine_vox2world",
            lr_vox_extent_fov_im_key="lr_dwi",
            affine_lr_patch_vox2world_key="target_affine_lr_patch",
            im_cropper=fr_cropper,
        )
        # Crop this "min hull" by 1 voxel on each side to accomodate the INR
        # output interpolation scheme.
        fr_crop_by_1_tf = monai.transforms.SpatialCropd(
            keys=("fodf", "mask", "fivett"), roi_slices=(slice(1, -1),) * 3
        )
        # These transforms are performed regardless of whether augmentation is being
        # done.
        augment_tfs.append(fr_crop_min_hull_tf)
        augment_tfs.append(fr_crop_by_1_tf)
        non_augment_tfs.append(fr_crop_min_hull_tf)
        non_augment_tfs.append(fr_crop_by_1_tf)

        # Apply random flips and rotations to both the lr patch and the fr patch.
        # Random 90 degree rotations
        rotate_tf = monai.transforms.RandRotate90d(
            keys=VOL_KEYS,
            max_k=3,
            spatial_axes=(0, 1, 2),
        )
        # Random flips
        flip_tf = monai.transforms.RandFlipd(keys=VOL_KEYS, spatial_axis=(0, 1, 2))
        augment_tfs.append(rotate_tf)
        augment_tfs.append(flip_tf)

        augment_tfs = monai.transforms.Compose(augment_tfs)
        non_augment_tfs = monai.transforms.Compose(non_augment_tfs)

        # Create a transformation branch conditioned on the chance of whether or not
        # to use augmentation.
        augment_branch_tf = monai.transforms.OneOf(
            transforms=(augment_tfs, non_augment_tfs),
            weights=(augmentation_prob, 1.0 - augmentation_prob),
        )
        feat_tfs.append(augment_branch_tf)

        # Remove the bvec and bval, as we may have altered the orientation of the images
        # and they are no longer correct (and are not used at this time). Also remove
        # tmp affine values.
        feat_tfs.append(
            monai.transforms.DeleteItemsd(
                keys=(
                    "_orig_affine_lr_patch_vox2world",
                    "target_affine_lr_patch",
                    "lr_bval",
                    "lr_bvec",
                )
            )
        )

        # Extract the new LR patch affine matrix.
        feat_tfs.append(
            functools.partial(
                _extract_affine,
                src_vol_key="lr_dwi",
                write_key="affine_lr_patch_vox2world",
            )
        )
        feat_tfs.append(
            monai.transforms.adaptor(
                monai.data.utils.affine_to_spacing,
                outputs="lr_vox_size",
                inputs={"affine_lr_patch_vox2world": "affine"},
            )
        )
        # Similarly for the FR patch.
        feat_tfs.append(
            functools.partial(
                _extract_affine,
                src_vol_key="fodf",
                write_key="affine_fr_patch_vox2world",
            )
        )
        feat_tfs.append(
            monai.transforms.adaptor(
                monai.data.utils.affine_to_spacing,
                outputs="fr_vox_size",
                inputs={"affine_fr_patch_vox2world": "affine"},
            )
        )

        # Remove unnecessary items from the data dict.
        # Sub-select keys to free memory.
        select_k_tf = monai.transforms.SelectItemsd(
            [
                "subj_id",
                "lr_dwi",
                "lr_mask",
                "lr_fodf",
                "lr_bval",
                "lr_bvec",
                "affine_lr_patch_vox2world",
                "fodf",
                "mask",
                "lr_vox_size",
                "fr_vox_size",
            ]
        )
        feat_tfs.append(select_k_tf)

        # Convert all MetaTensors to regular Tensors.
        to_tensor_tf = monai.transforms.ToTensord(
            [
                "lr_dwi",
                "lr_mask",
                "lr_fodf",
                "affine_lr_patch_vox2world",
                "lr_vox_size",
                "fodf",
                "mask",
                "affine_fr_patch_vox2world",
            ],
            track_meta=False,
        )
        feat_tfs.append(to_tensor_tf)

        select_k_tf = monai.transforms.SelectItemsd(
            [
                "subj_id",
                "lr_dwi",
                "lr_mask",
                "lr_fodf",
                "affine_lr_patch_vox2world",
                # "lr_bval",
                # "lr_bvec",
                "fodf",
                "mask",
                "vox_size",
                "lr_vox_size",
            ]
        )
        feat_tfs.append(select_k_tf)

        return monai.transforms.Compose(feat_tfs)


class HCPfODFINRWholeVolDataset(monai.data.Dataset):
    _SAMPLE_KEYS = (
        "subj_id",
        "lr_dwi",
        "lr_fodf",
        "lr_bval",
        "lr_bvec",
        "lr_vox_size",
        "lr_extent_world",
        "lr_freesurfer_seg",
        "lr_fivett",
        "affine_lr_vox2world",
        "fodf",
        "fivett",
        "mask",
        "vox_size",
        "extent_world",
        "affine_vox2world",
    )

    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        super().__init__(
            self.base_dataset,
            transform=transform,
        )

    @staticmethod
    def default_tf():
        # Transforms for extracting features for the network.
        feat_tfs = list()

        # Extract LR features.
        # Pad the low-res by 3 in each direction, to compensate for off-by-1 (1-ish)
        # errors in re-sampling in the network.
        off_by_1_padder = monai.transforms.BorderPadd(
            ["lr_dwi", "lr_fodf", "lr_mask", "lr_fivett", "lr_freesurfer_seg"],
            spatial_border=3,
            mode="constant",
            value=0,
        )
        feat_tfs.append(off_by_1_padder)
        # Extract the LR affine matrix.
        feat_tfs.append(
            functools.partial(
                _extract_affine,
                src_vol_key="lr_dwi",
                write_key="affine_lr_vox2world",
            )
        )
        lr_vox_size_tf = monai.transforms.adaptor(
            monai.data.utils.affine_to_spacing,
            outputs="lr_vox_size",
            inputs={"affine_lr_vox2world": "affine"},
        )
        feat_tfs.append(lr_vox_size_tf)

        lr_extent_world_tf = monai.transforms.adaptor(
            _get_extent_world,
            outputs="lr_extent_world",
            inputs={"lr_dwi": "x_pre_img", "affine_lr_vox2world": "affine"},
        )
        feat_tfs.append(lr_extent_world_tf)

        # Extract full-res features.
        # Crop the full-res by 2 in each direction, to compensate for off-by-1 (1-ish)
        # errors in re-sampling in the network.
        # off_by_1_cropper = monai.transforms.SpatialCropd(
        #     ["fodf", "mask"], roi_slices=(slice(2, -2), slice(2, -2), slice(2, -2))
        # )
        # feat_tfs.append(off_by_1_cropper)
        # Extract the full-res affine matrix.
        feat_tfs.append(
            functools.partial(
                _extract_affine,
                src_vol_key="fodf",
                write_key="affine_vox2world",
            )
        )
        vox_size_tf = monai.transforms.adaptor(
            monai.data.utils.affine_to_spacing,
            outputs="vox_size",
            inputs={"affine_vox2world": "affine"},
        )
        feat_tfs.append(vox_size_tf)

        extent_world_tf = monai.transforms.adaptor(
            _get_extent_world,
            outputs="extent_world",
            inputs={"fodf": "x_pre_img", "affine_vox2world": "affine"},
        )
        feat_tfs.append(extent_world_tf)

        # Remove unnecessary items from the data dict.
        # Sub-select keys to free memory.
        select_k_tf = monai.transforms.SelectItemsd(
            [
                "subj_id",
                "lr_dwi",
                "lr_fodf",
                "lr_mask",
                "affine_lr_vox2world",
                # "lr_bval",
                # "lr_bvec",
                "lr_freesurfer_seg",
                "lr_fivett",
                "lr_extent_world",
                "lr_vox_size",
                "fodf",
                "mask",
                "fivett",
                "extent_world",
                "affine_vox2world",
                "vox_size",
            ]
        )
        feat_tfs.append(select_k_tf)

        # Convert all MetaTensors to regular Tensors.
        to_tensor_tf = monai.transforms.ToTensord(
            [
                "lr_dwi",
                "lr_fodf",
                "lr_mask",
                "lr_extent_world",
                "lr_freesurfer_seg",
                "lr_fivett",
                "lr_vox_size",
                "affine_lr_vox2world",
                "fodf",
                "mask",
                "fivett",
                "extent_world",
                "vox_size",
                "affine_vox2world",
            ],
            track_meta=False,
        )
        feat_tfs.append(to_tensor_tf)
        # ~~Generate features from each DWI and the associated bval and bvec.~~

        select_k_tf = monai.transforms.SelectItemsd(
            [
                "subj_id",
                "lr_dwi",
                "lr_fodf",
                "lr_mask",
                "lr_extent_world",
                "lr_freesurfer_seg",
                "lr_fivett",
                "lr_vox_size",
                "affine_lr_vox2world",
                # "lr_bval",
                # "lr_bvec",
                "fodf",
                "mask",
                "fivett",
                "extent_world",
                "affine_vox2world",
                "vox_size",
            ]
        )
        feat_tfs.append(select_k_tf)

        return monai.transforms.Compose(feat_tfs)
