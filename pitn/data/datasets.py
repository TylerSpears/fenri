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
        "lr_vox_size",
        "lr_patch_extent_acpc",
        "lr_freesurfer_seg",
        "lr_fivett",
        "fodf",
        "mask",
        "fivett",
        "freesurfer_seg",
        "vox_size",
        "fr_patch_extent_acpc",
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
                lr_freesurfer_seg=lr_fodf_data["freesurfer_seg"],
                lr_fivett=lr_fodf_data["fivett"],
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
        tfs = list()
        # Load images
        vol_reader = monai.data.NibabelReader(
            as_closest_canonical=True, dtype=np.float32
        )
        tfs.append(
            monai.transforms.LoadImaged(
                (
                    "lr_dwi",
                    "fodf",
                    "lr_fodf",
                    "lr_mask",
                    "mask",
                    "fivett",
                    "lr_freesurfer_seg",
                    "lr_fivett",
                ),
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
                (
                    "lr_dwi",
                    "fodf",
                    "lr_fodf",
                    "lr_mask",
                    "mask",
                    "fivett",
                    "lr_freesurfer_seg",
                    "lr_fivett",
                ),
                track_meta=True,
            )
        )
        tfs.append(monai.transforms.ToTensord(("lr_bval", "lr_bvec"), track_meta=False))
        tfs.append(
            monai.transforms.CastToTyped(
                ("lr_mask", "mask", "fivett", "lr_fivett"), dtype=torch.uint8
            )
        )

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
                _extract_affine, src_vol_key="lr_dwi", write_key="affine_lrvox2acpc"
            )
        )
        tfs.append(
            functools.partial(
                _extract_affine, src_vol_key="fodf", write_key="affine_vox2acpc"
            )
        )
        vox_size_tf = monai.transforms.adaptor(
            monai.data.utils.affine_to_spacing,
            outputs="vox_size",
            inputs={"affine_vox2acpc": "affine"},
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
    d[write_key] = torch.clone(aff).to(torch.float32)
    return d


def _extract_lr_patch_info(lr_dwi, affine_lrvox2acpc, patch_size: tuple):
    # Extract low-resolution input information
    patch_center_lrvox = torch.clone(
        lr_dwi.meta["crop_center"].as_tensor().to(torch.int)
    )
    affine_lrvox2acpc = torch.as_tensor(affine_lrvox2acpc).to(torch.float32)
    vox_extent = list()
    for patch_dim_len, patch_center in zip(patch_size, patch_center_lrvox):
        half_patch_start = math.ceil(patch_dim_len // 2)
        half_patch_end = math.floor(patch_dim_len // 2)
        vox_extent.append(
            torch.arange(
                patch_center - half_patch_start, patch_center + half_patch_end
            ).to(patch_center_lrvox)
        )
    vox_extent = torch.stack(vox_extent, dim=-1)
    patch_extent_lrvox = vox_extent
    # Calculate acpc-space coordinates of the vox extent.
    acpc_extent = (affine_lrvox2acpc[:3, :3] @ vox_extent.T.to(affine_lrvox2acpc)) + (
        affine_lrvox2acpc[:3, 3:4]
    )
    acpc_extent = acpc_extent.T
    lr_patch_extent_acpc = acpc_extent

    return dict(
        patch_center_lrvox=patch_center_lrvox,
        lr_patch_extent_acpc=lr_patch_extent_acpc,
        patch_extent_lrvox=patch_extent_lrvox,
    )


def _crop_fr_patch(
    d,
    vols_to_crop_key_map: dict,
    affine_vox2acpc_key="affine_vox2acpc",
    fodf_key="fodf",
    vox_size_key="vox_size",
    lr_patch_extent_acpc_key="lr_patch_extent_acpc",
    lr_vox_size_key="lr_vox_size",
    write_fr_patch_extent_acpc_key="fr_patch_extent_acpc",
):
    # Calculate patch voxel coordinates from LR patch real/spatial coordinates.
    affine_vox2acpc = d[affine_vox2acpc_key]
    affine_acpc2vox = torch.inverse(affine_vox2acpc)
    lr_patch_extent_acpc = d[lr_patch_extent_acpc_key]
    lr_patch_extent_acpc = lr_patch_extent_acpc.to(affine_acpc2vox)
    lr_patch_extent_in_fr_vox = (affine_acpc2vox[:3, :3] @ lr_patch_extent_acpc.T) + (
        affine_acpc2vox[:3, 3:4]
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
        ((lr_vox_size[0] * (lr_patch_extent_acpc.shape[0] - 4)) / vox_size[0])
        - EPSILON_SPACE
    )
    # patch_len = torch.floor(
    #     ((lr_patch_extent_acpc.shape[0] * lr_vox_size[0]) - (2 * 2 * lr_vox_size[0]))
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

    fr_patch_extent_acpc = (
        affine_vox2acpc[:3, :3] @ patch_extent_vox.T.to(affine_vox2acpc)
    ) + (affine_vox2acpc[:3, 3:4])
    fr_patch_extent_acpc = fr_patch_extent_acpc.T
    assert (
        torch.amin(fr_patch_extent_acpc, 0) >= torch.amin(lr_patch_extent_acpc, 0)
    ).all()
    assert (
        torch.amax(fr_patch_extent_acpc, 0) <= torch.amax(lr_patch_extent_acpc, 0)
    ).all()
    d[write_fr_patch_extent_acpc_key] = fr_patch_extent_acpc

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
    def interior_spatial_shape(
        input_to_target_scale: float, float, fov_shape
    ) -> Tuple[float]:
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


class HCPfODFINRPatchDataset(monai.data.PatchDataset):

    _SAMPLE_KEYS = (
        "subj_id",
        "lr_dwi",
        "lr_mask",
        "lr_bval",
        "lr_bvec",
        "lr_fodf",
        "lr_vox_size",
        "lr_patch_extent_acpc",
        "affine_lr_patchvox2acpc",
        "fodf",
        "mask",
        "vox_size",
        "fr_patch_extent_acpc",
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

    def set_select_tf_keys(
        self,
        keys: Optional[List[str]] = None,
        remove_keys: Optional[List[str]] = None,
        add_keys: Optional[List[str]] = None,
        select_tf_idx=-1,
    ):
        raise NotImplementedError(
            "ERROR: Not implemented yet due to multiprocessing funny business"
        )
        if self.transform is None:
            new_tfs = None
        else:
            tfs = self.transform.transforms
            if keys is None and remove_keys is None and add_keys is None:
                raise ValueError(
                    "ERROR: One of 'keys', 'remove_keys', 'add_keys' must be set."
                )
            elif keys is not None and (remove_keys is not None or add_keys is not None):
                raise ValueError(
                    "ERROR: If 'keys' is set, then `remove_keys` and",
                    "'add_keys' must be None",
                )
            idx = select_tf_idx % len(tfs)
            new_tfs = list()
            new_tfs.extend(tfs[:idx])
            if keys is not None:
                new_select_tf = monai.transforms.SelectItemsd(keys)
            if remove_keys is not None or add_keys is not None:
                new_keys = tfs[idx].keys
                if remove_keys is not None:
                    new_keys = tuple(
                        filter(lambda x: x not in tuple(remove_keys), new_keys)
                    )
                if add_keys is not None:
                    # Only add keys that are not already in the set of keys.
                    new_keys = new_keys + tuple(
                        filter(lambda x: x not in new_keys, add_keys)
                    )
                new_select_tf = monai.transforms.SelectItemsd(new_keys)

            new_tfs.append(new_select_tf)
            new_tfs.extend(tfs[idx + 1 :])
            new_tfs = monai.transforms.Compose(new_tfs)
        self.transform = new_tfs

    @staticmethod
    def default_patch_func(
        keys=("lr_dwi", "lr_mask", "lr_fodf"),
        w_key="lr_sampling_mask",
        **sample_tf_kwargs,
    ):
        return _EfficientRandWeightedCropd(keys=keys, w_key=w_key, **sample_tf_kwargs)

    @staticmethod
    def default_feature_tf(
        patch_size: tuple,  #!Need to dynamically find this now that we have rand sizes
        # patch_scale_range: Optional[Tuple[float, float]] = None,
        # patch_scale_prob: Optional[float] = None,
    ):
        # Transforms for extracting features for the network.
        feat_tfs = list()

        # if patch_scale_range is not None and patch_scale_prob is not None:
        #     rand_lr_patch_resample_tf = RandIsotropicResampleAffineInteriord(
        #         [
        #             "lr_dwi",
        #             "lr_fodf",
        #             "lr_mask",
        #         ],
        #         prob=patch_scale_prob,
        #         isotropic_scale_range=patch_scale_range,
        #         mode=["bilinear", "bilinear", "nearest"],
        #         padding_mode="zeros",
        #     )
        #     feat_tfs.append(rand_lr_patch_resample_tf)

        # Extract the new LR patch affine matrix.
        feat_tfs.append(
            functools.partial(
                _extract_affine,
                src_vol_key="lr_dwi",
                write_key="affine_lr_patchvox2acpc",
            )
        )
        lr_vox_size_tf = monai.transforms.adaptor(
            monai.data.utils.affine_to_spacing,
            outputs="lr_vox_size",
            inputs={"affine_lr_patchvox2acpc": "affine"},
        )
        feat_tfs.append(lr_vox_size_tf)

        extract_lr_patch_meta_tf = monai.transforms.adaptor(
            functools.partial(_extract_lr_patch_info, patch_size=patch_size),
            outputs={
                "patch_center_lrvox": "patch_center_lrvox",
                "lr_patch_extent_acpc": "lr_patch_extent_acpc",
                "patch_extent_lrvox": "patch_extent_lrvox",
            },
        )
        feat_tfs.append(extract_lr_patch_meta_tf)

        # Crop full-res vols with the spatial extent of the LR patch sample.
        crop_fr_patch_tf = functools.partial(
            _crop_fr_patch,
            vols_to_crop_key_map={"fodf": "fodf", "mask": "mask"},
            affine_vox2acpc_key="affine_vox2acpc",
            fodf_key="fodf",
            vox_size_key="vox_size",
            lr_patch_extent_acpc_key="lr_patch_extent_acpc",
            lr_vox_size_key="lr_vox_size",
            write_fr_patch_extent_acpc_key="fr_patch_extent_acpc",
        )
        feat_tfs.append(crop_fr_patch_tf)

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
                "affine_lr_patchvox2acpc",
                "fodf",
                "mask",
                "lr_patch_extent_acpc",
                "fr_patch_extent_acpc",
                "lr_vox_size",
                "vox_size",
            ]
        )
        feat_tfs.append(select_k_tf)

        # These coordinates will be re-indexed later to match what is expected by `grid_sample()`.
        vox_physical_coords_tf = monai.transforms.Lambdad(
            [
                "lr_patch_extent_acpc",
                "fr_patch_extent_acpc",
            ],
            lambda c: einops.rearrange(
                torch.cartesian_prod(*c.T),
                "(p1 p2 p3) d -> d p1 p2 p3",
                p1=c.shape[0],
                p2=c.shape[0],
                p3=c.shape[0],
            ).to(torch.float32),
            overwrite=True,
        )
        feat_tfs.append(vox_physical_coords_tf)

        # Convert all MetaTensors to regular Tensors.
        to_tensor_tf = monai.transforms.ToTensord(
            [
                "lr_dwi",
                "lr_mask",
                "lr_fodf",
                "affine_lr_patchvox2acpc",
                "fodf",
                "mask",
                "lr_patch_extent_acpc",
                "fr_patch_extent_acpc",
                "lr_vox_size",
                "vox_size",
            ],
            track_meta=False,
        )
        feat_tfs.append(to_tensor_tf)
        # ~~Generate features from each DWI and the associated bval and bvec.~~

        select_k_tf = monai.transforms.SelectItemsd(
            [
                "subj_id",
                "lr_dwi",
                "lr_mask",
                "lr_fodf",
                "affine_lr_patchvox2acpc",
                # "lr_bval",
                # "lr_bvec",
                "fodf",
                "mask",
                "lr_patch_extent_acpc",
                "fr_patch_extent_acpc",
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
        "lr_extent_acpc",
        "lr_freesurfer_seg",
        "lr_fivett",
        "affine_lrvox2acpc",
        "fodf",
        "mask",
        "vox_size",
        "extent_acpc",
        "affine_vox2acpc",
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
                write_key="affine_lrvox2acpc",
            )
        )
        lr_vox_size_tf = monai.transforms.adaptor(
            monai.data.utils.affine_to_spacing,
            outputs="lr_vox_size",
            inputs={"affine_lrvox2acpc": "affine"},
        )
        feat_tfs.append(lr_vox_size_tf)

        lr_extent_acpc_tf = monai.transforms.adaptor(
            _get_extent_world,
            outputs="lr_extent_acpc",
            inputs={"lr_dwi": "x_pre_img", "affine_lrvox2acpc": "affine"},
        )
        feat_tfs.append(lr_extent_acpc_tf)

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
                write_key="affine_vox2acpc",
            )
        )
        vox_size_tf = monai.transforms.adaptor(
            monai.data.utils.affine_to_spacing,
            outputs="vox_size",
            inputs={"affine_vox2acpc": "affine"},
        )
        feat_tfs.append(vox_size_tf)

        extent_acpc_tf = monai.transforms.adaptor(
            _get_extent_world,
            outputs="extent_acpc",
            inputs={"fodf": "x_pre_img", "affine_vox2acpc": "affine"},
        )
        feat_tfs.append(extent_acpc_tf)

        # Remove unnecessary items from the data dict.
        # Sub-select keys to free memory.
        select_k_tf = monai.transforms.SelectItemsd(
            [
                "subj_id",
                "lr_dwi",
                "lr_fodf",
                "lr_mask",
                "affine_lrvox2acpc",
                # "lr_bval",
                # "lr_bvec",
                "lr_freesurfer_seg",
                "lr_fivett",
                "lr_extent_acpc",
                "lr_vox_size",
                "fodf",
                "mask",
                "extent_acpc",
                "affine_vox2acpc",
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
                "lr_extent_acpc",
                "lr_freesurfer_seg",
                "lr_fivett",
                "lr_vox_size",
                "affine_lrvox2acpc",
                "fodf",
                "mask",
                "extent_acpc",
                "vox_size",
                "affine_vox2acpc",
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
                "lr_extent_acpc",
                "lr_freesurfer_seg",
                "lr_fivett",
                "lr_vox_size",
                "affine_lrvox2acpc",
                # "lr_bval",
                # "lr_bvec",
                "fodf",
                "mask",
                "extent_acpc",
                "affine_vox2acpc",
                "vox_size",
            ]
        )
        feat_tfs.append(select_k_tf)

        return monai.transforms.Compose(feat_tfs)
