# -*- coding: utf-8 -*-
import collections
import functools
import itertools
import math
from pathlib import Path
from typing import List, Optional

import einops
import monai
import nibabel as nib
import numpy as np
import skimage
import torch

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

    @staticmethod
    def default_pre_sample_tf(mask_dilate_radius: int):
        tfs = list()
        # Load images
        vol_reader = monai.data.NibabelReader(
            as_closest_canonical=True, dtype=np.float32
        )
        tfs.append(
            monai.transforms.LoadImaged(
                ("lr_dwi", "fodf", "lr_fodf", "lr_mask", "mask", "fivett"),
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
                ("lr_dwi", "fodf", "lr_fodf", "lr_mask", "mask", "fivett"),
                track_meta=True,
            )
        )
        tfs.append(monai.transforms.ToTensord(("lr_bval", "lr_bvec"), track_meta=False))
        tfs.append(
            monai.transforms.CastToTyped(
                ("lr_mask", "mask", "fivett"), dtype=torch.uint8
            )
        )

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


class HCPINRfODFPatchDataset(monai.data.PatchDataset):

    _SAMPLE_KEYS = (
        "subj_id",
        "lr_dwi",
        "lr_mask",
        "lr_bval",
        "lr_bvec",
        "lr_fodf",
        "lr_vox_size",
        "lr_patch_extent_acpc",
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
        tfs = self.transform
        if tfs is None:
            new_tfs = None
        else:
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
                        filter(lambda x: x not in tuple(remove_keys)), new_keys
                    )
                if add_keys is not None:
                    # Only add keys that are not already in the set of keys.
                    new_keys = new_keys + tuple(
                        filter(lambda x: x not in new_keys), add_keys
                    )
                new_select_tf = monai.transforms.SelectItemsd(new_keys)

            new_tfs.append(new_select_tf)
            new_tfs.extend(tfs[idx + 1 :])
            new_tfs = monai.transforms.compose(new_tfs)
        self.transform = new_tfs

    @staticmethod
    def default_patch_func(
        keys=("lr_dwi", "lr_mask", "lr_fodf"),
        w_key="lr_sampling_mask",
        **sample_tf_kwargs,
    ):
        return monai.transforms.RandWeightedCropd(
            keys=keys, w_key=w_key, **sample_tf_kwargs
        )

    @staticmethod
    def default_feature_tf(patch_size: tuple):
        # Transforms for extracting features for the network.
        feat_tfs = list()

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

        # # Derive full-res patch properties based on low-res patch selection.
        # extract_full_res_patch_meta_tf = monai.transforms.adaptor(
        #     _extract_full_res_patch_info,
        #     outputs={
        #         "affine_vox2acpc": "affine_vox2acpc",
        #         "fr_patch_vox_lu_bound": "fr_patch_vox_lu_bound",
        #         "fr_patch_extent_acpc": "fr_patch_extent_acpc",
        #     },
        # )
        # feat_tfs.append(extract_full_res_patch_meta_tf)

        # crop_fr_tf = functools.partial(
        #     _crop_sample_fodf_mask,
        #     fr_patch_vox_lu_bound_key="fr_patch_vox_lu_bound",
        # )

        # feat_tfs.append(crop_fr_tf)

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
    max_patch_len = (
        torch.floor(lr_patch_extent_acpc.shape[0] * lr_vox_size[0] / vox_size[0]) - 4
    )
    max_patch_len = max_patch_len.to(torch.int).cpu().item()
    patch_center_vox_idx = (
        torch.round(torch.quantile(lr_patch_extent_in_fr_vox, q=0.5, dim=0))
        .to(torch.int)
        .cpu()
    )
    l_bound = patch_center_vox_idx - math.floor(max_patch_len / 2)
    u_bound = patch_center_vox_idx + math.ceil(max_patch_len / 2)

    # patch_center_vox_idx = torch.quantile(lr_patch_extent_in_fr_vox, q=0.5, dim=0)
    # l_bound = patch_center_vox_idx - (max_patch_len / 2)
    # u_bound = patch_center_vox_idx + (max_patch_len / 2)
    # l_bound = torch.round(l_bound).to(torch.int)
    # u_bound = torch.round(u_bound).to(torch.int)
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
        # roi_size=(max_patch_len,) * 3,
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


# def _extract_full_res_patch_info(
#     fodf,
#     lr_patch_extent_acpc,
# ):
#     # Extract full-resolution information.
#     affine_vox2acpc = torch.clone(torch.as_tensor(fodf.affine, dtype=torch.float32))
#     affine_acpc2vox = torch.inverse(affine_vox2acpc)
#     lr_patch_extent_acpc = lr_patch_extent_acpc.to(affine_acpc2vox)
#     patch_extent_vox = (affine_acpc2vox[:3, :3] @ lr_patch_extent_acpc.T) + (
#         affine_acpc2vox[:3, 3:4]
#     )

#     patch_extent_vox = patch_extent_vox.T
#     # Calculate the spatial bounds of the full-res patch to be *within* the coordinates
#     # of the low-res input, otherwise the network cannot be given distance-weighted
#     # inputs for the borders of the full-res patch.
#     l_bound = torch.ceil(patch_extent_vox.min(dim=0).values).to(torch.int)
#     u_bound = torch.floor(patch_extent_vox.max(dim=0).values).to(torch.int)
#     fr_patch_shape = u_bound - l_bound
#     if (fr_patch_shape != fr_patch_shape.max()).any():
#         target_size = fr_patch_shape.max()
#         vol_shape = torch.tensor(fodf.shape[1:])
#         for dim, dim_size in enumerate(fr_patch_shape):
#             if dim_size != target_size:
#                 diff = target_size - dim_size
#                 # Try to increase the upper bound first.
#                 if u_bound[dim] + diff <= vol_shape[dim]:
#                     u_bound[dim] = u_bound[dim] + diff
#                 elif l_bound[dim] - diff >= 0:
#                     l_bound[dim] = l_bound[dim] - diff
#                 else:
#                     raise RuntimeError(
#                         "ERROR: Non-isotropic full-res patch shape", f"{fr_patch_shape}"
#                     )
#         fr_patch_shape = u_bound - l_bound

#     # Store the vox upper and lower bounds now for later patch extraction from the full-
#     # res fodf and mask (*much* faster to index into a tensor with a range() than each
#     # individual index).
#     fr_patch_vox_lu_bound = torch.stack([l_bound, u_bound], dim=-1)
#     extent_vox_l = list()
#     for l, u in zip(l_bound.cpu().tolist(), u_bound.cpu().tolist()):
#         extent_vox_l.append(torch.arange(l, u).to(torch.int))
#     patch_extent_vox = torch.stack(extent_vox_l, dim=-1).to(torch.int32)

#     fr_patch_extent_acpc = (
#         affine_vox2acpc[:3, :3] @ patch_extent_vox.T.to(affine_vox2acpc)
#     ) + (affine_vox2acpc[:3, 3:4])
#     fr_patch_extent_acpc = fr_patch_extent_acpc.T
#     fr_patch_extent_acpc = fr_patch_extent_acpc

#     return dict(
#         affine_vox2acpc=affine_vox2acpc,
#         fr_patch_vox_lu_bound=fr_patch_vox_lu_bound,
#         fr_patch_extent_acpc=fr_patch_extent_acpc,
#     )


# # Slice into fodf and full-res mask with the LR patch's spatial extent.
# def _crop_sample_fodf_mask(
#     data_dict: dict,
#     fr_patch_vox_lu_bound_key: str,
#     vol_key_map: dict,
# ):
#     # Find start and end of the ROI
#     lu_bound = data_dict[fr_patch_vox_lu_bound_key]
#     roi_start = lu_bound[:, 0]
#     roi_end = lu_bound[:, 1]
#     # Crop vols with the SpatialCrop transform.
#     cropper = monai.transforms.SpatialCropd(
#         list(vol_key_map.keys()), roi_start=roi_start, roi_end=roi_end
#     )
#     to_crop = {v: data_dict[v] for v in vol_key_map.keys()}
#     cropped = cropper(to_crop)
#     # If crops are are cube shaped, then at least one of the roi indices is either
#     # 1) < 0, or 2) > bounds of the entire volume. Check for that.
#     sample_vol = list(cropped.values())[0]
#     if not (torch.as_tensor(sample_vol.shape[1:]) == sample_vol.shape[1]).all():
#         pad_pre = torch.zeros_like(roi_start)
#         pad_pre[roi_start < 0] = torch.abs(roi_start[roi_start < 0])
#         dim_limits = torch.as_tensor(data_dict[list(cropped.keys())[0]].shape[1:]).to(
#             torch.int32
#         )

#         pad_post = torch.zeros_like(roi_end)
#         pad_post[roi_end > dim_limits] = (roi_end - dim_limits)[roi_end > dim_limits]
#         padder = monai.transforms.BorderPadd(
#             keys=list(vol_key_map.keys()),
#             spatial_border=list(
#                 itertools.chain.from_iterable(zip(pad_post.tolist(), pad_pre.tolist()))
#             ),
#             mode="constant",
#             value=0,
#         )
#         cropped = padder(cropped)
#         sample_vol = list(cropped.values())[0]
#         assert (torch.as_tensor(sample_vol.shape[1:]) == sample_vol.shape[1]).all()

#     # Store the cropped vols into the data dict with the (possibly) new keys.
#     for old_v in cropped.keys():
#         data_dict[vol_key_map[old_v]] = cropped[old_v]

#     return data_dict


class HCPINRfODFWholeVolDataset(monai.data.Dataset):
    def __init__(self, data, transform=None):
        super().__init__(data, transform=transform)
