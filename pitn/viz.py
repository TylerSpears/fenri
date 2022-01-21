# -*- coding: utf-8 -*-
import collections

import addict
from addict import Addict
from box import Box

import numpy as np
import torch
import torchio
import pandas as pd

import dipy
import dipy.core
import dipy.reconst
import dipy.reconst.dti
import dipy.segment.mask

import matplotlib.pyplot as plt
import seaborn as sns

# Create FA map from DTI's
def fa_map(dti, channels_first=True) -> np.ndarray:
    if torch.is_tensor(dti):
        t = dti.cpu().numpy()
    else:
        t = np.asarray(dti)
    # Reshape to work with dipy.
    if channels_first:
        t = t.transpose(1, 2, 3, 0)

    # Re-create the symmetric DTI's (3x3) from the lower-triangular portion (6).
    t = dipy.reconst.dti.from_lower_triangular(t)
    eigvals, eigvecs = dipy.reconst.dti.decompose_tensor(t)

    fa = dipy.reconst.dti.fractional_anisotropy(eigvals)

    return fa


# Generate FA-weighted diffusion direction map.
def direction_map(dti, channels_first=True) -> np.ndarray:

    if torch.is_tensor(dti):
        t = dti.cpu().numpy()
    else:
        t = np.asarray(dti)
    # Reshape to work with dipy.
    if channels_first:
        t = t.transpose(1, 2, 3, 0)

    # Re-create the symmetric DTI's (3x3) from the lower-triangular portion (6).
    t = dipy.reconst.dti.from_lower_triangular(t)
    eigvals, eigvecs = dipy.reconst.dti.decompose_tensor(t)

    fa = dipy.reconst.dti.fractional_anisotropy(eigvals)
    direction_map = dipy.reconst.dti.color_fa(fa, eigvecs)

    if channels_first:
        return direction_map.transpose(3, 0, 1, 2)

    return direction_map


# Return type wrapper
MultiresGridSample = collections.namedtuple(
    "MultiresGridSample",
    ("low_res", "full_res", "full_res_locations", "full_res_masks", "subj_ids"),
)

# Collate function for the DataLoader to combine multiple samples with locations.
def collate_locations(
    samples,
    full_res_key: str,
    low_res_key: str,
    subj_id_key: str,
    full_res_mask_key: str = None,
):
    full_res_stack = torch.stack([subj[full_res_key].data for subj in samples])
    full_res_location_stack = torch.stack(
        [torch.as_tensor(subj["location"]) for subj in samples]
    )
    # Assume the low-res data are dicts, not `torchio.Image`'s
    low_res_stack = torch.stack([subj[low_res_key]["data"] for subj in samples])

    # Grab mask tensors, if requested.
    if full_res_mask_key is not None:
        fr_masks = torch.stack(
            [torch.as_tensor(subj[full_res_mask_key]) for subj in samples]
        )
        fr_masks = fr_masks.bool()
    else:
        fr_masks = torch.ones_like(full_res_stack).bool()

    # If the masks do not have a channel dimension, add one right after the batch
    # dimension.
    if fr_masks.ndim == 4:
        fr_masks = fr_masks[:, None, ...]

    # Grab subject IDs
    subj_ids = torch.as_tensor(
        [subj[subj_id_key] for subj in samples], dtype=torch.int
    ).to(full_res_stack.device)

    return MultiresGridSample(
        low_res=low_res_stack,
        full_res=full_res_stack,
        full_res_locations=full_res_location_stack,
        full_res_masks=fr_masks,
        subj_ids=subj_ids,
    )


def collate_locs_and_keys(
    samples,
    full_res_key: str,
    low_res_key: str,
    full_res_mask_key: str = None,
    **extra_keys,
) -> Addict:

    # # Return type wrapper
    # MultiKeySample = collections.namedtuple(
    #     "MultiKeySample",
    #     ("low_res", "full_res", "full_res_masks", "full_res_locs", *extra_keys.keys()),
    # )

    # Grab the mandatory and specialized values first.
    full_res_stack = torch.stack([subj[full_res_key][torchio.DATA] for subj in samples])
    full_res_location_stack = torch.stack(
        [torch.as_tensor(subj[torchio.LOCATION]) for subj in samples]
    )
    # Assume the low-res data are dicts, not `torchio.Image`'s
    low_res_stack = torch.stack([subj[low_res_key]["data"] for subj in samples])

    # Grab mask tensors, if requested.
    if full_res_mask_key is not None:
        fr_masks = torch.stack(
            [torch.as_tensor(subj[full_res_mask_key]) for subj in samples]
        )
        fr_masks = fr_masks.bool()
    else:
        fr_masks = torch.ones_like(full_res_stack).bool()

    # If the masks do not have a channel dimension, add one right after the batch
    # dimension.
    if fr_masks.ndim == 4:
        fr_masks = fr_masks[:, None, ...]

    specified_keys = dict()
    for k_new, k_old in extra_keys.items():
        v = [subj[k_old] for subj in samples]
        try:
            v = torch.as_tensor(v).to(full_res_stack.device)
        except ValueError:
            pass

        specified_keys[k_new] = v

    return dict(
        low_res=low_res_stack,
        full_res=full_res_stack,
        full_res_masks=fr_masks,
        full_res_locs=full_res_location_stack,
        **specified_keys,
    )


def make_grid(
    tensor, nrow=8, padding=2, pad_value=0, normalize=False, vmin=None, vmax=None
):
    """Create grid of 2D imager for visualization.

    tensor:
        Sequence of 2-dimensional pytorch Tensors of the *same shape*.

        Each element of `tensor` is assumed to have dimensions `H x W`.

    Based on implementation of `torchvision.utils.make_grid`.
    """
    if torch.is_tensor(tensor):
        tensor = list(tensor)
    assert all(
        [
            len(t.shape)
            in {
                2,
            }
            for t in tensor
        ]
    )

    num_imgs = len(tensor)
    ncols = np.ceil(num_imgs / nrow).astype(int)

    img_w = tensor[0].shape[-1]
    img_h = tensor[0].shape[-2]
    total_pix_w = ((img_w + padding) * ncols) + padding
    total_pix_h = ((img_h + padding) * nrow) + padding

    grid = torch.ones(total_pix_h, total_pix_w) * pad_value

    curr_img_idx = 0
    # Iterate over rows.
    for i_row, start_y in enumerate(range(padding, grid.shape[0], padding + img_h)):
        # Iterate over columns.
        for j_col, start_x in enumerate(range(padding, grid.shape[1], padding + img_w)):
            try:
                grid[start_y : (start_y + img_h), start_x : (start_x + img_w)] = (
                    tensor[curr_img_idx].cpu().to(grid)
                )
                curr_img_idx += 1
            except IndexError:
                break

    if normalize:
        vmin = grid.min() if vmin is None else vmin
        vmax = grid.max() if vmax is None else vmax
        # Normalize values in grid without intermediary copies.
        grid.sub_(vmin).div_(max(vmax - vmin, 1e-5))

    return grid


class SubGridAggregator(torchio.GridAggregator):
    def __init__(
        self,
        spatial_shape,
        patch_overlap,
        location_offset=0,
        overlap_mode: str = "crop",
    ):

        if np.isscalar(patch_overlap):
            patch_overlap = (patch_overlap,) * 3
        self.patch_overlap = torch.as_tensor(patch_overlap)
        if (self.patch_overlap == 0).all():
            self.volume_padded = False
        else:
            self.volume_padded = True

        self.spatial_shape = spatial_shape
        if np.isscalar(location_offset):
            location_offset = (location_offset,) * 3
        self.location_offset = torch.as_tensor(location_offset)

        self._output_tensor = None
        self.parse_overlap_mode(overlap_mode)
        self.overlap_mode = overlap_mode
        self._avgmask_tensor = None

    def add_batch(self, batch_tensor: torch.Tensor, locations: torch.Tensor) -> None:
        adjusted_locs = locations.clone()
        adjusted_locs[:, :3] = adjusted_locs[:, :3] - self.location_offset
        adjusted_locs[:, 3:] = adjusted_locs[:, 3:] - self.location_offset

        return super().add_batch(batch_tensor, adjusted_locs)


def plot_dti_box_row(
    fig,
    grid,
    row_idx: int,
    subj_id: int,
    shared_axs_rows: list,
    shared_axs_cols: list,
    fr_vol: np.ndarray,
    lr_vol: np.ndarray,
    colors: list = list(sns.color_palette("Set2", n_colors=2)),
):

    dti_channel_names = [
        "$D_{xx}$",
        "$D_{xy}$",
        "$D_{yy}$",
        "$D_{xz}$",
        "$D_{yz}$",
        "$D_{zz}$",
    ]

    for i_channel, channel_name in enumerate(dti_channel_names):
        cell = grid[row_idx, i_channel]

        ax = fig.add_subplot(
            cell,
            sharex=shared_axs_cols[channel_name],
            sharey=shared_axs_rows[subj_id],
        )
        if shared_axs_cols[channel_name] is None:
            shared_axs_cols[channel_name] = ax
        if shared_axs_rows[subj_id] is None:
            shared_axs_rows[subj_id] = ax

        #         quantile_outlier_cutoff = (0.1, 0.9)
        fr_channel = fr_vol[i_channel]
        #         fr_nn = fr_nn[
        #             (np.quantile(fr_nn, quantile_outlier_cutoff[0]) <= fr_nn)
        #             & (fr_nn <= np.quantile(fr_nn, quantile_outlier_cutoff[1]))
        #         ]
        lr_channel = lr_vol[i_channel]
        #         lr_nn = lr_nn[
        #             (np.quantile(lr_nn, quantile_outlier_cutoff[0]) <= lr_nn)
        #             & (lr_nn <= np.quantile(lr_nn, quantile_outlier_cutoff[1]))
        #         ]
        #         fr_norm = normed_fr_vol[i_channel].detach().cpu().numpy()
        #         lr_norm = normed_lr_vol[i_channel].detach().cpu().numpy()

        num_fr_vox = len(fr_channel)
        num_lr_vox = len(lr_channel)

        resolution_labels = (["FR",] * num_fr_vox) + (
            [
                "LR",
            ]
            * num_lr_vox
        )

        df = pd.DataFrame(
            {
                "data": np.concatenate([fr_channel, lr_channel]),
                "resolution": resolution_labels,
            }
        )

        sns.boxenplot(
            data=df,
            y="resolution",
            x="data",
            orient="h",
            ax=ax,
            palette=colors,
            k_depth="proportion",
            outlier_prop=0.11,
            showfliers=False,
        )

        if not cell.is_last_row():
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            plt.setp(ax.get_xticklabels(), fontsize="x-small", rotation=25)

        if not cell.is_first_col():
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylabel("")
        else:
            ax.set_ylabel(subj_id)

        ax.set_xlabel("")
        if cell.is_first_row():
            ax.set_title(channel_name)

    return fig, shared_axs_rows, shared_axs_cols
