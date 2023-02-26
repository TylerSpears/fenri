# -*- coding: utf-8 -*-
import collections
import itertools
import math
from ast import Param
from typing import List, Optional, Sequence, Union

import dipy
import dipy.core
import dipy.reconst
import dipy.reconst.dti
import dipy.segment.mask
import einops
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchio
from box import Box
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from pitn._lazy_loader import LazyLoader

# Make pitn lazy load to avoid circular imports.
pitn = LazyLoader("pitn", globals(), "pitn")


def plot_fodf_3d(theta, phi, sphere_vals, fig=None, **plot_trisurf_kwargs):
    theta = torch.Tensor(theta)
    phi = torch.Tensor(phi)
    sphere_vals = torch.Tensor(sphere_vals)
    if (
        (theta.ndim > 1 and (theta.numel() / theta.shape[-1]) > 1)
        or (phi.ndim > 1 and (phi.numel() / phi.shape[-1]) > 1)
        or (sphere_vals.ndim > 1 and (sphere_vals.numel() / sphere_vals.shape[-1]) > 1)
    ):
        raise ValueError(
            "Plotting can only accept 1 sphere to plot, got",
            f"{tuple(theta.shape)}, {tuple(phi.shape)}, {tuple(sphere_vals.shape)}",
        )
    elif (theta.numel() != phi.numel()) or (phi.numel() != sphere_vals.numel()):
        raise ValueError(
            "Coordinates and function values must have the same number of elements",
            f"got {theta.numel()}, {phi.numel()}, {sphere_vals.numel()}",
        )

    vals = sphere_vals.detach().cpu().numpy().flatten()
    r = (vals - vals.min()) / (vals - vals.min()).max()
    r = vals / vals.sum()

    zyx = pitn.tract.local.unit_sphere2zyx(theta, phi)
    x = r * zyx[:, 2].detach().cpu().numpy().flatten()
    y = r * zyx[:, 1].detach().cpu().numpy().flatten()
    z = r * zyx[:, 0].detach().cpu().numpy().flatten()
    theta = theta.detach().cpu().numpy().flatten()
    phi = phi.detach().cpu().numpy().flatten()

    if fig is None:
        fig = plt.figure(dpi=120)
    ax = fig.add_subplot(projection="3d")
    tri = mpl.tri.Triangulation(phi, theta)
    ax.plot_trisurf(
        x,
        y,
        z,
        triangles=tri.triangles,
        **{**dict(cmap="gnuplot", alpha=1, linewidth=0), **plot_trisurf_kwargs},
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return fig


def plot_sphere_fn_vals(
    theta, phi, fn_vals, subplots_kwargs: dict = dict(), **scatter_kwargs
):
    zyx = pitn.tract.local.unit_sphere2zyx(theta, phi)
    x = zyx[:, 2].detach().cpu().numpy().flatten()
    y = zyx[:, 1].detach().cpu().numpy().flatten()
    z = zyx[:, 0].detach().cpu().numpy().flatten()
    vals = fn_vals.detach().cpu().numpy().flatten()
    vmax = vals.max()
    vmin = -vmax
    fig, axs = plt.subplots(
        nrows=1, ncols=2, **{**{"dpi": 120, "figsize": (7, 3.5)}, **subplots_kwargs}
    )
    ax = axs[0]
    distance_from_xy_plane_vals = np.copy(vals)
    distance_from_xy_plane_vals[z < 0] = -vals[z < 0]
    ax.scatter(
        x,
        y,
        c=distance_from_xy_plane_vals,
        **{**{"vmin": vmin, "vmax": vmax, "cmap": "coolwarm"}, **scatter_kwargs},
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax = axs[1]
    distance_from_xz_plane_vals = np.copy(vals)
    distance_from_xz_plane_vals[y < 0] = -vals[y < 0]
    ax.scatter(
        x,
        z,
        c=distance_from_xz_plane_vals,
        **{**{"vmin": vmin, "vmax": vmax, "cmap": "coolwarm"}, **scatter_kwargs},
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Z")

    return fig


def plot_im_grid(
    *ims,
    nrows: int = 3,
    title: Optional[str] = None,
    row_headers: Optional[List[str]] = None,
    col_headers: Optional[List[str]] = None,
    colorbars: Optional[str] = None,
    fig=None,
    **imshow_kwargs,
):
    """Plot sequence of 2d arrays as a grid of images with optional titles & colorbars.

    Parameters
    ----------
    ims: sequence
        Sequence of numpy ndarrays or pytorch Tensors to plot into a grid.
    nrows : int, optional
        Number of rows in the grid, by default 3
    title : Optional[str], optional
        The `suptitle` of the image grid, by default None
    row_headers : Optional[List[str]], optional
        Titles for each row, by default None
    col_headers : Optional[List[str]], optional
        Titles for each column., by default None
    colorbars : Optional[str], optional
        Set the type of colorbar and intensity normalization to use, by default None

        Valid options are:
            None - no colorbar or intensity normalization.
            "global" - one colorbar is created for the entire grid, and all images are
                normalized to have color intensity ranges match.
            "each" - every image has its own colorbar with no intensity normalization.
            "col", "cols", "column", "columns" - Every column is normalized and
                given a colorbar.
            "row", "rows" - Every row is normalized and given a colorbar.

    fig : Figure, optional
        Figure to plot into, by default None
    imshow_kwargs : dict
        Kwargs to pass to the `.imshow()` function call of each image.

    Returns
    -------
    Figure

    Raises
    ------
    ValueError
        Invalid option value for `colorbars`
    """

    # AX_TITLE_SIZE_PERC = 0.05
    # SUPTITLE_SIZE_PERC = 0.1
    # AX_CBAR_SIZE_PERC = 0.1
    # EACH_CBAR_SUBPLOT_SIZE_PERC = "7%"

    if fig is None:
        fig = plt.gcf()

    ims = list(ims)
    # Canonical form of ims.
    if len(ims) == 1 and isinstance(ims[0], (list, tuple)):
        ims = list(ims[0])
    elif len(ims) == 1 and (isinstance(ims[0], np.ndarray) or torch.is_tensor(ims[0])):
        # If the tensor/ndarray is batched.
        if len(ims.shape) == 4:
            ims = list(ims)
        else:
            ims = [
                ims,
            ]
    for i, im in enumerate(ims):
        if torch.is_tensor(im):
            ims[i] = im.detach().cpu().numpy()
        ims[i] = ims[i].astype(float)
    ncols = math.ceil(len(ims) / nrows)
    # Canonical representation of image labels.
    row_headers = (
        row_headers if row_headers is not None else list(itertools.repeat(None, nrows))
    )
    col_headers = (
        col_headers if col_headers is not None else list(itertools.repeat(None, ncols))
    )

    if len(row_headers) != nrows:
        raise RuntimeError(
            f"ERROR: Number of row headers {len(row_headers)} != number of rows {nrows}"
        )
    if len(col_headers) != ncols:
        raise RuntimeError(
            f"ERROR: Number of row headers {len(col_headers)} != number of rows {ncols}"
        )
    # Canonical colorbar setting values.
    cbars = colorbars.casefold() if colorbars is not None else colorbars
    cbars = "col" if cbars in {"column", "columns", "cols", "col"} else cbars
    cbars = "row" if cbars in {"row", "rows"} else cbars
    if cbars not in {"global", "each", "row", "col", None}:
        raise ValueError(f"ERROR: Colorbars value {colorbars} not valid.")

    # Pad im list with None objects.
    pad_ims = list(
        itertools.islice(itertools.chain(ims, itertools.repeat(None)), nrows * ncols)
    )
    ims_grid = list()
    for i in range(0, nrows * ncols, ncols):
        ims_grid.append(pad_ims[i : i + ncols])

    # Calculate grid shape in number of pixels/array elements in both directions.
    col_widths = [
        sum(map(lambda im: im.shape[1] if im is not None else 1, l)) for l in ims_grid
    ]
    # Bring row elements close together.
    rows = [l[i] for i in range(ncols) for l in itertools.chain(ims_grid)]
    # Group row elements into their own lists.
    rows = [rows[s : s + ncols] for s in range(0, nrows * ncols, ncols)]
    row_heights = [
        sum(map(lambda im: im.shape[0] if im is not None else 1, r)) for r in rows
    ]

    # Correct fig size according to the size of the actual arrays.
    grid_pix_dim = (max(row_heights), max(col_widths))
    fig_hw_ratio = grid_pix_dim[0] / grid_pix_dim[1]
    fig.set_figheight(fig.get_figwidth() * fig_hw_ratio)

    # Create gridspec.
    grid = mpl.gridspec.GridSpec(
        nrows=nrows,
        ncols=ncols,
        figure=fig,
        # width_ratios=np.asarray(col_widths) / sum(col_widths),
        # height_ratios=np.asarray(row_heights) / sum(row_heights),
        left=0.05,
        right=0.95,
        top=0.95,
        bottom=0.05,
        wspace=0.01,
        hspace=0.01,
    )

    # Keep track of each image's min and max values.
    min_max_vals = np.zeros((nrows, ncols, 2))
    # Keep track of each created axis in the grid.
    axs = list()
    # Keep track of the highest subplot position in order to avoid overlap with the
    # suptitle.
    max_subplot_height = 0
    # Step through the grid.
    for i_row, (ims_row_i, row_i_header) in enumerate(zip(ims_grid, row_headers)):
        row_axs = list()
        for j_col, (im, col_j_header) in enumerate(zip(ims_row_i, col_headers)):
            # If no im was given here, skip everything in this loop.
            if im is None:
                continue

            # Create Axes object at the grid location.
            ax = fig.add_subplot(grid[i_row, j_col])
            ax.imshow(im, **imshow_kwargs)
            # Set headers.
            if row_i_header is not None and ax.get_subplotspec().is_first_col():
                ax.set_ylabel(row_i_header)
            if col_j_header is not None and ax.get_subplotspec().is_first_row():
                ax.set_xlabel(col_j_header)
                ax.xaxis.set_label_position("top")
            # Remove pixel coordinate axis ticks.
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect("equal")
            # Update highest subplot to put the `suptitle` later on.
            max_subplot_height = max(
                max_subplot_height, ax.get_position(original=False).get_points()[1, 1]
            )
            # Update min and max im values.
            min_max_vals[i_row, j_col] = (im.min(), im.max())
            row_axs.append(ax)
        axs.append(row_axs)

    # Handle colorbar creation.
    # If there should only be one colorbar for the entire grid.
    if cbars == "global":
        min_v = min_max_vals[:, :, 0].min()
        max_v = min_max_vals[:, :, 1].max()
        color_norm = mpl.colors.Normalize(vmin=min_v, vmax=max_v)
        cmap = None
        for ax_row in axs:
            for ax in ax_row:
                disp_im = ax.get_images()[0]
                disp_im.set(norm=color_norm)
                cmap = disp_im.cmap if cmap is None else cmap

        fig.colorbar(
            mpl.cm.ScalarMappable(norm=color_norm, cmap=cmap),
            ax=list(itertools.chain.from_iterable(axs)),
            location="right",
            fraction=0.1,
            pad=0.03,
        )
    # If colorbar setting is row, col, or each.
    elif cbars is not None:
        # Step through all subplots in the grid.
        for i_row, ax_row in enumerate(axs):
            for j_col, ax in enumerate(ax_row):
                # Determine value range depending on the cbars setting.
                if cbars == "row":
                    min_v = min_max_vals[i_row, :, 0].min()
                    max_v = min_max_vals[i_row, :, 1].max()
                elif cbars == "col":
                    min_v = min_max_vals[:, j_col, 0].min()
                    max_v = min_max_vals[:, j_col, 1].max()
                elif cbars == "each":
                    min_v = min_max_vals[i_row, j_col, 0]
                    max_v = min_max_vals[i_row, j_col, 1]
                else:
                    raise RuntimeError(f"ERROR: Invalid option {cbars}")

                # Get AxesImage object (the actual image plotted) from the subplot.
                disp_im = ax.get_images()[0]
                # Set up color/cmap scaling.
                color_norm = mpl.colors.Normalize(vmin=min_v, vmax=max_v)
                disp_im.set(norm=color_norm)
                color_mappable = mpl.cm.ScalarMappable(
                    norm=color_norm, cmap=disp_im.cmap
                )

                # Use the (somewhat new) AxesDivider utility to divide the subplot
                # and add a colorbar with the corresponding min/max values.
                # See
                # <https://matplotlib.org/stable/gallery/axes_grid1/demo_colorbar_with_axes_divider.html>
                if cbars == "row":
                    if ax.get_subplotspec().is_last_col():
                        ax_div = make_axes_locatable(ax)
                        cax = ax_div.append_axes("right", size="7%", pad="4%")
                        fig.colorbar(color_mappable, cax=cax, orientation="vertical")
                elif cbars == "col":
                    if ax.get_subplotspec().is_last_row():
                        ax_div = make_axes_locatable(ax)
                        cax = ax_div.append_axes("bottom", size="7%", pad="4%")
                        cbar = fig.colorbar(
                            color_mappable,
                            cax=cax,
                            orientation="horizontal",
                            format="%.4e",
                        )
                        cax.xaxis.set_ticks_position("bottom")
                        cbar.ax.tick_params(rotation=90)
                elif cbars == "each":
                    ax_div = make_axes_locatable(ax)
                    cax = ax_div.append_axes("right", size="7%", pad="4%")
                    fig.colorbar(color_mappable, cax=cax, orientation="vertical")

    if title is not None:
        fig.suptitle(title, y=max_subplot_height + 0.05, verticalalignment="bottom")

    return fig


def plot_vol_slices(
    *vols: Sequence[Union[torch.Tensor, np.ndarray]],
    slice_idx=(0.5, 0.5, 0.5),
    title: Optional[str] = None,
    vol_labels: Optional[List[str]] = None,
    slice_labels: Optional[List[str]] = None,
    channel_labels: Optional[List[str]] = None,
    colorbars: Optional[str] = None,
    fig=None,
    **imshow_kwargs,
):

    # Canonical format of vols.
    # Enforce a B x C x D x H x W shape.
    bcdwh_vols = list()
    for vol in vols:
        if len(vol.shape) == 3:
            vol = vol.reshape(1, *vol.shape)
        if len(vol.shape) == 4:
            vol = vol.reshape(1, *vol.shape)
        bcdwh_vols.append(vol)
    # Flatten into a list of C x ... arrays.
    bcdwh_vols = list(itertools.chain.from_iterable(bcdwh_vols))

    row_slice_by_vol_labels = list()
    flat_slices = list()
    for i_b, chan_v in enumerate(bcdwh_vols):
        for k_s, s in enumerate(slice_idx):
            # Use None as a sentinal to only create a new row label for every
            # (vol x slice) pairing, disregarding the channel index.
            row_label = None
            # Need channel index to be the inner-most loop for plotting.
            for v in chan_v:
                # If slice idx was None, skip this slice.
                if s is None:
                    continue
                # If slice idx was a fraction, interpret that as a percent of the total
                # dim size.
                if isinstance(s, float) and s >= 0.0 and s <= 1.0:
                    idx = math.floor(s * v.shape[k_s])
                else:
                    idx = math.floor(s)

                # Generate the slice(None) objects that follow the integer index.
                slice_after = tuple(
                    itertools.repeat(slice(None), len(slice_idx) - (k_s + 1))
                )
                slicer = (
                    ...,
                    idx,
                ) + slice_after
                vol_slice = v[slicer]

                if torch.is_tensor(vol_slice):
                    vol_slice = vol_slice.detach().cpu().numpy()
                flat_slices.append(vol_slice)

                # Handle labelling of the rows, only one label per row.
                if row_label is None:
                    row_label = ""
                    if vol_labels is not None:
                        row_label = row_label + vol_labels[i_b] + " "
                    if slice_labels is not None:
                        row_label = row_label + slice_labels[k_s]

                    row_slice_by_vol_labels.append(row_label.strip())

    maybe_empty_row_vol_labels = (
        None
        if all(map(lambda s: s == "", row_slice_by_vol_labels))
        else row_slice_by_vol_labels
    )

    return plot_im_grid(
        *flat_slices,
        nrows=len(row_slice_by_vol_labels),
        title=title,
        row_headers=maybe_empty_row_vol_labels,
        col_headers=channel_labels,
        colorbars=colorbars,
        fig=fig,
        **imshow_kwargs,
    )


# Create FA map from DTI's
def fa_map(dti, channels_first=True) -> np.ndarray:
    if torch.is_tensor(dti):
        t = dti.cpu().numpy()
    else:
        t = np.asarray(dti)
    # Reshape to work with dipy.
    if channels_first:
        t = einops.rearrange(t, "c ... -> ... c")

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
        t = einops.rearrange(t, "c ... -> ... c")

    # Re-create the symmetric DTI's (3x3) from the lower-triangular portion (6).
    t = dipy.reconst.dti.from_lower_triangular(t)
    eigvals, eigvecs = dipy.reconst.dti.decompose_tensor(t)

    fa = dipy.reconst.dti.fractional_anisotropy(eigvals)
    direction_map = dipy.reconst.dti.color_fa(fa, eigvecs)

    if channels_first:
        direction_map = einops.rearrange(direction_map, "... c -> c ...")

    return direction_map


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


def plot_fodf_coeff_slices(
    *fodf_vols,
    fig,
    rect=111,
    vol_slice_idx_as_proportions=(0.5, 0.5, 0.5),
    fodf_coeff_idx=(0, 3, 10, 21, 36),
    fodf_vol_labels=None,
    imshow_kwargs: dict = dict(),
    image_grid_kwargs: dict = dict(),
):
    vols = list()
    for v in fodf_vols:
        v = v.detach().cpu()
        if v.ndim == 5 and int(v.shape[0]) == 1:
            v = v[0]
        v = v.numpy()
        vols.append(v)
    n_vols = len(vols)
    if fodf_vol_labels is None:
        vol_labels = list(itertools.repeat(None, n_vols))
    else:
        vol_labels = list(fodf_vol_labels)
    # n_fod_coeffs = int(vols[0].shape[0])
    n_slices = len(list(filter(lambda x: x is not None, vol_slice_idx_as_proportions)))
    n_fod_coeffs_to_plot = len(fodf_coeff_idx)
    image_grid_kwargs = {
        "fig": fig,
        "rect": rect,
        "nrows_ncols": (n_fod_coeffs_to_plot, n_vols * n_slices),
        "label_mode": "1",
        "cbar_mode": "edge",
        "cbar_location": "right",
        "cbar_size": "9%",
    } | image_grid_kwargs

    imshow_kwargs = {"cmap": "gray", "interpolation": "antialiased"} | imshow_kwargs

    grid = ImageGrid(**image_grid_kwargs)

    row_order_grid_idx = 0
    for i_fodf_coeff_idx, fodf_coeff_idx in enumerate(fodf_coeff_idx):
        for j_vol, vol in enumerate(vols):
            vol_label = vol_labels[j_vol]
            for k_slice, slice_idx_prop in enumerate(vol_slice_idx_as_proportions):
                if slice_idx_prop is None:
                    continue
                shape = tuple(vol.shape[1:])
                vol_slice_idx = round(shape[k_slice] * slice_idx_prop)
                slicer = [slice(None), slice(None), slice(None)]
                slicer[k_slice] = vol_slice_idx
                slicer = [fodf_coeff_idx] + slicer
                slicer = tuple(slicer)
                im = vol[slicer]

                ax = grid[row_order_grid_idx]
                ax.imshow(im, **imshow_kwargs)
                ax.set_xticks([])
                ax.set_yticks([])
                if vol_label is not None and i_fodf_coeff_idx == 0:
                    ax.set_title(vol_label)

                row_order_grid_idx += 1

    return fig, grid


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
