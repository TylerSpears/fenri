# -*- coding: utf-8 -*-
import collections

import numpy as np
import torch

BoxplotStats = collections.namedtuple(
    "BoxplotStats",
    ["low_outliers", "low", "q1", "median", "q3", "high", "high_outliers"],
)


def batch_boxplot_stats(batch):
    """Quick calculation of a batch of 1D values for showing boxplot stats."""
    q1, median, q3 = np.quantile(batch, q=[0.25, 0.5, 0.75], axis=1)
    iqr = q3 - q1
    low = q1 - (1.5 * iqr)
    high = q3 + (1.5 * iqr)
    low_outliers = list()
    high_outliers = list()
    # Number of outliers may be different for each batch, so it needs to be a list of
    # arrays.
    for i_batch in range(len(batch)):
        batch_i = batch[i_batch]
        low_i = low[i_batch]
        low_outliers.append(batch_i[np.where(batch_i < low_i)])
        high_i = high[i_batch]
        high_outliers.append(batch_i[np.where(batch_i > high_i)])

    return BoxplotStats(low_outliers, low, q1, median, q3, high, high_outliers)


# Quick check on full volume/batch distributions.
def desc_channel_dists(vols, mask=None):
    t_vols = torch.as_tensor(vols)

    if t_vols.ndim == 4:
        t_vols = t_vols[None, ...]

    if mask is not None:
        t_mask = torch.as_tensor(mask)
        if mask.ndim == 4:
            mask = mask[0]
    else:
        t_mask = torch.ones_like(t_vols[0, 0]).bool()

    results = "means | vars\n"
    for t_vol_i in t_vols:
        masked_vol = torch.masked_select(t_vol_i, t_mask).reshape(t_vol_i.shape[0], -1)
        mean_i = torch.mean(masked_vol, dim=1)
        var_i = torch.var(masked_vol, dim=1)
        mvs = [
            (f"{mv[0]} | {mv[1]}\n")
            for mv in torch.stack([mean_i, var_i], dim=-1).tolist()
        ]
        results = results + "".join(mvs)
        results = results + ("=" * (len(mvs[-1]) - 1)) + "\n"

    return results
