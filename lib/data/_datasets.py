# -*- coding: utf-8 -*-
import numpy as np
import torch
import monai


class MaskFilteredPatchDataset3d(torch.utils.data.Dataset):
    def __init__(self, img: torch.Tensor, mask: torch.Tensor, patch_size: tuple):
        """Dataset of static patches where the center voxels lies in the foreground/mask.

        Parameters
        ----------
        img : torch.Tensor
            Source image, of shape [C x H x W x D]
        mask : torch.Tensor
            Mask that corresponds to the target locations in the img, of shape [H x W x D]
        patch_size : tuple

        NOTE In the case of an odd-valued patch shape, the "left" side is rounded down
            in size, and the "right" side is rounded up.
        """

        super().__init__()
        self.patch_size = monai.utils.misc.ensure_tuple_rep(patch_size, 3)
        self.img = img

        if mask.ndim == 4:
            mask = mask[0]
        if self.img.ndim == 3:
            self.img = self.img[None, ...]

        # Grab all voxel coordinates in the given mask.
        patch_centers = torch.stack(torch.where(mask))
        # Shave off patch centers that would have a patch go out of bounds of the image.
        for i_dim, patch_dim_size in enumerate(self.patch_size):
            offset_lower = int(np.floor(patch_dim_size / 2))
            offset_upper = int(np.ceil(patch_dim_size / 2))
            patch_centers = patch_centers[
                :,
                (patch_centers[i_dim] >= offset_lower)
                & (patch_centers[i_dim] <= img.shape[i_dim + 1] - offset_upper),
            ]

        # Switch to the starting index per-patch, for easier sampling.
        self.patch_starts = (
            patch_centers
            - torch.floor(torch.as_tensor(self.patch_size)[:, None] / 2).int()
        ).T

    def __len__(self):
        return len(self.patch_starts)

    def __getitem__(self, index):
        patch_start = self.patch_starts[index]
        patch = self.img[
            :,
            patch_start[0] : patch_start[0] + self.patch_size[0],
            patch_start[1] : patch_start[1] + self.patch_size[1],
            patch_start[2] : patch_start[2] + self.patch_size[2],
        ]
        return patch
