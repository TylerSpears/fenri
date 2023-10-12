# -*- coding: utf-8 -*-
import itertools

import monai
import nibabel as nib
import numpy as np
import torch

import pitn


class TestSpatialTransformationsVsMonai:

    sample_vol_size = (10, 11, 12)
    sample_vol = torch.arange(np.prod(sample_vol_size)).reshape(sample_vol_size) + 0.01

    basic_affine = torch.Tensor(
        [
            [1, 0, 0, 5],
            [0, 1, 0, 5.5],
            [0, 0, 1, 6],
            [0, 0, 0, 1],
        ]
    )
    ornt_affines = dict()
    start_code = ("R", "A", "S")
    start_ornt = nib.orientations.axcodes2ornt(start_code)

    # Get all possible permutations of axis orientations.
    for o_x, o_y, o_z in itertools.product(("R", "L"), ("A", "P"), ("S", "I")):
        for o_x_order, o_y_order, o_z_order in itertools.permutations(
            [o_x, o_y, o_z], 3
        ):
            k = (o_x_order, o_y_order, o_z_order)
            if k not in ornt_affines:
                target_code = k
                target_ornt = nib.orientations.axcodes2ornt(target_code)
                start2target_ornt = nib.orientations.ornt_transform(
                    start_ornt, target_ornt
                )
                reoriented_vol = nib.orientations.apply_orientation(
                    sample_vol, start2target_ornt
                )
                reoriented_vol = torch.from_numpy(reoriented_vol.copy()).to(sample_vol)
                start2target_affine = np.round(
                    np.linalg.inv(
                        nib.orientations.inv_ornt_aff(
                            start2target_ornt, shape=sample_vol_size
                        )
                    ),
                    decimals=7,
                )
                start2target_affine = torch.from_numpy(start2target_affine.copy()).to(
                    basic_affine
                )
                reoriented_affine = start2target_affine @ basic_affine
                ornt_affines[k] = (reoriented_vol, reoriented_affine)

    # def test_crop_affine_space(self):
    #     dim_crops = (0, 1, 2, 3)
    #     low_high_dim_crops = itertools.product(*([dim_crops] * 6))
    #     monai_crop_fn = monai.transforms.Crop()

    #     for ornt_code, reornt_v in self.ornt_affines.items():
    #         reornt_vol, reornt_affine = reornt_v
    #         fov_x = reornt_vol.shape[0]
    #         fov_y = reornt_vol.shape[1]
    #         fov_z = reornt_vol.shape[2]

    #         s = pitn.affine.AffineSpace(
    #             reornt_affine,
    #             pitn.affine.fov_bb_coords_from_vox_shape(
    #                 reornt_affine, shape=reornt_vol.shape
    #             ),
    #         )
    #         meta_tensor = monai.data.MetaTensor(reornt_vol[None], reornt_affine)

    #         for x_low, x_high, y_low, y_high, z_low, z_high in low_high_dim_crops:

    #             pitn_crop = pitn.transforms.functional.crop_affine_space_by_vox(
    #                 s, (x_low, x_high), (y_low, y_high), (z_low, z_high)
    #             )

    #             monai_crop = monai_crop_fn(
    #                 meta_tensor,
    #                 slices=monai_crop_fn.compute_slices(
    #                     roi_start=(x_low, y_low, z_low),
    #                     roi_end=(fov_x - x_high, fov_y - y_high, fov_z - z_high),
    #                 ),
    #             )
    #             # Test resulting affines.
    #             assert torch.isclose(
    #                 pitn_crop.affine.float(), monai_crop.affine.float()
    #             ).all()

    #             # Test fov bounding box coordinates.
    #             monai_fov_low = nib.affines.apply_affine(
    #                 monai_crop.affine.numpy(), np.zeros(3).astype(float)
    #             )
    #             monai_fov_high = nib.affines.apply_affine(
    #                 monai_crop.affine.numpy(),
    #                 (np.array(monai_crop.shape[1:])).astype(float) - 1,
    #             )
    #             monai_fov = torch.stack(
    #                 [torch.from_numpy(monai_fov_low), torch.from_numpy(monai_fov_high)],
    #                 dim=0,
    #             )

    #             assert torch.isclose(
    #                 pitn_crop.fov_bb_coords.float(), monai_fov.float()
    #             ).all()

    # def test_pad_affine_space(self):
    #     dim_pads = (0, 1, 2, 3)
    #     low_high_dim_pads = itertools.product(*([dim_pads] * 6))
    #     monai_pad_fn = monai.transforms.Pad()

    #     for ornt_code, reornt_v in self.ornt_affines.items():
    #         reornt_vol, reornt_affine = reornt_v
    #         fov_x = reornt_vol.shape[0]
    #         fov_y = reornt_vol.shape[1]
    #         fov_z = reornt_vol.shape[2]

    #         s = pitn.affine.AffineSpace(
    #             reornt_affine,
    #             pitn.affine.fov_bb_coords_from_vox_shape(
    #                 reornt_affine, shape=reornt_vol.shape
    #             ),
    #         )
    #         # Expand with a channel dimension.
    #         meta_tensor = monai.data.MetaTensor(reornt_vol[None], reornt_affine)

    #         for x_low, x_high, y_low, y_high, z_low, z_high in low_high_dim_pads:

    #             pitn_pad = pitn.transforms.functional.pad_affine_space_by_vox(
    #                 s, (x_low, x_high), (y_low, y_high), (z_low, z_high)
    #             )

    #             monai_pad = monai_pad_fn(
    #                 meta_tensor,
    #                 to_pad=[(0, 0), (x_low, x_high), (y_low, y_high), (z_low, z_high)],
    #             )
    #             # Test resulting affines.
    #             assert torch.isclose(
    #                 pitn_pad.affine.float(), monai_pad.affine.float()
    #             ).all()

    #             # Test fov bounding box coordinates.
    #             monai_fov_low = nib.affines.apply_affine(
    #                 monai_pad.affine.numpy(), np.zeros(3).astype(float)
    #             )
    #             monai_fov_high = nib.affines.apply_affine(
    #                 monai_pad.affine.numpy(),
    #                 (np.array(monai_pad.shape[1:])).astype(float) - 1,
    #             )
    #             monai_fov = torch.stack(
    #                 [torch.from_numpy(monai_fov_low), torch.from_numpy(monai_fov_high)],
    #                 dim=0,
    #             )

    #             assert torch.isclose(
    #                 pitn_pad.fov_bb_coords.float(), monai_fov.float()
    #             ).all()

    def test_reorient_vol_axes(self):
        start_vol = self.sample_vol
        start_affine = self.basic_affine
        s_start = pitn.affine.AffineSpace(
            start_affine,
            pitn.affine.fov_bb_coords_from_vox_shape(
                start_affine, shape=start_vol.shape
            ),
        )

        for reornt_code, reornt_v in self.ornt_affines.items():
            reornt_vol, reornt_affine = reornt_v

            s_reoriented = pitn.transforms.functional.reorient_affine_space(
                s_start, "".join(reornt_code)
            )

            # Test resulting affines.
            assert torch.isclose(
                s_reoriented.affine.float(), reornt_affine.float()
            ).all()

            # Test fov bounding box coordinates.
            gt_fov_low = nib.affines.apply_affine(
                reornt_affine.numpy(), np.zeros(3).astype(float)
            )
            gt_fov_high = nib.affines.apply_affine(
                reornt_affine.numpy(),
                (np.array(start_vol.shape)).astype(float) - 1,
            )
            gt_fov = torch.stack(
                [torch.from_numpy(gt_fov_low), torch.from_numpy(gt_fov_high)],
                dim=0,
            )
            assert torch.isclose(
                s_reoriented.fov_bb_coords.float(), gt_fov.float()
            ).all()
