# -*- coding: utf-8 -*-
import torch

# Gradient direction table for Human Connectome Project (HCP) dMRI (WU-Minn
# consortium). Each table acquired once with right-to-left and left-to-right
# phase encoding polarities, using the CMRR multiband diffusion sequence,
# with bmax = 3000 s/mm^2.  Shells at 1000, 2000, and 3000 s/mm^2 are
# obtained through appropriate weighting of the vector norm.
#
# The diffusion directions were obtained using a toolbox available from
# INRIA that returns uniformly distributed directions in multiple q-space
# shells.  The directions are optimized so that every subset of the first M
# directions is also isotropic. References and the INRIA toolbox can be
# found at:
# http://www-sop.inria.fr/members/Emmanuel.Caruyer/q-space-sampling.php
#
# NOTE: These gradient directions were tested for use on the HCP customized
# Skyra scanner with 100 mT/m gradients and may not be optimal for use with
# other scanner systems.
#
# Further questions can be directed to the HCP Data Users mailing list
# (hcp-user@humanconnectome.org) by signing up at
# http://www.humanconnectome.org/contact

HCP_STANDARD_3T_BVEC = torch.tensor(
    [
        # segment 1/3 with interleaved shells. Interleaved bval ratio 1:2:3
        # [directions=95]
        # CoordinateSystem = xyz
        # Normalisation = none
        (0.000, 0.000, 0.000),
        (-0.533, 0.147, -0.167),
        (-0.303, -0.642, 0.404),
        (0.106, -0.563, -0.819),
        (0.110, 0.550, 0.137),
        (0.430, -0.713, 0.554),
        (0.590, -0.467, -0.317),
        (-0.193, -0.092, 0.536),
        (0.483, 0.180, 0.633),
        (0.898, 0.410, -0.161),
        (-0.106, 0.462, -0.330),
        (-0.212, 0.048, 0.787),
        (-0.852, 0.483, 0.203),
        (-0.234, -0.234, -0.473),
        (0.795, 0.184, 0.027),
        (0.281, 0.960, -0.011),
        (0.000, 0.000, 0.000),
        (-0.519, -0.143, 0.208),
        (-0.356, 0.686, -0.264),
        (0.102, 0.251, -0.963),
        (-0.753, -0.183, -0.632),
        (-0.318, 0.393, 0.280),
        (-0.146, -0.714, -0.369),
        (0.453, 0.339, 0.117),
        (0.252, -0.343, 0.697),
        (-0.697, -0.007, 0.717),
        (-0.255, 0.154, -0.494),
        (0.647, 0.145, -0.476),
        (0.318, -0.929, -0.189),
        (-0.276, -0.435, 0.260),
        (0.711, -0.308, 0.256),
        (0.508, 0.690, 0.516),
        (0.000, 0.000, 0.000),
        (0.293, 0.713, -0.638),
        (-0.347, 0.460, -0.036),
        (-0.500, -0.640, -0.085),
        (-0.130, 0.242, 0.507),
        (-0.198, 0.516, 0.601),
        (0.847, -0.308, 0.433),
        (-0.521, 0.150, 0.199),
        (0.282, 0.105, 0.954),
        (-0.033, 0.815, -0.041),
        (0.579, -0.515, -0.632),
        (0.478, 0.107, 0.305),
        (0.198, 0.364, -0.704),
        (0.003, -0.567, 0.107),
        (0.645, 0.346, 0.362),
        (-1.000, -0.001, -0.011),
        (0.000, 0.000, 0.000),
        (0.022, 0.275, -0.507),
        (0.778, -0.181, -0.172),
        (-0.082, 0.935, -0.345),
        (0.638, 0.726, -0.258),
        (-0.029, -0.442, -0.371),
        (0.131, 0.265, 0.761),
        (0.387, -0.062, -0.424),
        (0.449, -0.357, 0.819),
        (0.368, -0.711, -0.163),
        (0.376, -0.323, 0.296),
        (0.553, -0.214, 0.561),
        (-0.044, -0.867, -0.496),
        (0.333, 0.471, -0.028),
        (-0.585, -0.379, 0.717),
        (0.611, 0.465, -0.277),
        (0.000, 0.000, 0.000),
        (0.073, 0.047, 0.571),
        (0.496, -0.216, -0.611),
        (0.682, -0.719, 0.132),
        (0.572, 0.071, 0.035),
        (-0.060, 0.647, -0.495),
        (-0.832, -0.474, -0.288),
        (-0.154, 0.533, 0.160),
        (-0.331, -0.514, -0.541),
        (0.917, -0.076, -0.392),
        (0.355, 0.254, -0.378),
        (0.623, -0.525, 0.052),
        (0.276, -0.186, -0.943),
        (0.324, 0.364, 0.309),
        (-0.101, 0.091, -0.805),
        (0.277, 0.762, -0.095),
        (0.000, 0.000, 0.000),
        (0.493, -0.300, -0.023),
        (-0.330, -0.483, -0.811),
        (-0.551, 0.755, 0.355),
        (-0.381, 0.021, -0.433),
        (0.757, 0.015, 0.305),
        (-0.079, 0.614, -0.785),
        (0.098, 0.438, -0.363),
        (0.442, 0.191, -0.660),
        (-0.553, -0.816, -0.167),
        (-0.308, 0.249, 0.420),
        (0.154, -0.713, -0.367),
        (0.696, -0.137, 0.705),
        (0.357, -0.920, 0.161),
        (0.476, 0.308, -0.109),
        # segment 2/3 with interleaved shells. Interleaved bval ratio 1:2:3
        # [directions=96]
        # CoordinateSystem = xyz
        # Normalisation = none
        (0.000, 0.000, 0.000),
        (-0.813, -0.341, 0.472),
        (0.208, -0.514, 0.159),
        (-0.772, -0.123, 0.235),
        (0.427, -0.503, 0.481),
        (-0.055, -0.276, -0.504),
        (-0.678, -0.452, -0.055),
        (-0.286, -0.890, 0.354),
        (-0.197, 0.319, -0.439),
        (0.135, -0.311, -0.743),
        (-0.935, 0.333, -0.119),
        (-0.263, -0.482, -0.178),
        (-0.030, -0.762, 0.292),
        (-0.013, -0.247, -0.969),
        (0.568, -0.099, -0.034),
        (-0.370, 0.033, -0.727),
        (0.934, 0.143, 0.327),
        (0.000, 0.000, 0.000),
        (-0.110, 0.048, 0.565),
        (-0.265, 0.759, 0.595),
        (0.667, -0.211, -0.421),
        (-0.166, -0.540, 0.121),
        (-0.408, -0.631, -0.319),
        (-0.416, -0.141, 0.898),
        (-0.430, -0.088, 0.375),
        (0.431, 0.450, -0.528),
        (0.026, 0.985, 0.169),
        (0.455, -0.147, 0.324),
        (0.396, -0.561, -0.442),
        (-0.776, 0.316, 0.546),
        (0.141, -0.460, -0.319),
        (0.798, -0.151, 0.087),
        (-0.205, 0.251, -0.946),
        (0.000, 0.000, 0.000),
        (-0.398, 0.397, 0.131),
        (-0.688, 0.577, -0.439),
        (0.029, -0.454, 0.678),
        (0.195, 0.281, -0.465),
        (0.522, 0.187, 0.162),
        (-0.089, -0.534, -0.611),
        (0.613, 0.441, 0.656),
        (0.290, -0.762, 0.048),
        (-0.767, -0.641, -0.009),
        (-0.125, 0.564, 0.008),
        (-0.639, -0.043, -0.506),
        (0.205, 0.718, 0.665),
        (0.245, 0.061, 0.519),
        (0.165, 0.789, 0.130),
        (-0.245, -0.512, 0.823),
        (0.000, 0.000, 0.000),
        (0.059, -0.139, 0.557),
        (-0.569, 0.513, -0.282),
        (0.507, -0.288, -0.813),
        (-0.433, 0.234, 0.302),
        (-0.430, 0.027, 0.694),
        (-0.969, -0.142, 0.203),
        (-0.183, -0.423, -0.348),
        (0.475, 0.636, -0.192),
        (-0.534, -0.082, -0.841),
        (0.444, -0.338, 0.149),
        (-0.115, -0.159, 0.793),
        (0.120, -0.810, 0.575),
        (0.707, -0.394, -0.109),
        (-0.404, -0.336, 0.239),
        (-0.671, 0.731, 0.120),
        (0.000, 0.000, 0.000),
        (0.389, 0.214, 0.370),
        (0.583, 0.630, -0.514),
        (0.503, 0.398, 0.505),
        (-0.312, -0.895, -0.320),
        (-0.552, -0.010, 0.170),
        (-0.726, -0.325, 0.185),
        (0.028, 0.517, -0.255),
        (0.243, -0.568, 0.533),
        (0.615, -0.447, 0.650),
        (-0.094, 0.992, -0.082),
        (-0.255, 0.091, 0.510),
        (0.360, -0.378, -0.628),
        (0.270, -0.428, 0.278),
        (0.099, -0.791, -0.177),
        (0.958, -0.247, -0.146),
        (0.000, 0.000, 0.000),
        (0.002, -0.531, -0.226),
        (-0.031, -0.011, 0.999),
        (0.176, 0.565, -0.563),
        (0.549, 0.003, 0.180),
        (0.292, 0.183, 0.740),
        (-0.754, -0.216, -0.229),
        (0.529, 0.232, -0.006),
        (-0.347, 0.537, 0.769),
        (0.942, 0.316, 0.113),
        (-0.107, 0.362, 0.437),
        (-0.530, 0.606, 0.138),
        (-0.882, 0.067, -0.467),
        (-0.479, -0.866, 0.141),
        (0.205, 0.539, 0.021),
        # segment 3/3 with interleaved shells. Interleaved bval ratio 1:2:3
        # [directions=97]
        # CoordinateSystem = xyz
        # Normalisation = none
        (0.000, 0.000, 0.000),
        (-0.711, 0.022, 0.401),
        (-0.039, 0.378, -0.435),
        (0.510, -0.800, 0.317),
        (0.607, -0.329, 0.435),
        (-0.318, -0.102, 0.471),
        (0.003, -0.109, -0.809),
        (-0.826, -0.112, 0.553),
        (-0.290, 0.489, 0.102),
        (-0.578, -0.522, -0.246),
        (-0.205, 0.898, 0.390),
        (0.345, -0.214, 0.410),
        (-0.163, 0.752, -0.273),
        (0.274, -0.524, 0.806),
        (-0.123, -0.162, -0.540),
        (0.681, 0.617, 0.395),
        (-0.581, -0.344, 0.460),
        (0.000, 0.000, 0.000),
        (-0.131, -0.437, -0.890),
        (0.484, -0.059, -0.309),
        (-0.029, 0.644, 0.502),
        (-0.371, -0.427, -0.113),
        (-0.402, 0.225, -0.674),
        (-0.739, 0.532, 0.413),
        (-0.238, -0.373, 0.371),
        (0.815, -0.009, -0.055),
        (-0.127, -0.833, 0.539),
        (-0.528, 0.225, -0.066),
        (-0.336, -0.735, -0.116),
        (-0.400, -0.349, 0.848),
        (0.833, -0.524, 0.178),
        (-0.173, 0.066, -0.547),
        (-0.535, 0.381, 0.485),
        (0.000, 0.000, 0.000),
        (0.016, 0.576, 0.036),
        (-0.489, -0.294, -0.821),
        (-0.053, 0.283, -0.764),
        (0.433, 0.284, 0.256),
        (0.211, 0.744, -0.264),
        (0.779, 0.529, -0.337),
        (-0.501, 0.072, 0.862),
        (-0.468, 0.287, 0.178),
        (0.231, 0.411, 0.667),
        (-0.490, 0.871, 0.046),
        (-0.087, -0.156, 0.549),
        (0.475, -0.656, 0.102),
        (-0.238, 0.359, 0.385),
        (-0.705, 0.136, -0.389),
        (-0.424, 0.132, -0.896),
        (0.000, 0.000, 0.000),
        (-0.968, 0.099, -0.229),
        (-0.385, -0.412, 0.124),
        (-0.582, 0.027, 0.572),
        (0.477, -0.019, 0.325),
        (-0.291, -0.197, 0.737),
        (-0.134, -0.975, 0.180),
        (-0.340, 0.437, -0.164),
        (-0.611, -0.534, 0.091),
        (-0.029, 0.724, 0.689),
        (0.151, 0.342, 0.439),
        (-0.323, 0.672, 0.332),
        (0.812, 0.342, 0.473),
        (0.451, 0.207, -0.295),
        (-0.004, 0.433, -0.901),
        (0.617, 0.217, 0.489),
        (0.000, 0.000, 0.000),
        (0.044, -0.157, -0.554),
        (-0.738, 0.344, -0.061),
        (0.324, 0.804, 0.499),
        (-0.092, 0.531, -0.207),
        (-0.292, 0.217, 0.731),
        (-0.301, 0.854, -0.425),
        (0.560, 0.119, -0.078),
        (0.461, 0.559, -0.376),
        (-0.709, 0.200, 0.676),
        (0.360, 0.107, 0.439),
        (0.252, 0.630, 0.454),
        (0.456, 0.784, -0.421),
        (-0.122, 0.251, -0.505),
        (0.453, -0.713, -0.535),
        (-0.384, 0.402, -0.598),
        (0.000, 0.000, 0.000),
        (0.141, 0.498, 0.257),
        (0.112, 0.806, -0.066),
        (0.889, -0.305, -0.342),
        (0.168, 0.493, -0.249),
        (-0.251, 0.472, 0.219),
        (0.713, 0.676, 0.185),
        (-0.235, -0.045, 0.971),
        (0.704, -0.309, -0.275),
        (0.212, 0.029, 0.788),
        (-0.466, 0.244, -0.237),
        (0.740, -0.299, 0.602),
        (0.252, -0.665, 0.402),
        (0.386, -0.167, -0.395),
        (0.804, 0.040, 0.138),
        (0.462, 0.553, -0.693),
    ]
    # Transpose to be 3 x 288
).T
