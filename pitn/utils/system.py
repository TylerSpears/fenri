# -*- coding: utf-8 -*-
from pathlib import Path

import torch

from pitn._lazy_loader import LazyLoader

GPUtil = LazyLoader("GPUtil", globals(), "GPUtil")
tabulate = LazyLoader("tabulate", globals(), "tabulate")


def get_gpu_specs():
    """Return string describing GPU specifications.

    Taken from
    <https://www.thepythoncode.com/article/get-hardware-system-information-python>.

    Returns
    -------
    str
        Human-readable string of specifications.
    """

    gpus = GPUtil.getGPUs()
    specs = list()
    specs.append("".join(["=" * 50, "GPU Specs", "=" * 50]))
    list_gpus = []
    for gpu in gpus:
        # get the GPU id
        gpu_id = gpu.id
        # name of GPU
        gpu_name = gpu.name
        driver_version = gpu.driver
        cuda_version = torch.version.cuda
        # get total memory
        gpu_total_memory = f"{gpu.memoryTotal}MB"
        gpu_uuid = gpu.uuid
        list_gpus.append(
            (
                gpu_id,
                gpu_name,
                driver_version,
                cuda_version,
                gpu_total_memory,
                gpu_uuid,
            )
        )

    table = tabulate.tabulate(
        list_gpus,
        headers=(
            "id",
            "Name",
            "Driver Version",
            "CUDA Version",
            "Total Memory",
            "uuid",
        ),
    )

    specs.append(table)

    return "\n".join(specs)


def get_file_glob_unique(root_path: Path, glob_pattern: str) -> Path:
    root_path = Path(root_path)
    glob_pattern = str(glob_pattern)
    files = list(root_path.glob(glob_pattern))

    if len(files) == 0:
        files = list(root_path.rglob(glob_pattern))

    if len(files) > 1:
        raise RuntimeError(
            "ERROR: More than one file matches glob pattern "
            + f"{glob_pattern} under directory {str(root_path)}. "
            + "Expect only one match."
        )
    elif len(files) == 0:
        raise RuntimeError(
            "ERROR: No files match glob pattern "
            + f"{glob_pattern} under directory {str(root_path)}; "
            + "Expect one match."
        )

    return files[0]
