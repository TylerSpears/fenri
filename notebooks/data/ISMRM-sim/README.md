# Code & Processing Steps for ISMRM-sim Simulations

Authors: Tyler Spears, Tom Fletcher

This directory contains scripts used to create the ISMRM-sim dataset, as well as an overview of the entire process.
Scripts/directories are numbered by the order they should be run, and non-numbered files are optional post-processing scripts.
These scripts have not been tested end-to-end, so there may be small snippets of "glue code" required between some parts of the pipeline.

## Processing Steps

Recreating (or expanding) the ISMRM-sim data can be done with the following steps. To start, you will need the ISMRM 2015 challenge simulation files found at <https://tractometer.org/ismrm2015/dwi_data/>, specifically the simulated T1w image and the tissue component partial volume maps, as well as the 2023 updated ground truth bundles and ROI masks at <https://tractometer.org/ismrm2015/tools/>. Next, you will need image data for the HCP target subject, specifically the subject's T1 and Freesurfer parcellation images. Finally, you will need the MNI-ICBM 152 2009 T1w template, which can be downloaded at <https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009>. For this work we used the nonlinear asymmetric 1.0mm T1w brain template image, but other templates may give better registration results.

### 1. Registration from ISMRM to HCP

Warping from the original ISMRM challenge dataset to each HCP target subject is done via the MNI-152 T1w template. Using SyN nonlinear registration, the challenge dataset's simulated T1w is registered to the MNI template. Similary, each target subject's real T1w image is registered to the MNI template.

The `01_reg_mni_to_hcp.sh` script registers each target HCP subject's T1w to the given template; you can use the registration commands in this script to also register the ISMRM simulated T1 to the template T1. Then, `02_create_ismrm_to_hcp_warps.sh` will produce the full ISMRM -> HCP warps, both push-forward and pullback warps.

### 2. Tissue Compartment Partial Volume Map Creation

With four tissue compartments, Fiberfox needs an explicit partial volume fraction of at least one non-WM tissue compartment. Our goal was to create realistic tissue partial volumes/tissue borders that matched the exact locations of WM fibers in the new simulated subject. Because the registration between ISMRM T1 and HCP T1 is not perfect, it is risky to estimate the tissue PVs from the HCP subject's real T1 image. While the ISMRM challenge data does include a CSF partial volume map, it is in 2.0mm resolution, and produces low-quality tissue boundaries in the 0.9mm simulated DWIs.

Instead, we chose to estimate high-resolution CSF PVs from the given simulated 1.0mm T1 image through a simple 1-dimensional regression model that predicts CSF PV fractions from T1 values. This high(er) resolution CSF PV is then warped from ISMRM to the target HCP. More details and code examples can be found in `03_tissue_compartment_partial_vol/`. Note that in these simulations, compartment 4 is proximal to a "CSF" tissue compartment.

### 3. Fiberfox Prep

The `04_warp_and_fiberfox_prep.sh` script prepares the Fiberfox input files by warping the ISMRM T1, brain mask, CSF compartment partial volume fraction, and fiber streamlines to the target HCP subject. Additionally, each subject requires a "template" DWI volume in the .nrrd image format, with gradient directions/magnitudes embedded in the .nrrd file itself. This can be done by loading a DWI image (an empty volume is sufficient, we only care about the image parameters) into the MITK Diffusion GUI and saving out in the NRRD format.

### 4. Fiberfox Simulation

Fiberfox can finally be run with the `05_run_fiberfox.sh` script. The input parameters file can be found in the `params/` directory. The input parameters, PV fraction, and brain mask must all follow the Fiberfox expected naming scheme.

*NOTE* Fiberfox is incredibly CPU-intensive, and can easily run for days when simulating at 0.9mm, even on a powerful multi-core CPU. We recommend using a CPU compute cluster, if possible.

For the purposes of batch scheduling, we chose to split the simulation into 4 different runs by dividing the target gradient directions/magnitudes into chunks. The rest of these scripts assume this splitting, but the simulation results should be identical regardless of splitting.

### 5. Extracting & Correcting DWIs

The Fiberfox simulation output DWIs contain gradient orientation errors, at least when bringing images over to MRtrix. The `06_combine_splites.sh` script combines the split Fiberfox output DWIs into a single DWI image and uses MRtrix's `dwigradcheck` command to correct gradient orientations.

### 6. (Optional) Normalising & Model Fitting

There are several optional post-processing steps that are performed after simulation. The `tissue_mask_creation.sh` script creates a five tissue-type (5tt) segmentation image given an ISMRM 5tt and the target subject's compartmental partial volume fraction images. This script relies on the `fs_sim_fivett_merge.py` python script, which applies fraction thresholds to the partial volume fractions to create the 5tt image. The `fit_odf_ismrm-2015-sims.sh` script normalizes the DWIs (with MRtrix's `dwinormalise individual`), estimates ODF SH coefficients with MSMT-CSD, and normalizes those same SH coefficients. The `proc_subj.sh` script shows an example pipeline that chains these scripts together.

These scripts should be seen as more experimental/subject to change than the preprocessing and simulation scripts.

## Software Environment

All preprocessing and postprocessing steps were run on Ubuntu Linux 22.04, and Fiberfox simulations were run on a CentOS 7 cluster with the SLURM job scheduler.
A container was needed to run Fiberfox on a cluster which can be found in (relative to the repository root) `docker/mitk_diffusion`.
This docker image essentially just downloads MITK Diffusion, so it is not a hard requirement for re-running these simulations.

Software Versions:

* MITK Diffusion (Fiberfox), build `ubuntu-20.04_2023.08.21_a754b053_32d7d08a_NoPython`
* MRtrix 3.0.4
* ANTS 2.4.3
* Python 3.10
* scikit-learn 1.1.3
