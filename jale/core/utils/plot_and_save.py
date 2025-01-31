import nibabel as nb
import numpy as np
from nilearn import plotting

from jale.core.utils.template import MNI_AFFINE


def plot_and_save(project_path, analysis_name, arr):
    """
    Save a brain data array as a NIfTI file.

    Applies a mask to the array based on a prior and saves it as a NIfTI image.

    Parameters
    ----------
    arr : numpy.ndarray
        Brain data array to save.
    nii_path : str or Path
        Path to save the NIfTI file.

    Returns
    -------
    None
    """

    # Function that takes brain array and transforms it to NIFTI1 format
    # Saves it as a Nifti file
    arr = np.nan_to_num(arr, nan=0, posinf=0, neginf=0)
    # arr[GM_PRIOR == 0] = 0
    nii_img = nb.nifti1.Nifti1Image(arr, MNI_AFFINE)
    if arr.any() > 0:
        plotting.plot_stat_map(
            nii_img, output_file=project_path / f"Figures/{analysis_name}"
        )
        nb.loadsave.save(nii_img, project_path / f"Volumes/{analysis_name}")
    else:
        nb.loadsave.save(nii_img, project_path / f"Volumes/{analysis_name}_empty")
