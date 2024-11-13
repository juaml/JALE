import nibabel as nb

from jale.core.utils.template import GM_PRIOR, MNI_AFFINE


def plot_and_save(arr, nii_path):
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
    arr_masked = arr
    arr_masked[GM_PRIOR == 0] = 0
    nii_img = nb.nifti1.Nifti1Image(arr_masked, MNI_AFFINE)
    nb.loadsave.save(nii_img, nii_path)
