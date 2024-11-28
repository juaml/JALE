import math

import numpy as np

from jale.core.utils.template import PAD_SHAPE


def create_kernel_array(exp_df):
    """
    Create an array of 3D Gaussian kernels based on subject count for each experiment.

    This function calculates a smoothing kernel for each experiment in the dataset,
    taking into account both template and subject uncertainties. The kernel is stored
    in a 3D array representing the Gaussian spread for each experiment.

    Parameters
    ----------
    exp_df : pandas.DataFrame
        DataFrame containing experiment information with a 'Subjects' column.

    Returns
    -------
    numpy.ndarray
        4D array of Gaussian kernels, with shape (num_experiments, 31, 31, 31).
    """
    # Initialize an empty array for storing kernels, with fixed dimensions (31, 31, 31)
    kernel_array = np.empty((exp_df.shape[0], 31, 31, 31))

    # Calculate template uncertainty (constant across experiments)
    template_uncertainty = 5.7 / (2 * np.sqrt(2 / np.pi)) * np.sqrt(8 * np.log(2))

    # Generate a kernel for each experiment based on subject-specific uncertainty
    for i, n_subjects in enumerate(exp_df.Subjects.values):
        # Calculate subject uncertainty based on the number of subjects
        subj_uncertainty = (
            11.6 / (2 * np.sqrt(2 / np.pi)) * np.sqrt(8 * np.log(2))
        ) / np.sqrt(n_subjects)

        # Combine uncertainties to get smoothing value for the Gaussian kernel
        smoothing = np.sqrt(template_uncertainty**2 + subj_uncertainty**2)

        # Compute and store the 3D kernel for the current experiment
        kernel_array[i] = compute_3dkernel(smoothing, 31)

    return kernel_array


def compute_3dkernel(fwhm, dims):
    """
    Compute a 3D Gaussian kernel based on full width at half maximum (FWHM).

    This function generates a 3D Gaussian kernel by convolving three 1D Gaussians.
    The kernel is padded to fit the specified dimensions.

    Parameters
    ----------
    fwhm : float
        Full width at half maximum, used to calculate the Gaussian spread.
    dims : int
        Desired size for the output kernel (dims x dims x dims).

    Returns
    -------
    numpy.ndarray
        3D Gaussian kernel array with specified dimensions.
    """
    # Calculate the variance from FWHM
    s = (
        fwhm / 2 / math.sqrt(8 * math.log(2)) + np.finfo(float).eps
    ) ** 2  # fwhm -> sigma

    # Define kernel range in one dimension
    half_k_length = math.ceil(3.5 * math.sqrt(s))
    x = list(range(-half_k_length, half_k_length + 1))

    # Create a normalized 1D Gaussian kernel
    oned_kernel = np.exp(-0.5 * np.multiply(x, x) / s) / math.sqrt(2 * math.pi * s)
    oned_kernel /= np.sum(oned_kernel)

    # Create a 3D Gaussian by convolving 1D Gaussian along each axis
    gkern3d = (
        oned_kernel[:, None, None]
        * oned_kernel[None, :, None]
        * oned_kernel[None, None, :]
    )

    # Pad the 3D Gaussian to match desired dimensions
    pad_size = int((dims - len(x)) / 2)
    gkern3d = np.pad(
        gkern3d,
        ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size)),
        "constant",
        constant_values=0,
    )
    return gkern3d


def kernel_convolution(foci, kernel):
    """
    Apply a Gaussian kernel to foci coordinates within a 3D space.

    This function convolves a Gaussian kernel over a 3D array, centered at each
    specified focus point, using maximum value in overlapping areas.

    Parameters
    ----------
    foci : list of tuples
        List of coordinates where each focus point is applied in 3D space.
    kernel : numpy.ndarray
        3D Gaussian kernel array used for convolution.

    Returns
    -------
    numpy.ndarray
        3D array with kernel applied at each focus point, trimmed to the padded size.
    """
    # Determine kernel size and padding required for each axis
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2

    # Initialize a 3D array with padding around the focus coordinates
    data = np.zeros(PAD_SHAPE)

    # Iterate over each focus point and apply the kernel centered at the focus
    for focus in foci:
        # Define the boundaries of the region where the kernel will be applied
        x_start, x_end = focus[0], focus[0] + kernel_size
        y_start, y_end = focus[1], focus[1] + kernel_size
        z_start, z_end = focus[2], focus[2] + kernel_size

        # Use maximum operation for overlapping kernel regions
        data[x_start:x_end, y_start:y_end, z_start:z_end] = np.maximum(
            data[x_start:x_end, y_start:y_end, z_start:z_end], kernel
        )

    # Trim padding and return the final 3D array
    return data[
        pad : data.shape[0] - pad, pad : data.shape[1] - pad, pad : data.shape[2] - pad
    ]
