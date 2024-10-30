import numpy as np


def tal2icbm_spm(inpoints):
    """
    Convert coordinates from Talairach space to MNI space using the SPM transformation.

    This function applies the tal2icbm transformation matrix developed by Jack Lancaster
    at the Research Imaging Center in San Antonio, Texas. The transformation is used to
    convert Talairach coordinates to MNI space as normalized by the SPM software package.

    Parameters
    ----------
    inpoints : numpy.ndarray
        Array of Talairach coordinates to be converted, with shape (n, 3), where n is the
        number of coordinates.

    Returns
    -------
    numpy.ndarray
        Array of converted coordinates in MNI space with shape (n, 3).
    """
    # Define the transformation matrix for Talairach to MNI (SPM)
    icbm_spm = np.array(
        (
            [0.9254, 0.0024, -0.0118, -1.0207],
            [-0.0048, 0.9316, -0.0871, -1.7667],
            [0.0152, 0.0883, 0.8924, 4.0926],
            [0.0000, 0.0000, 0.0000, 1.0000],
        )
    )

    # Invert the transformation matrix to apply to Talairach points
    icbm_spm = np.linalg.inv(icbm_spm)

    # Pad input coordinates to homogeneous form by adding a fourth dimension of ones
    inpoints = np.pad(inpoints, ((0, 0), (0, 1)), "constant", constant_values=1)

    # Apply the transformation matrix to convert coordinates to MNI space
    inpoints = np.dot(icbm_spm, inpoints.T)

    # Round the transformed coordinates to the nearest integer for final output
    outpoints = np.round(inpoints[:3])

    return outpoints.T
