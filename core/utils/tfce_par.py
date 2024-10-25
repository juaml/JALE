import operator
from functools import reduce

import numpy as np
from scipy import ndimage


def tfce_par(invol, h, dh, voxel_dims=[2, 2, 2], E=0.6, H=2):
    """
    Compute the Threshold-Free Cluster Enhancement (TFCE) update values for a given threshold.

    This function applies a threshold to an input volume, identifies clusters above the threshold,
    and calculates TFCE update values based on cluster sizes and specified parameters.

    Parameters
    ----------
    invol : numpy.ndarray
        Input 3D volume to be thresholded and analyzed.
    h : float
        Intensity threshold applied to `invol`.
    dh : float
        Step size for threshold increments in TFCE calculation.
    voxel_dims : list of int, optional
        Dimensions of each voxel in millimeters, by default [2, 2, 2].
    E : float, optional
        TFCE parameter that adjusts the emphasis on cluster size, by default 0.6.
    H : float, optional
        TFCE parameter that adjusts the emphasis on intensity height, by default 2.

    Returns
    -------
    tuple
        - numpy.ndarray : TFCE update values calculated for suprathreshold clusters.
        - numpy.ndarray : Mask of labeled suprathreshold areas.
    """
    # Threshold input volume based on intensity threshold `h`
    thresh = invol > h

    # Identify suprathreshold clusters and get the number of unique clusters
    labels, cluster_count = ndimage.label(thresh)  # type: ignore

    # Calculate sizes of each cluster, converting to voxel volume by multiplying dimensions
    sizes = np.bincount(labels.ravel()) * reduce(operator.mul, voxel_dims)

    # Create a mask of only the labeled (clustered) areas
    mask = labels > 0

    # Compute TFCE update values for each cluster using the specified E and H parameters
    update_vals = (h**H) * dh * (sizes[labels[mask]] ** E)

    return update_vals, mask
