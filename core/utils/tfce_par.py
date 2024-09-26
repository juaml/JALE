import numpy as np
from scipy import ndimage
from functools import reduce
import operator


def tfce_par(invol, h, dh, voxel_dims=[2, 2, 2], E=0.6, H=2):
    # Threshold input volume
    thresh = invol > h

    # Identify suprathreshold clusters and count their sizes
    labels, cluster_count = ndimage.label(thresh)

    # Get unique labels and their sizes, excluding background (label 0)
    sizes = np.bincount(labels.ravel()) * reduce(operator.mul, voxel_dims)

    # Mask out the non-labeled areas
    mask = labels > 0

    # Get cluster sizes for labeled areas and compute TFCE update values
    update_vals = (h ** H) * dh * (sizes[labels[mask]] ** E)

    return update_vals, mask
