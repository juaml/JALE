import math
import numpy as np
from core.utils.template import PAD_SHAPE


def kernel_calc(affine, fwhm, dims):
    # Convert FWHM to sigma
    sigma = (fwhm / (2 * math.sqrt(8 * math.log(2))) +
             np.finfo(float).eps) ** 2

    # Generate 1D Gaussian kernel based on sigma
    # Half of required kernel length
    half_k_length = int(np.ceil(3.5 * np.sqrt(sigma)))
    x = np.arange(-half_k_length, half_k_length + 1)

    oned_kernel = np.exp(-0.5 * (x ** 2) / sigma)
    oned_kernel /= np.sum(oned_kernel)  # Normalize kernel

    # Create 3D Gaussian kernel using outer product
    gkern3d = np.outer(oned_kernel, oned_kernel)[:, None] * oned_kernel

    # Calculate padding size to match desired dimensions
    pad_size = (dims - len(x)) // 2
    gkern3d = np.pad(gkern3d, ((pad_size, pad_size),
                               (pad_size, pad_size),
                               (pad_size, pad_size)), 'constant', constant_values=0)

    return gkern3d


def kernel_conv(peaks, kernel):
    kernel_size = kernel.shape[0]  # Assuming kernel is a cubic shape
    pad = kernel_size // 2  # Padding based on kernel size

    data = np.zeros(PAD_SHAPE)

    # Iterate over peaks and apply kernel using maximum value at each location
    for peak in peaks:
        # Define slice ranges for x, y, z based on peak coordinates and kernel size
        x_start, x_end = peak[0], peak[0] + kernel_size
        y_start, y_end = peak[1], peak[1] + kernel_size
        z_start, z_end = peak[2], peak[2] + kernel_size

        # Apply the maximum operation in the local region
        data[x_start:x_end, y_start:y_end, z_start:z_end] = np.maximum(
            data[x_start:x_end, y_start:y_end, z_start:z_end], kernel
        )

    # Return the data after trimming the padding
    return data[pad:data.shape[0] - pad, pad:data.shape[1] - pad, pad:data.shape[2] - pad]
