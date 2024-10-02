import numpy as np
import math
from core.utils.template import PAD_SHAPE


def create_kernel_array(exp_df):
    kernel_array = np.empty((exp_df.shape[0], 31, 31, 31))
    template_uncertainty = 5.7 / \
        (2 * np.sqrt(2 / np.pi)) * np.sqrt(8 * np.log(2))
    for i, n_subjects in enumerate(exp_df.Subjects.values):
        subj_uncertainty = (11.6 / (2 * np.sqrt(2 / np.pi)) *
                            np.sqrt(8 * np.log(2))) / np.sqrt(n_subjects)
        smoothing = np.sqrt(template_uncertainty**2 + subj_uncertainty**2)
        kernel_array[i] = compute_3dkernel(smoothing, 31)
    return kernel_array


def compute_3dkernel(fwhm, dims):
    s = (fwhm/2/math.sqrt(8*math.log(2)) +
         np.finfo(float).eps)**2  # fwhm -> sigma

    # 1D Gaussian based on sigma

    # Half of required kernel length
    half_k_length = math.ceil(3.5*math.sqrt(s))
    x = list(range(-half_k_length, half_k_length + 1))
    oned_kernel = np.exp(-0.5 * np.multiply(x, x) / s) / math.sqrt(2*math.pi*s)
    oned_kernel = np.divide(oned_kernel, np.sum(oned_kernel))

    # Convolution of 1D Gaussians to create 3D Gaussian
    gkern3d = oned_kernel[:, None, None] * \
        oned_kernel[None, :, None] * oned_kernel[None, None, :]

    # Padding to get matrix of desired size
    pad_size = int((dims - len(x)) / 2)
    gkern3d = np.pad(gkern3d, ((pad_size, pad_size),
                               (pad_size, pad_size),
                               (pad_size, pad_size)), 'constant', constant_values=0)
    return gkern3d


def kernel_convolution(foci, kernel):
    kernel_size = kernel.shape[0]  # Assuming kernel is a cubic shape
    pad = kernel_size // 2  # Padding based on kernel size

    data = np.zeros(PAD_SHAPE)

    # Iterate over focis and apply kernel using maximum value at each location
    for focus in foci:
        # Define slice ranges for x, y, z based on foci coordinates and kernel size
        x_start, x_end = focus[0], focus[0] + kernel_size
        y_start, y_end = focus[1], focus[1] + kernel_size
        z_start, z_end = focus[2], focus[2] + kernel_size

        # Apply the maximum operation in the local region
        data[x_start:x_end, y_start:y_end, z_start:z_end] = np.maximum(
            data[x_start:x_end, y_start:y_end, z_start:z_end], kernel
        )

    # Return the data after trimming the padding
    return data[pad:data.shape[0] - pad, pad:data.shape[1] - pad, pad:data.shape[2] - pad]
