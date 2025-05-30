import numpy as np
import pandas as pd

from jale.core.utils.kernel import (
    compute_3dkernel,
    create_kernel_array,
    kernel_convolution,
)
from jale.core.utils.template import PAD_SHAPE


def test_compute_3dkernel_shape_and_sum():
    kernel = compute_3dkernel(fwhm=5.0, dims=31)
    assert kernel.shape == (31, 31, 31)
    assert np.isclose(kernel.sum(), 1.0, atol=1e-6)


def test_create_kernel_array_output():
    df = pd.DataFrame({"Subjects": [10, 20, 30]})
    kernel_arr = create_kernel_array(df)

    assert kernel_arr.shape == (3, 31, 31, 31)
    for kernel in kernel_arr:
        assert np.isclose(kernel.sum(), 1.0, atol=1e-4)


def test_kernel_convolution_single_focus():
    kernel = np.zeros((3, 3, 3))
    kernel[1, 1, 1] = 1.0  # Simple kernel with one peak

    foci = [(10, 10, 10)]
    output = kernel_convolution(foci, kernel)

    expected_shape = tuple(s - kernel.shape[0] + 1 for s in PAD_SHAPE)
    assert output.shape == expected_shape

    # Check that the max value (1.0) is at the expected position
    assert output[foci[0][0], foci[0][1], foci[0][2]] == 1.0
