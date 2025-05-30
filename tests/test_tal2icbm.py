import numpy as np
from numpy.testing import assert_array_almost_equal

from jale.core.utils.tal2icbm_spm import tal2icbm_spm


def test_tal2icbm_spm_shape_and_type():
    tal_coords = np.array([[0.0, 0.0, 0.0], [10.0, -10.0, 20.0]])
    mni_coords = tal2icbm_spm(tal_coords)

    assert isinstance(mni_coords, np.ndarray)
    assert mni_coords.shape == (2, 3)
    assert mni_coords.dtype.kind in {"f", "i"}


def test_tal2icbm_spm_consistency():
    tal_coords = np.array([[10.0, 20.0, -30.0]])
    result1 = tal2icbm_spm(tal_coords)
    result2 = tal2icbm_spm(tal_coords)

    np.testing.assert_array_equal(result1, result2)


def test_tal2icbm_spm_rounding_behavior():
    tal_coords = np.array([[0.0, 0.0, 0.0]])
    mni_coords = tal2icbm_spm(tal_coords)

    # Test only that it rounds to integers
    assert np.all(np.equal(np.round(mni_coords), mni_coords))


def test_tal2icbm_spm_identity_input_does_not_crash():
    tal_coords = np.zeros((10, 3))  # all-zero points
    output = tal2icbm_spm(tal_coords)

    assert output.shape == (10, 3)


def test_tal2icbm_spm_origin():
    # Test the transformation at the origin
    tal_coords = np.array([[0.0, 0.0, 0.0]])
    mni_coords = tal2icbm_spm(tal_coords)

    expected = np.array([[1, 1, -5]])
    assert_array_almost_equal(mni_coords, expected, decimal=0)
