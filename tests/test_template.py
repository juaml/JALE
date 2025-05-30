import numpy as np

from jale.core.utils.template import (
    BRAIN_ARRAY_SHAPE,
    GM_PRIOR,
    GM_SAMPLE_SPACE,
    MNI_AFFINE,
    PAD_SHAPE,
    data,
)


def test_data_shape_and_type():
    assert isinstance(data, np.ndarray)
    assert len(BRAIN_ARRAY_SHAPE) == 3  # Typical 3D brain shape
    assert all(isinstance(dim, int) for dim in BRAIN_ARRAY_SHAPE)


def test_pad_shape_is_larger():
    assert np.all(PAD_SHAPE >= BRAIN_ARRAY_SHAPE[0] + 30)


def test_gm_prior_is_binary_and_shape():
    assert GM_PRIOR.dtype == bool
    assert GM_PRIOR.shape == BRAIN_ARRAY_SHAPE
    # Values are only 0 or 1 (False or True)
    unique_vals = np.unique(GM_PRIOR)
    assert set(unique_vals).issubset({False, True})


def test_gm_sample_space_consistent():
    assert GM_SAMPLE_SPACE.shape[0] == 3  # 3 coordinates arrays
    # All coords within bounds
    assert np.all(GM_SAMPLE_SPACE >= 0)
    for dim, size in enumerate(BRAIN_ARRAY_SHAPE):
        assert np.all(GM_SAMPLE_SPACE[dim] < size)


def test_mni_affine_shape():
    assert MNI_AFFINE.shape == (4, 4)
