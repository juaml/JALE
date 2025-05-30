from unittest.mock import patch

import numpy as np

from jale.core.utils.compute import (
    compute_ale,
    compute_balanced_ale_diff,
    compute_balanced_null_diff,
    compute_clusters,
    compute_hx,
    compute_hx_conv,
    compute_ma,
    compute_monte_carlo_null,
    compute_null_ale,
    compute_permuted_ale_diff,
    compute_sig_diff,
    compute_tfce,
    compute_z,
    generate_unique_subsamples,
    illustrate_foci,
)
from jale.core.utils.template import BRAIN_ARRAY_SHAPE


def test_illustrate_foci():
    foci = [np.array([[10, 10, 10], [20, 20, 20]])]
    result = illustrate_foci(foci)
    assert result.shape == BRAIN_ARRAY_SHAPE
    assert result[10, 10, 10] == 1
    assert result[20, 20, 20] == 1


def test_compute_ma():
    foci = [np.array([[10, 10, 10]])]
    kernels = [np.ones((30, 30, 30))]  # Dummy kernel
    ma = compute_ma(foci, kernels)
    assert ma.shape[0] == 1
    assert ma.shape[1:] == BRAIN_ARRAY_SHAPE


def test_compute_hx():
    ma = np.zeros((3,) + BRAIN_ARRAY_SHAPE)
    ma[0, 10, 10, 10] = 0.5
    ma[1, 20, 20, 20] = 0.8
    bin_edges = np.linspace(0, 1, 11)
    hx = compute_hx(ma, bin_edges)
    assert hx.shape == (3, len(bin_edges))
    assert np.sum(hx) > 0


def test_compute_ale():
    ma = np.zeros((2,) + BRAIN_ARRAY_SHAPE)
    ma[0, 10, 10, 10] = 0.2
    ma[1, 10, 10, 10] = 0.3
    ale = compute_ale(ma)
    assert ale.shape == BRAIN_ARRAY_SHAPE
    assert 0 < ale[10, 10, 10] < 1


def test_compute_hx_conv():
    hx = np.zeros((2, 10))
    hx[0, 2] = 10
    hx[1, 3] = 10
    bin_centers = np.linspace(0, 1, 10)
    step = 9
    hx_conv = compute_hx_conv(hx, bin_centers, step)
    assert hx_conv.ndim == 1
    assert len(hx_conv) <= len(bin_centers)


def test_compute_z():
    ale = np.array([0.1, 0.3, 0.5])
    hx_conv = np.linspace(1, 0.001, 100)
    step = 99
    z = compute_z(ale, hx_conv, step)
    assert z.shape == ale.shape
    assert np.all(z >= 0)


def test_compute_tfce():
    z = np.zeros(BRAIN_ARRAY_SHAPE)
    z[10, 10, 10] = 3.0
    tfce = compute_tfce(z, delta_t_steps=10)
    assert tfce.shape == z.shape
    assert tfce[10, 10, 10] > 0


def test_compute_clusters():
    z = np.zeros(BRAIN_ARRAY_SHAPE)
    z[10:12, 10:12, 10:12] = 4.0  # Simulate small cluster

    thresh_z, max_clust = compute_clusters(z, cluster_forming_threshold=0.001)
    assert thresh_z.shape == BRAIN_ARRAY_SHAPE
    assert max_clust > 0

    # With cfwe threshold that removes all clusters
    thresh_z2, max_clust2 = compute_clusters(
        z, cluster_forming_threshold=0.001, cfwe_threshold=1000
    )
    assert np.count_nonzero(thresh_z2) == 0


def test_compute_null_ale():
    num_foci = [5, 10]
    kernels = [np.ones((30, 30, 30)) for _ in num_foci]  # Dummy kernels

    ma, ale = compute_null_ale(num_foci, kernels)
    assert ma.shape[0] == len(num_foci)
    assert ale.shape == BRAIN_ARRAY_SHAPE
    assert np.max(ale) <= 1.0 and np.min(ale) >= 0.0


def test_compute_monte_carlo_null():
    num_foci = np.array([8, 9, 10])
    kernels = [np.full((30, 30, 30), 0.01) for _ in num_foci]

    bin_steps = 0.0001
    bin_edges = np.arange(0.00005, 1 - 0.02 + 0.001, bin_steps)
    bin_centers = np.arange(0, 1 - 0.02 + 0.001, bin_steps)
    step = int(1 / bin_steps)

    null_max_ale, null_max_cluster, null_max_tfce = compute_monte_carlo_null(
        num_foci=num_foci,
        kernels=kernels,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        step=step,
        cluster_forming_threshold=0.001,
        tfce_enabled=True,
    )

    assert isinstance(null_max_ale, float)
    assert isinstance(null_max_cluster, (int, np.integer))
    assert isinstance(null_max_tfce, float)
    assert null_max_ale > 0


def test_generate_unique_subsamples_returns_unique():
    total_n = 10
    target_n = 5
    sample_n = 50

    subsamples = generate_unique_subsamples(total_n, target_n, sample_n)
    assert len(subsamples) == sample_n
    assert all(len(np.unique(sample)) == target_n for sample in subsamples)

    # Ensure uniqueness
    subsample_set = set(tuple(s) for s in subsamples)
    assert len(subsample_set) == len(subsamples)


def test_generate_unique_subsamples_max_combinations():
    total_n = 5
    target_n = 5
    sample_n = 10  # Should clip to 1 (only one way to choose all items)

    subsamples = generate_unique_subsamples(total_n, target_n, sample_n)
    assert len(subsamples) == 1
    assert np.array_equal(subsamples[0], np.arange(5))


@patch("jale.core.utils.compute.compute_ale")
def test_compute_permuted_ale_diff(mock_compute_ale):
    n = 10
    shape = (n, 3)
    ma_merge = np.random.rand(*shape)

    # Mock ALE output
    mock_compute_ale.side_effect = [np.ones(3), np.zeros(3)]

    result = compute_permuted_ale_diff(ma_merge, 5)
    expected = np.ones(3)  # 1 - 0
    np.testing.assert_array_equal(result, expected)


def test_compute_sig_diff_significant():
    ale_diff = np.array([0.9, 0.1, 0.4])
    null_diff = np.array(
        [
            [0.1, 0.2, 0.1],
            [0.2, 0.3, 0.2],
            [0.3, 0.4, 0.3],
            [0.4, 0.5, 0.4],
        ]
    )

    z_diff, sig_idxs = compute_sig_diff(
        ale_diff, null_diff, significance_threshold=0.05
    )
    assert isinstance(z_diff, np.ndarray)
    assert isinstance(sig_idxs, tuple)
    assert np.all(z_diff >= 0)
    assert len(sig_idxs[0]) == len(z_diff)


def test_compute_sig_diff_no_significance():
    ale_diff = np.zeros(5)
    null_diff = np.ones((10, 5))

    z_diff, sig_idxs = compute_sig_diff(ale_diff, null_diff)
    assert z_diff == 0
    assert sig_idxs == 0


@patch("jale.core.utils.compute.compute_ale")
def test_compute_balanced_ale_diff(mock_compute_ale):
    prior = np.array([True, False, True, False])
    ma1 = np.random.rand(5, 4)
    ma2 = np.random.rand(5, 4)

    mock_compute_ale.side_effect = [
        np.array([0.2, 0.4]),  # ale1
        np.array([0.1, 0.3]),  # ale2
    ]

    diff = compute_balanced_ale_diff(ma1, ma2, prior, target_n=3)
    expected = np.array([0.1, 0.1])
    np.testing.assert_allclose(diff, expected)


@patch("jale.core.utils.compute.compute_ma")
@patch("jale.core.utils.compute.compute_balanced_ale_diff")
def test_compute_balanced_null_diff(mock_diff, mock_ma):
    target_n = 2
    iterations = 3
    prior = np.array([True, True, False, True])
    nfoci1 = [3, 4]
    nfoci2 = [3, 4]
    kernels1 = [np.ones(3)] * 2
    kernels2 = [np.ones(3)] * 2

    mock_ma.side_effect = [np.random.rand(2, 4), np.random.rand(2, 4)]
    mock_diff.return_value = np.array([0.2, -0.1, 0.05])

    min_diff, max_diff = compute_balanced_null_diff(
        nfoci1, kernels1, nfoci2, kernels2, prior, target_n, iterations
    )

    assert isinstance(min_diff, float)
    assert isinstance(max_diff, float)
    assert min_diff <= max_diff
