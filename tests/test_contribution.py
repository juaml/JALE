import io
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
import pytest

from jale.core.utils import contribution


@pytest.fixture
def dummy_exp_df():
    # Minimal dummy experiment DataFrame with required columns
    return pd.DataFrame(
        {
            "Coordinates": [np.array([[1, 2, 3], [4, 5, 6]])] * 3,
            "Subjects": [10, 20, 30],
            "Articles": ["Art1", "Art2", "Art3"],
        }
    )


@pytest.fixture
def dummy_tasks():
    return pd.DataFrame({"Name": ["TaskA", "TaskB"], "ExpIndex": [[0, 1], [2]]})


def test_compute_cluster_center():
    cluster_idxs = (
        np.array([1, 2, 3]),
        np.array([4, 5, 6]),
        np.array([7, 8, 9]),
    )
    center = contribution.compute_cluster_center(cluster_idxs)
    assert center.shape == (4,)
    # The last coord is homogeneous coordinate 1
    assert np.isclose(center[-1], 1)


def test_calculate_contributions_shape():
    ma = np.random.rand(5, 10)
    ale = np.mean(ma, axis=0)
    contrib = contribution.calculate_contributions(ma, ale)
    assert contrib.shape == (5, 4)


def test_get_clusters_min_size():
    labels = np.zeros((5, 5, 5), dtype=int)
    labels[0:2, 0:2, 0:2] = 1
    labels[3:5, 3:5, 3:5] = 2
    clusters = contribution.get_clusters(labels, min_size=5)
    assert len(clusters) == 2


def test_load_corrected_results(tmp_path):
    import nibabel as nib

    data = np.ones((2, 2, 2))
    img = nib.Nifti1Image(data, affine=np.eye(4))

    # Create directory structure and file
    vol_dir = tmp_path / "Results/MainEffect/Volumes"
    vol_dir.mkdir(parents=True)
    file_path = vol_dir / "test_vFWE.nii"
    nib.save(img, file_path)

    results = contribution.load_corrected_results(tmp_path, "test", "vFWE")
    assert results.shape == (2, 2, 2)
    assert np.all(results == 1)


@pytest.mark.parametrize(
    "contrib_vals",
    [
        np.array([[1, 2, 3, 6], [4, 5, 6, 7]]),  # 4 columns now
        np.zeros((2, 4)),
    ],
)
def test_write_experiment_contributions_output(contrib_vals, dummy_exp_df):
    # Use StringIO to capture writes
    buf = io.StringIO()
    contribution.write_experiment_contributions(buf, dummy_exp_df, contrib_vals)
    out_str = buf.getvalue()
    assert isinstance(out_str, str)
    # Should include at least one article name or be empty (depending on filter)
    if np.any(contrib_vals[:, 2] > 0.1) or np.any(contrib_vals[:, 3] > 5):
        assert (
            dummy_exp_df.Articles.iloc[0] in out_str
            or dummy_exp_df.Articles.iloc[1] in out_str
        )


def test_write_task_contributions_output(dummy_tasks):
    buf = io.StringIO()
    exp_idxs_full = np.array([0, 1, 2])
    contribution_arr = np.array(
        [
            [1.0, 2.0, 30.0, 0.0],
            [4.0, 5.0, 60.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    contribution.write_task_contributions(
        buf, dummy_tasks, exp_idxs_full, contribution_arr
    )
    out_str = buf.getvalue()
    assert "Task Contributions" in out_str
    assert "TaskA" in out_str or "TaskB" in out_str


@patch("jale.core.utils.contribution.compute_ma")
@patch("jale.core.utils.contribution.compute_ale")
@patch("jale.core.utils.contribution.create_kernel_array")
@patch("jale.core.utils.contribution.load_corrected_results")
@patch("builtins.open", new_callable=mock_open)
def test_contribution_integration(
    mock_file,
    mock_load_corrected_results,
    mock_create_kernel_array,
    mock_compute_ale,
    mock_compute_ma,
    dummy_exp_df,
    dummy_tasks,
):
    # Setup mocks
    mock_create_kernel_array.return_value = np.ones((len(dummy_exp_df), 3, 3, 3))
    mock_compute_ma.return_value = np.ones((len(dummy_exp_df), 3, 3, 3))
    mock_compute_ale.return_value = np.array([0.5, 0.5, 0.5])
    # Return binary volume with one cluster label 1, size > 5
    mock_load_corrected_results.return_value = np.ones((3, 3, 3))

    # exp_idxs_full: use a simple array
    exp_idxs_full = np.arange(len(dummy_exp_df))

    # Call function
    project_path = Path(".")
    contribution.contribution(
        project_path=project_path,
        exp_df=dummy_exp_df,
        exp_idxs_full=exp_idxs_full,
        exp_name="test_exp",
        tasks=dummy_tasks,
        tfce_enabled=False,
    )

    # Check file was opened for writing (3 methods: vFWE, cFWE, TFCE)
    assert mock_file.call_count >= 2
    handle = mock_file()
    written_content = "".join(call.args[0] for call in handle.write.call_args_list)
    assert (
        "Starting with test_exp!" in written_content
        or "No significant clusters found!" in written_content
    )
