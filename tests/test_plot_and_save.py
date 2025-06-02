from unittest.mock import patch

import numpy as np
import pytest

from jale.core.utils.plot_and_save import plot_and_save


@pytest.fixture
def project_path(tmp_path):
    # Provide a temporary directory as project_path
    return tmp_path


def test_plot_and_save_saves_files_and_plots(project_path):
    arr = np.ones((5, 5, 5))  # array with positive values

    with (
        patch("jale.core.utils.template.nb.loadsave.save") as mock_save,
        patch("nilearn.plotting.plot_stat_map") as mock_plot,
    ):
        plot_and_save(project_path, "test_analysis", arr)

        # Check plotting called once
        mock_plot.assert_called_once()
        # Check save called once to expected path
        expected_path = project_path / "Volumes" / "test_analysis"
        mock_save.assert_called_once()
        assert mock_save.call_args[0][1] == expected_path


def test_plot_and_save_skips_plot_when_no_positive(project_path):
    arr = np.zeros((5, 5, 5))  # no positive values

    with (
        patch("jale.core.utils.template.nb.loadsave.save") as mock_save,
        patch("nilearn.plotting.plot_stat_map") as mock_plot,
    ):
        plot_and_save(project_path, "test_analysis", arr)

        # Plot should not be called
        mock_plot.assert_not_called()
        # Save called once with "_empty" suffix
        expected_path = project_path / "Volumes" / "test_analysis_empty"
        mock_save.assert_called_once()
        assert mock_save.call_args[0][1] == expected_path


def test_plot_and_save_replaces_nan_and_inf(project_path):
    arr = np.array([np.nan, np.inf, -np.inf, 1, 0])

    with (
        patch("jale.core.utils.template.nb.loadsave.save") as mock_save,
        patch("nilearn.plotting.plot_stat_map") as _,
    ):
        plot_and_save(project_path, "test_analysis", arr)

        # Extract the saved Nifti image argument (first argument of save)
        saved_img = mock_save.call_args[0][0]
        data = saved_img.get_fdata()

        # NaN and inf should be replaced by 0
        assert np.all(np.isfinite(data))
        assert data[0] == 0
        assert data[1] == 0
        assert data[2] == 0
        # Positive value remains
        assert data[3] == 1
