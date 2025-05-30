import pytest

from jale.core.utils.folder_setup import (
    folder_setup,
)


@pytest.mark.parametrize(
    "analysis_type, expected_dirs",
    [
        (
            "MainEffect",
            [
                "Results/MainEffect/Volumes",
                "Results/MainEffect/Contribution",
                "Results/MainEffect/NullDistributions",
                "Results/MainEffect/Figures",
            ],
        ),
        (
            "Probabilistic",
            [
                "Results/Probabilistic/Volumes",
                "Results/Probabilistic/NullDistributions",
                "Results/Probabilistic/Figures",
            ],
        ),
        (
            "Contrast",
            [
                "Results/Contrast/Volumes",
                "Results/Contrast/NullDistributions",
                "Results/Contrast/Figures",
            ],
        ),
        (
            "BalancedContrast",
            [
                "Results/BalancedContrast/Volumes",
                "Results/BalancedContrast/NullDistributions",
                "Results/BalancedContrast/Figures",
            ],
        ),
        (
            "MA_Clustering",
            [
                "Results/MA_Clustering/tmp",
                "Results/MA_Clustering/labels",
                "Results/MA_Clustering/dendograms",
            ],
        ),
        (
            "ROI",
            [
                "Results/MainEffect/ROI",
            ],
        ),
    ],
)
def test_folder_setup_creates_dirs(tmp_path, analysis_type, expected_dirs):
    folder_setup(tmp_path, analysis_type)

    # Check each expected folder was created
    for folder in expected_dirs:
        dir_path = tmp_path / folder
        assert dir_path.exists()
        assert dir_path.is_dir()


def test_folder_setup_invalid_type(tmp_path):
    with pytest.raises(ValueError, match="Invalid analysis type"):
        folder_setup(tmp_path, "InvalidAnalysisType")
