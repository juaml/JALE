"""This module provides a function to set up a directory structure for storing analysis results."""

from pathlib import Path


def folder_setup(path, type_of_analysis):
    """
    Set up a directory structure for storing analysis results.

    This function creates a nested folder structure under the specified path
    for organizing result files related to various analyses. The structure
    depends on the specified type of analysis.

    Parameters
    ----------
    path : Path or str
        Base path where the directory structure should be created.

    type_of_analysis : str
        Type of analysis to set up folders for. Options are:
        - "MainEffect"
        - "Probabilistic"
        - "Contrast"
        - "BalancedContrast"
        - "MA_Clustering"
        - "ROI"

    Returns
    -------
    None
    """

    # Convert path to Path object if it's a string
    path = Path(path)

    # Define folder structures for each analysis type
    analysis_folders = {
        "MainEffect": [
            "Results/MainEffect/Volumes",
            "Results/MainEffect/Contribution",
            "Results/MainEffect/NullDistributions",
            "Results/MainEffect/Figures",
        ],
        "Probabilistic": [
            "Results/Probabilistic/Volumes",
            "Results/Probabilistic/NullDistributions",
            "Results/Probabilistic/Figures",
        ],
        "Contrast": [
            "Results/Contrast/Volumes",
            "Results/Contrast/NullDistributions",
            "Results/Contrast/Figures",
        ],
        "BalancedContrast": [
            "Results/BalancedContrast/Volumes",
            "Results/BalancedContrast/NullDistributions",
            "Results/BalancedContrast/Figures",
        ],
        "MA_Clustering": [
            "Results/MA_Clustering/tmp",
            "Results/MA_Clustering/labels",
            "Results/MA_Clustering/dendograms",
        ],
        "ROI": [
            "Results/MainEffect/ROI",
        ],
    }

    # Validate the type_of_analysis
    if type_of_analysis not in analysis_folders:
        raise ValueError(
            f"Invalid analysis type: {type_of_analysis}. "
            "Valid options are: MainEffect, Probabilistic, Contrast, BalancedContrast, MA_Clustering, ROI."
        )

    # Get the folder structure for the selected analysis type
    folders_to_create = analysis_folders[type_of_analysis]

    # Create the directories for the selected analysis type
    for folder in folders_to_create:
        (path / folder).mkdir(parents=True, exist_ok=True)
