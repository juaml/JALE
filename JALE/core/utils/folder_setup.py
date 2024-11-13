def folder_setup(path):
    """
    Set up a directory structure for storing analysis results.

    This function creates a nested folder structure under the specified path
    for organizing result files related to various analyses (e.g., MainEffect,
    Contrast). If the folders already exist, they will not be recreated.

    Parameters
    ----------
    path : Path or str
        Base path where the directory structure should be created.

    Returns
    -------
    None
    """

    # Define a dictionary for the folder structure
    # Each key represents a base directory and each value is a list of subdirectories to create
    folder_structure = {
        "Results/MainEffect/Full": [
            "Volumes",
            "Foci",
            "Contribution",
            "NullDistributions",
        ],
        "Results/MainEffect/CV": ["Volumes", "NullDistributions"],
        "Results/Contrast/Full": ["NullDistributions", "Conjunctions"],
        "Results/Contrast/Balanced": ["NullDistributions", "Conjunctions"],
    }

    # Iterate over the base directories and their subdirectories
    for base, subfolders in folder_structure.items():
        # Construct the base path
        basepath = path / base
        # Create each subfolder within the base path
        for folder in subfolders:
            (basepath / folder).mkdir(
                parents=True, exist_ok=True
            )  # Create folder, including any necessary parents

    (path / "logs").mkdir(parents=True, exist_ok=True)
    (path / "Results/MainEffect/ROI").mkdir(parents=True, exist_ok=True)

    (path / "Results/MA_Clustering").mkdir(parents=True, exist_ok=True)
