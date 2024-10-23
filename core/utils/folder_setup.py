def folder_setup(path):
    # Define a dictionary for directory structures
    folder_structure = {
        "Results/MainEffect/Full": [
            "Volumes",
            "Foci",
            "Contribution",
            "NullDistributions",
        ],
        "Results/MainEffect/CV": ["Volumes", "NullDistributions"],
        "Results/MainEffect/ROI": ["Plots", "NullDistributions"],
        "Results/Contrast/Full": ["NullDistributions", "Conjunctions"],
        "Results/Contrast/Balanced": ["NullDistributions", "Conjunctions"],
        "Results/Contrast/ROI": ["Plots", "NullDistributions"],
    }

    # Loop over the dictionary and create directories
    for base, subfolders in folder_structure.items():
        basepath = path / base
        for folder in subfolders:
            (basepath / folder).mkdir(parents=True, exist_ok=True)
