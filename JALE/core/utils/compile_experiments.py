import nibabel as nb
import numpy as np


def compile_experiments(conditions, tasks):
    """
    Process conditions to compile a list of experiments and corresponding masks.

    Parameters
    ----------
    conditions : list of str
        Conditions for experiment selection:
        - `+tag`: Include experiments that have tag. Logical AND
        - `-tag`: Exclude experiments that have tag. Logical NOT
        - `?`: Intersect included experiments. Logical OR
        - `$file`: Load mask from file.

    tasks : pandas.DataFrame
        DataFrame with 'Name' and 'ExpIndex' columns for experiment lookup.

    Returns
    -------
    exp_to_use : list
        List of experiment indices to use.

    masks : list of numpy.ndarray
        List of masks from files.

    mask_names : list of str
        List of mask file names without extensions.
    """
    included_experiments = set()
    excluded_experiments = set()
    masks = []
    mask_names = []

    for condition in conditions:
        operation = condition[0]
        tag = condition[1:].lower()

        # Check if the experiment exists in tasks and handle exceptions
        try:
            experiment_index = tasks[tasks.Name == tag].ExpIndex.values[0]
        except IndexError:
            raise ValueError(f"Experiment '{tag}' not found in tasks.")

        if operation == "+":
            included_experiments.update(experiment_index)

        elif operation == "-":
            excluded_experiments.update(experiment_index)

        elif operation == "?":
            # Intersect experiments in included_experiments
            included_experiments = set(
                included_experiments
            )  # Ensure unique entries only

        elif operation == "$":
            mask_file = condition[1:]
            mask = nb.loadsave.load(mask_file).get_fdata()

            if np.unique(mask).shape[0] == 2:
                # Binary mask
                masks.append(mask.astype(bool))
            else:
                # Labeled mask
                masks.append(mask.astype(int))
            mask_names.append(mask_file[:-4])

    # Apply difference between included and excluded experiments
    included_experiments = included_experiments.difference(excluded_experiments)

    # Convert back to list for final result
    included_experiments = list(included_experiments)

    return included_experiments, masks, mask_names
