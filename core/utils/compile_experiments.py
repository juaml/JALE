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
    included_experiments = []
    excluded_experiments = []
    masks = []
    mask_names = []

    for condition in conditions:
        operation = condition[0]
        tag = condition[1:].lower()

        # Check if the experiment exists in tasks and handle exceptions
        try:
            experiment_index = tasks[tasks.Name == tag].ExpIndex.to_list()[0]
        except IndexError:
            raise ValueError(f"Experiment '{tag}' not found in tasks.")

        if operation == "+":
            included_experiments.append(experiment_index)

        elif operation == "-":
            excluded_experiments.append(experiment_index)

        elif operation == "?":
            # Intersect experiments in included_experiments
            flat_list = [exp for exp_list in included_experiments for exp in exp_list]
            # Deduplicate and reassign
            included_experiments = [list(set(flat_list))]

        elif operation == "$":
            mask_file = condition[1:]
            mask = nb.load(mask_file).get_fdata()  # type: ignore

            if np.unique(mask).shape[0] == 2:
                # Binary mask
                masks.append(mask.astype(bool))
            else:
                # Labeled mask
                masks.append(mask.astype(int))
            mask_names.append(mask_file[:-4])

    # Convert included_experiments and excluded_experiments to sets
    included_set = set(included_experiments)

    if excluded_experiments:
        excluded_set = set(excluded_experiments)
        included_set = included_set.difference(excluded_set)

    # Convert back to list for final result
    included_experiments = list(included_set)

    return included_experiments, masks, mask_names
