import nibabel as nb
import numpy as np


def compile_experiments(conditions, tasks):
    """
    Process conditions to compile a list of experiments and corresponding masks.

    Parameters
    ----------
    conditions : list of str
        Conditions for experiment selection:
        - `+tag`: Include experiments that have tag. Logical AND.
        - `-tag`: Exclude experiments that have tag. Logical NOT.
        - `?`: Combine included experiments (Logical OR).
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
    exp_to_use = []
    not_to_use = []
    masks = []
    mask_names = []

    for condition in conditions:
        operation = condition[0]
        argument = condition[1:]

        if operation == "+":
            # Logical AND
            tag = argument.lower()
            matches = tasks[tasks["Name"].str.lower() == tag]["ExpIndex"].values
            if not matches:
                raise ValueError(f"No experiments found for tag: {tag}")
            exp_to_use.append(matches[0])

        elif operation == "-":
            # Logical NOT
            tag = argument.lower()
            matches = tasks[tasks["Name"].str.lower() == tag]["ExpIndex"].values
            if not matches:
                raise ValueError(f"No experiments found for tag: {tag}")
            not_to_use.append(matches[0])

        elif operation == "?":
            # Logical OR: Combine all included experiments
            flat_list = [idx for sublist in exp_to_use for idx in sublist]
            exp_to_use = [list(set(flat_list))]

        elif operation == "$":
            # Load mask from file
            try:
                mask = nb.load(argument).get_fdata()
            except FileNotFoundError:
                raise FileNotFoundError(f"Mask file not found: {argument}")
            except Exception as e:
                raise ValueError(f"Error loading mask file {argument}: {e}")

            # Check binary or multi-class mask
            if np.unique(mask).size == 2:
                masks.append(mask.astype(bool))
            else:
                masks.append(mask.astype(int))

            mask_names.append(argument.rsplit(".", 1)[0])

    # Compute final set of experiments to use
    if exp_to_use:
        use_sets = map(set, exp_to_use)
        exp_to_use = list(set.intersection(*use_sets))
        if len(exp_to_use) == 0:
            raise ValueError("Bad tag combination. No experiments found.")
    else:
        raise ValueError("Bad tag combination. No experiments found.")

    # Remove excluded experiments
    if not_to_use:
        not_use_sets = map(set, not_to_use)
        not_to_use = list(set.union(*not_use_sets))
        exp_to_use = list(set(exp_to_use) - set(not_to_use))

    return exp_to_use, masks, mask_names
