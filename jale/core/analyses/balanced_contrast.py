import logging
import pickle
from pathlib import Path

import nibabel as nb
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm

from jale.core.utils.compute import (
    compute_balanced_ale_diff,
    compute_balanced_null_diff,
)
from jale.core.utils.folder_setup import folder_setup
from jale.core.utils.plot_and_save import plot_and_save
from jale.core.utils.template import BRAIN_ARRAY_SHAPE, GM_PRIOR

logger = logging.getLogger("ale_logger")


def balanced_contrast(
    project_path,
    exp_dfs,
    meta_names,
    target_n,
    correction_method="cFWE",
    difference_iterations=1000,
    monte_carlo_iterations=1000,
    nprocesses=2,
):
    """
    Compute and save balanced statistical contrasts between two meta-analyses.

    This function performs a balanced contrast analysis between two meta-analyses, specified
    by `meta_names`, with matched sample sizes (`target_n`). The function calculates both
    conjunctions and significant contrasts. If results are already available, they are loaded;
    otherwise, the function computes the required contrasts by estimating null distributions
    through Monte Carlo sampling and permutation testing.

    Parameters
    ----------
    project_path : str or Path
        Path to the project directory containing the "Results" folder.
    exp_dfs : list of pandas.DataFrame
        DataFrames for each meta-analysis, containing information on experimental data.
    meta_names : list of str
        Names of the meta-analyses to compare; expects two names in the list.
    target_n : int
        Target number of samples for balanced analysis.
    difference_iterations : int, optional
        Number of iterations for computing the difference distribution, by default 1000.
    monte_carlo_iterations : int, optional
        Number of Monte Carlo iterations for estimating null distributions, by default 1000.
    nprocesses : int, optional
        Number of parallel processes for computation, by default 2.

    Returns
    -------
    None
        The function performs computations and saves the results as NIfTI files in the
        specified `project_path` directory.
    """
    folder_setup(project_path, "BalancedContrast")

    # set results folder as path
    project_path = (Path(project_path) / "Results/").resolve()

    kernels1 = np.load(project_path / f"Probabilistic/{meta_names[0]}_kernels.npy")

    kernels2 = np.load(project_path / f"Probabilistic/{meta_names[1]}_kernels.npy")

    ma1 = np.load(project_path / f"Probabilistic/{meta_names[0]}_ma.npz")["arr_0"]

    ma2 = np.load(project_path / f"Probabilistic/{meta_names[1]}_ma.npz")["arr_0"]

    main_effect1 = nb.loadsave.load(
        project_path / f"Probabilistic/Volumes/{meta_names[0]}_sub_ale_{target_n}.nii"
    ).get_fdata()  # type: ignore
    main_effect2 = nb.loadsave.load(
        project_path / f"Probabilistic/Volumes/{meta_names[1]}_sub_ale_{target_n}.nii"
    ).get_fdata()  # type: ignore

    if not Path(
        project_path
        / f"BalancedContrast/Volumes/{meta_names[0]}_AND_{meta_names[1]}_{target_n}.nii"
    ).exists():
        logger.info(f"{meta_names[0]} x {meta_names[1]} - computing conjunction")
        conjunction = np.minimum(main_effect1, main_effect2)
        conjunction = plot_and_save(
            project_path / "BalancedContrast",
            f"{meta_names[0]}_AND_{meta_names[1]}_{target_n}",
            conjunction,
        )

    if Path(
        project_path
        / f"BalancedContrast/NullDistributions/{meta_names[0]}_x_{meta_names[1]}_{target_n}.pickle"
    ).exists():
        logger.info(
            f"{meta_names[0]} vs {meta_names[1]} - loading actual diff and null extremes"
        )
        with open(
            project_path
            / f"BalancedContrast/NullDistributions/{meta_names[0]}_x_{meta_names[1]}_{target_n}.pickle",
            "rb",
        ) as f:
            r_diff, prior, min_diff, max_diff = pickle.load(f)
    else:
        logger.info(
            f"{meta_names[0]} vs {meta_names[1]} - computing average subsample difference"
        )
        prior = np.zeros(BRAIN_ARRAY_SHAPE).astype(bool)
        prior[GM_PRIOR] = 1

        r_diff = Parallel(n_jobs=nprocesses, verbose=2)(
            delayed(compute_balanced_ale_diff)(ma1, ma2, prior, target_n)
            for i in range(difference_iterations)
        )
        r_diff = np.mean(np.array(r_diff), axis=0)

        logger.info(
            f"{meta_names[0]} vs {meta_names[1]} - computing null distribution of balanced differences"
        )
        nfoci1 = exp_dfs[0].NumberOfFoci
        nfoci2 = exp_dfs[1].NumberOfFoci
        min_diff, max_diff = zip(
            *Parallel(n_jobs=nprocesses, verbose=2)(
                delayed(compute_balanced_null_diff)(
                    nfoci1,
                    kernels1,
                    nfoci2,
                    kernels2,
                    prior,
                    target_n,
                    difference_iterations,
                )
                for i in range(monte_carlo_iterations)
            )
        )

        pickle_object = (r_diff, prior, min_diff, max_diff)
        with open(
            project_path
            / f"BalancedContrast/NullDistributions/{meta_names[0]}_vs_{meta_names[1]}_{target_n}.pickle",
            "wb",
        ) as f:
            pickle.dump(pickle_object, f)

    if not Path(
        f"BalancedContrast/Volumes/{meta_names[0]}_vs_{meta_names[1]}_{target_n}_vFWE05.nii"
    ).exists():
        logger.info(
            f"{meta_names[0]} vs {meta_names[1]} - computing significant contrast"
        )

        # Calculate thresholds
        low_threshold = np.percentile(min_diff, 2.5)
        high_threshold = np.percentile(max_diff, 97.5)

        # Identify significant differences
        is_significant = np.logical_or(r_diff < low_threshold, r_diff > high_threshold)
        sig_diff = r_diff * is_significant

        # Calculate z-values for positive differences
        positive_diffs = sig_diff > 0
        sig_diff[positive_diffs] = [
            -1 * norm.ppf((np.sum(max_diff >= diff)) / monte_carlo_iterations)
            for diff in sig_diff[positive_diffs]
        ]

        # Calculate z-values for negative differences
        negative_diffs = sig_diff < 0
        sig_diff[negative_diffs] = [
            norm.ppf((np.sum(min_diff <= diff)) / monte_carlo_iterations)
            for diff in sig_diff[negative_diffs]
        ]

        # Create the final brain difference map
        brain_sig_diff = np.zeros(BRAIN_ARRAY_SHAPE)
        brain_sig_diff[prior] = sig_diff

        plot_and_save(
            project_path / "BalancedContrast",
            f"{meta_names[0]}_vs_{meta_names[1]}_{target_n}_vFWE05",
            brain_sig_diff,
        )

    logger.info(
        f"{meta_names[0]} vs {meta_names[1]} balanced (n = {target_n}) contrast done!"
    )
