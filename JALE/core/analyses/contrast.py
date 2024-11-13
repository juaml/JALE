import logging
import pickle
from pathlib import Path

import nibabel as nb
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm

from jale.core.utils.compute import (
    compute_ale,
    compute_balanced_ale_diff,
    compute_balanced_null_diff,
    compute_permuted_ale_diff,
    compute_sig_diff,
)
from jale.core.utils.plot_and_save import plot_and_save
from jale.core.utils.template import BRAIN_ARRAY_SHAPE, GM_PRIOR

logger = logging.getLogger("ale_logger")


def contrast(
    project_path,
    meta_names,
    significance_threshold=0.05,
    null_repeats=10000,
    nprocesses=2,
):
    """
    Compute and save statistical contrasts and conjunctions for meta-analyses.

    This function calculates positive and negative contrasts, as well as conjunctions,
    between two meta-analyses specified by `meta_names`. If these results are already
    available, they are loaded from the saved files. Otherwise, the function computes
    the contrasts by estimating a null distribution through permutation testing, identifies
    significant voxels, and saves the results as NIfTI images.

    Parameters
    ----------
    project_path : str or Path
        Path to the project directory containing the "Results" folder.
    meta_names : list of str
        Names of the meta-analyses to compare; expects two names in the list.
    significance_threshold : float, optional
        Significance threshold for identifying significant voxels, by default 0.05.
    null_repeats : int, optional
        Number of permutations for generating the null distribution, by default 10000.
    nprocesses : int, optional
        Number of parallel processes for permutation testing, by default 4.

    Returns
    -------
    None
        The function performs computations and saves the results as NIfTI files in the
        specified `project_path` directory.
    """

    # set results folder as path
    project_path = (Path(project_path) / "Results").resolve()

    ma1 = np.load(project_path / f"MainEffect/{meta_names[0]}_ma.npz")["arr_0"]
    ale1 = compute_ale(ma1)
    n_meta_group1 = ma1.shape[0]

    ma2 = np.load(project_path / f"MainEffect/{meta_names[1]}_ma.npz")["arr_0"]
    ale2 = compute_ale(ma1)
    n_meta_group2 = ma2.shape[0]

    # Check if contrast has already been calculated
    if Path(
        project_path / f"Contrast/Full/{meta_names[0]}_vs_{meta_names[1]}.nii"
    ).exists():
        logger.info(f"{meta_names[0]} x {meta_names[1]} - Loading contrast.")
        contrast_arr = nb.loadsave.load(
            project_path / f"Contrast/Full/{meta_names[0]}_vs_{meta_names[1]}.nii"
        ).get_fdata()  # type: ignore
    else:
        logger.info(f"{meta_names[0]} x {meta_names[1]} - Computing positive contrast.")  # noqa
        main_effect1 = nb.loadsave.load(
            project_path / f"MainEffect/Full/Volumes/{meta_names[0]}_cFWE05.nii"
        ).get_fdata()  # type: ignore
        significance_mask1 = main_effect1 > 0
        if significance_mask1.sum() > 0:
            stacked_masked_ma = np.vstack(
                (ma1[:, significance_mask1], ma2[:, significance_mask1])
            )

            ale_difference1 = ale1 - ale2
            # estimate null distribution of difference values if studies
            # would be randomly assigned to either meta analysis
            null_difference1 = Parallel(n_jobs=nprocesses)(
                delayed(compute_permuted_ale_diff)(stacked_masked_ma, n_meta_group1)
                for i in range(null_repeats)
            )
            z1, sig_idxs1 = compute_sig_diff(
                ale_difference1[significance_mask1],
                null_difference1,
                significance_threshold,
            )

        else:
            logger.warning(f"{meta_names[0]}: No significant indices!")
            z1, sig_idxs1 = [], []

        logger.info(f"{meta_names[1]} x {meta_names[0]} - Computing negative contrast.")
        main_effect2 = nb.loadsave.load(
            project_path / f"MainEffect/Full/Volumes/{meta_names[1]}_cFWE05.nii"
        ).get_fdata()  # type: ignore
        significance_mask2 = main_effect2 > 0
        if significance_mask2.sum() > 0:
            stacked_masked_ma = np.vstack(
                (ma1[:, significance_mask2], ma2[:, significance_mask2])
            )
            ale_difference2 = ale2 - ale1
            null_difference2 = Parallel(n_jobs=nprocesses)(
                delayed(compute_permuted_ale_diff)(stacked_masked_ma, n_meta_group2)
                for i in range(null_repeats)
            )
            z2, sig_idxs2 = compute_sig_diff(
                ale_difference2[significance_mask2],
                null_difference2,
                significance_threshold,
            )

        else:
            logger.warning(f"{meta_names[1]}: No significant indices!")
            z2, sig_idxs2 = np.array([]), []

        logger.info(f"{meta_names[0]} vs {meta_names[1]} - Inference and printing.")
        contrast_arr = np.zeros(BRAIN_ARRAY_SHAPE)
        contrast_arr[significance_mask1][sig_idxs1] = z1
        contrast_arr[significance_mask2][sig_idxs2] = -z2
        plot_and_save(
            contrast_arr,
            nii_path=project_path
            / f"Contrast/Full/{meta_names[0]}_vs_{meta_names[1]}_cFWE.nii",
        )

    # Check if conjunction has already been calculated
    if Path(
        project_path / f"Contrast/Full/{meta_names[0]}_AND_{meta_names[1]}_cFWE.nii"
    ).exists():
        logger.info(f"{meta_names[0]} & {meta_names[1]} - Loading conjunction.")
        conj_arr = nb.loadsave.load(
            project_path / f"Contrast/Full/{meta_names[0]}_AND_{meta_names[1]}_cFWE.nii"
        ).get_fdata()  # type: ignore
    else:
        logger.info(f"{meta_names[0]} & {meta_names[1]} - Computing conjunction.")
        conj_arr = np.minimum(main_effect1, main_effect2)
        if conj_arr is not None:
            plot_and_save(
                conj_arr,
                nii_path=project_path
                / f"Contrast/Full/{meta_names[0]}_AND_{meta_names[1]}_cFWE.nii",
            )

    logger.info(f"{meta_names[0]} & {meta_names[1]} - done!")


def balanced_contrast(
    project_path,
    exp_dfs,
    meta_names,
    target_n,
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

    # set results folder as path
    project_path = (Path(project_path) / "Results").resolve()

    kernels1 = np.load(project_path / f"MainEffect/{meta_names[0]}_kernels.npy")

    kernels2 = np.load(project_path / f"MainEffect/{meta_names[1]}_kernels.npy")

    ma1 = np.load(project_path / f"MainEffect/{meta_names[0]}_ma.npz")["arr_0"]

    ma2 = np.load(project_path / f"MainEffect/{meta_names[1]}_ma.npz")["arr_0"]

    main_effect1 = nb.loadsave.load(
        project_path / f"MainEffect/CV/Volumes/{meta_names[0]}_sub_ale_{target_n}.nii"
    ).get_fdata()  # type: ignore
    main_effect2 = nb.loadsave.load(
        project_path / f"MainEffect/CV/Volumes/{meta_names[1]}_sub_ale_{target_n}.nii"
    ).get_fdata()  # type: ignore

    if not Path(
        project_path
        / f"Contrast/Conjunctions/{meta_names[0]}_AND_{meta_names[1]}_{target_n}.nii"
    ).exists():
        logger.info(f"{meta_names[0]} x {meta_names[1]} - computing conjunction")
        conjunction = np.minimum(main_effect1, main_effect2)
        conjunction = plot_and_save(
            conjunction,
            nii_path=project_path
            / f"Contrast/Balanced/Conjunctions/{meta_names[0]}_AND_{meta_names[1]}_{target_n}.nii",
        )

    if Path(
        project_path
        / f"Contrast/Balanced/NullDistributions/{meta_names[0]}_x_{meta_names[1]}_{target_n}.pickle"
    ).exists():
        logger.info(
            f"{meta_names[0]} x {meta_names[1]} - loading actual diff and null extremes"
        )
        with open(
            project_path
            / f"Contrast/Balanced/NullDistributions/{meta_names[0]}_x_{meta_names[1]}_{target_n}.pickle",
            "rb",
        ) as f:
            r_diff, prior, min_diff, max_diff = pickle.load(f)
    else:
        logger.info(
            f"{meta_names[0]} x {meta_names[1]} - computing average subsample difference"
        )
        prior = np.zeros(BRAIN_ARRAY_SHAPE).astype(bool)
        prior[GM_PRIOR] = 1

        r_diff = Parallel(n_jobs=nprocesses, verbose=2)(
            delayed(compute_balanced_ale_diff)(ma1, ma2, prior, target_n)
            for i in range(difference_iterations)
        )
        r_diff = np.mean(np.array(r_diff), axis=0)

        logger.info(
            f"{meta_names[0]} x {meta_names[1]} - computing null distribution of balanced differences"
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
            / f"Contrast/Balanced/NullDistributions/{meta_names[0]}_x_{meta_names[1]}_{target_n}.pickle",
            "wb",
        ) as f:
            pickle.dump(pickle_object, f)

    if not Path(
        f"Contrast/Balanced/{meta_names[0]}_x_{meta_names[1]}_{target_n}_vFWE05.nii"
    ).exists():
        logger.info(
            f"{meta_names[0]} x {meta_names[1]} - computing significant contrast"
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
            brain_sig_diff,
            nii_path=project_path
            / f"Contrast/Balanced/{meta_names[0]}_x_{meta_names[1]}_{target_n}_FWE05.nii",
        )

    logger.info(
        f"{meta_names[0]} x {meta_names[1]} balanced (n = {target_n}) contrast done!"
    )
