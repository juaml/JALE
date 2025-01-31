import logging
from pathlib import Path

import nibabel as nb
import numpy as np
from joblib import Parallel, delayed

from jale.core.utils.compute import (
    compute_ale,
    compute_permuted_ale_diff,
    compute_sig_diff,
)
from jale.core.utils.folder_setup import folder_setup
from jale.core.utils.plot_and_save import plot_and_save
from jale.core.utils.template import BRAIN_ARRAY_SHAPE

logger = logging.getLogger("ale_logger")


def contrast(
    project_path,
    meta_names,
    correction_method="cFWE",
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
        Significance threshold for identifying significant voxels, by default 0..
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
    folder_setup(project_path, "Contrast")
    project_path = (Path(project_path) / "Results/").resolve()

    ma1 = np.load(project_path / f"MainEffect/{meta_names[0]}_ma.npz")["arr_0"]
    ale1 = compute_ale(ma1)
    n_meta_group1 = ma1.shape[0]

    ma2 = np.load(project_path / f"MainEffect/{meta_names[1]}_ma.npz")["arr_0"]
    ale2 = compute_ale(ma2)
    n_meta_group2 = ma2.shape[0]

    # Check if contrast has already been calculated
    if Path(
        project_path / f"Contrast/Volumes/{meta_names[0]}_vs_{meta_names[1]}.nii"
    ).exists():
        logger.info(
            f"{meta_names[0]} x {meta_names[1]} - contrast already computed. Skipping."
        )
    else:
        logger.info(f"{meta_names[0]} x {meta_names[1]} - Computing positive contrast.")
        main_effect1 = nb.loadsave.load(
            project_path / f"MainEffect/Volumes/{meta_names[0]}_{correction_method}.nii"
        ).get_fdata()
        significance_mask1 = main_effect1 > 0
        if significance_mask1.sum() > 0:
            stacked_masked_ma = np.vstack(
                (ma1[:, significance_mask1], ma2[:, significance_mask1])
            )

            ale_difference1 = ale1 - ale2
            # estimate null distribution of difference values if studies
            # would be randomly assigned to either meta analysis
            null_difference1 = Parallel(n_jobs=nprocesses, verbose=2)(
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
            project_path / f"MainEffect/Volumes/{meta_names[1]}_{correction_method}.nii"
        ).get_fdata()  # type: ignore
        significance_mask2 = main_effect2 > 0
        if significance_mask2.sum() > 0:
            stacked_masked_ma = np.vstack(
                (ma1[:, significance_mask2], ma2[:, significance_mask2])
            )
            ale_difference2 = ale2 - ale1
            null_difference2 = Parallel(n_jobs=nprocesses, verbose=2)(
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
        flat_mask1 = np.where(significance_mask1)
        contrast_arr[
            flat_mask1[0][sig_idxs1], flat_mask1[1][sig_idxs1], flat_mask1[2][sig_idxs1]
        ] = z1

        flat_mask2 = np.where(significance_mask2)
        contrast_arr[
            flat_mask2[0][sig_idxs2], flat_mask2[1][sig_idxs2], flat_mask2[2][sig_idxs2]
        ] = -z2
        plot_and_save(
            project_path / "Contrast",
            f"{meta_names[0]}_vs_{meta_names[1]}",
            contrast_arr,
        )

    # Check if conjunction has already been calculated
    if Path(
        project_path / f"Contrast/Volumes/{meta_names[0]}_AND_{meta_names[1]}.nii"
    ).exists():
        logger.info(
            f"{meta_names[0]} & {meta_names[1]} - Conjunction already computed. Skipping."
        )
    else:
        logger.info(f"{meta_names[0]} & {meta_names[1]} - Computing conjunction.")
        conj_arr = np.minimum(main_effect1, main_effect2)
        if conj_arr is not None:
            plot_and_save(
                project_path / "Contrast",
                f"{meta_names[0]}_AND_{meta_names[1]}",
                conj_arr,
            )

    logger.info(f"{meta_names[0]} & {meta_names[1]} - done!")
