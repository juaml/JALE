from pathlib import Path
import nibabel as nb
import numpy as np
import pickle
from scipy.stats import norm
from joblib import Parallel, delayed
from core.utils.template import BRAIN_ARRAY_SHAPE, GM_PRIOR
from core.utils.compute import (
    compute_ale, compute_permuted_ale_diff, compute_sig_diff, plot_and_save,
    compute_balanced_ale_diff, compute_balanced_null_diff
)


def contrast(project_path,
             meta_names,
             significance_threshold=0.05,
             null_repeats=10000,
             nprocesses=4):
    # set results folder as path
    project_path = (Path(project_path) / "Results").resolve()

    ma1 = np.load(project_path /
                  f"MainEffect/{meta_names[0]}_ma.npy")
    ale1 = compute_ale(ma1)
    n_meta_group1 = ma1.shape[0]

    ma2 = np.load(project_path /
                  f"MainEffect/{meta_names[1]}_ma.npy")
    ale2 = compute_ale(ma1)
    n_meta_group2 = ma2.shape[0]

    # Check if contrast has already been calculated
    if Path(project_path /
            f"Contrast/Full/{meta_names[0]}_vs_{meta_names[1]}.nii").exists():
        print(f"{meta_names[0]} x {meta_names[1]} - Loading contrast.")
        contrast_arr = nb.load(
            project_path /
            f"Contrast/Full/{meta_names[0]}_vs_{meta_names[1]}.nii").get_fdata()
    else:
        print(f"{meta_names[0]} x {meta_names[1]} - Computing positive contrast.")  # noqa
        main_effect1 = nb.load(
            project_path /
            f"MainEffect/Full/Volumes/{meta_names[0]}_cFWE05.nii").get_fdata()
        significance_mask1 = main_effect1 > 0
        if significance_mask1.sum() > 0:
            stacked_masked_ma = np.vstack(
                (ma1[:, significance_mask1],
                 ma2[:, significance_mask1])
            )

            ale_difference1 = ale1 - ale2
            # estimate null distribution of difference values if studies
            # would be randomly assigned to either meta analysis
            null_difference1 = Parallel(n_jobs=nprocesses)(
                delayed(compute_permuted_ale_diff)(stacked_masked_ma,
                                                   n_meta_group1) for i in range(null_repeats))
            z1, sig_idxs1 = compute_sig_diff(ale_difference1[significance_mask1],
                                             null_difference1,
                                             significance_threshold)

        else:
            print(f"{meta_names[0]}: No significant indices!")
            z1, sig_idxs1 = [], []

        print(
            f"{meta_names[1]} x {meta_names[0]} - Computing negative contrast.")
        main_effect2 = nb.load(
            project_path /
            f"MainEffect/Full/Volumes/{meta_names[1]}_cFWE05.nii").get_fdata()
        significance_mask2 = main_effect2 > 0
        if significance_mask2.sum() > 0:
            stacked_masked_ma = np.vstack(
                (ma1[:, significance_mask2],
                 ma2[:, significance_mask2])
            )
            ale_difference2 = ale2 - ale1
            null_difference2 = Parallel(n_jobs=nprocesses)(
                delayed(compute_permuted_ale_diff)(stacked_masked_ma,
                                                   n_meta_group2) for i in range(null_repeats))
            z2, sig_idxs2 = compute_sig_diff(ale_difference2[significance_mask2],
                                             null_difference2,
                                             significance_threshold)

        else:
            print(f"{meta_names[1]}: No significant indices!")
            z2, sig_idxs2 = [], []

        print(f"{meta_names[0]} vs {meta_names[1]} - Inference and printing.")
        contrast_arr = np.zeros(BRAIN_ARRAY_SHAPE)
        contrast_arr[significance_mask1][sig_idxs1] = z1
        contrast_arr[significance_mask2][sig_idxs2] = -z2
        plot_and_save(contrast_arr,
                      nii_path=project_path /
                      f"Contrast/Full/{meta_names[0]}_vs_{meta_names[1]}_cFWE.nii")

    # Check if conjunction has already been calculated
    if Path(project_path /
            f"Contrast/Full/{meta_names[0]}_AND_{meta_names[1]}_cFWE.nii").exists():
        print(f"{meta_names[0]} & {meta_names[1]} - Loading conjunction.")
        conj_arr = nb.load(project_path /
                           f"Contrast/Full/{meta_names[0]}_AND_{meta_names[1]}_cFWE.nii").get_fdata()
    else:
        print(f"{meta_names[0]} & {meta_names[1]} - Computing conjunction.")
        conj_arr = np.minimum(main_effect1, main_effect2)
        if conj_arr is not None:
            plot_and_save(conj_arr,
                          nii_path=project_path /
                          f"Contrast/Full/{meta_names[0]}_AND_{meta_names[1]}_cFWE.nii")

    print(f"{meta_names[0]} & {meta_names[1]} - done!")


def balanced_contrast(project_path,
                      exp_dfs,
                      meta_names,
                      target_n,
                      difference_iterations=1000,
                      monte_carlo_iterations=1000,
                      nprocesses=2):

    # set results folder as path
    project_path = (Path(project_path) / "Results").resolve()

    kernels1 = np.load(project_path /
                       f"MainEffect/{meta_names[0]}_kernels.npy")

    kernels2 = np.load(project_path /
                       f"MainEffect/{meta_names[1]}_kernels.npy")

    ma1 = np.load(project_path /
                  f"MainEffect/{meta_names[0]}_ma.npy")

    ma2 = np.load(project_path /
                  f"MainEffect/{meta_names[1]}_ma.npy")

    main_effect1 = nb.load(project_path /
                           f"MainEffect/CV/Volumes/{meta_names[0]}_{target_n}.nii").get_fdata()
    main_effect2 = nb.load(project_path /
                           f"MainEffect/CV/Volumes/{meta_names[1]}_{target_n}.nii").get_fdata()

    if not Path(project_path /
                f"Contrast/Conjunctions/{meta_names[0]}_AND_{meta_names[1]}_{target_n}.nii").exists():
        print(f'{meta_names[0]} x {meta_names[1]} - computing conjunction')
        conjunction = np.minimum(main_effect1, main_effect2)
        conjunction = plot_and_save(
            conjunction,
            nii_folder=f"Contrast/Conjunctions/{meta_names[0]}_AND_{meta_names[1]}_{target_n}.nii")

    if Path(f"Contrast/NullDistributions/{meta_names[0]}_x_{meta_names[1]}_{target_n}.pickle").exists():
        print(f'{meta_names[0]} x {meta_names[1]} - loading actual diff and null extremes')  # noqa
        with open(f"Contrast/NullDistributions/{meta_names[0]}_x_{meta_names[1]}_{target_n}.pickle", 'rb') as f:
            r_diff, prior, min_diff, max_diff = pickle.load(f)
    else:
        print(f'{meta_names[0]} x {meta_names[1]} - computing actual diff and null extremes')  # noqa
        prior = np.zeros(BRAIN_ARRAY_SHAPE).astype(bool)
        prior[GM_PRIOR] = 1

        r_diff = Parallel(n_jobs=nprocesses)(
            delayed(compute_balanced_ale_diff)(ma1,
                                               ma2,
                                               prior,
                                               target_n) for i in range(difference_iterations))
        r_diff = np.mean(r_diff, axis=0)

        nfoci1 = exp_dfs[0].NumberOfFoci
        nfoci2 = exp_dfs[1].NumberOfFoci
        min_diff, max_diff = zip(*Parallel(n_jobs=nprocesses)(
            delayed(compute_balanced_null_diff)(nfoci1,
                                                kernels1,
                                                nfoci2,
                                                kernels2,
                                                prior,
                                                target_n,
                                                difference_iterations) for i in range(monte_carlo_iterations)))

        pickle_object = (r_diff, prior, min_diff, max_diff)
        with open(f"Contrast/NullDistributions/{meta_names[0]}_x_{meta_names[1]}_{target_n}.pickle", "wb") as f:
            pickle.dump(pickle_object, f)

    if not Path(f"Contrast/{meta_names[0]}_x_{meta_names[1]}_{target_n}_vFWE05.nii").exists():
        print(f'{meta_names[0]} x {meta_names[1]} - computing significant contrast')  # noqa

        # Calculate thresholds
        low_threshold = np.percentile(min_diff, 2.5)
        high_threshold = np.percentile(max_diff, 97.5)

        # Identify significant differences
        is_significant = np.logical_or(r_diff < low_threshold,
                                       r_diff > high_threshold)
        sig_diff = r_diff * is_significant

        # Calculate z-values for positive differences
        positive_diffs = sig_diff > 0
        sig_diff[positive_diffs] = [
            -1 * norm.ppf((np.sum(max_diff >= diff) + 1) /
                          (monte_carlo_iterations + 1))
            for diff in sig_diff[positive_diffs]
        ]

        # Calculate z-values for negative differences
        negative_diffs = sig_diff < 0
        sig_diff[negative_diffs] = [
            norm.ppf((np.sum(min_diff <= diff) + 1) /
                     (monte_carlo_iterations + 1))
            for diff in sig_diff[negative_diffs]
        ]

        # Create the final brain difference map
        brain_sig_diff = np.zeros(BRAIN_ARRAY_SHAPE)
        brain_sig_diff[prior] = sig_diff

        plot_and_save(brain_sig_diff,
                      nii_folder=f"Contrast/{meta_names[0]}_x_{meta_names[1]}_{target_n}_FWE05.nii")

    print(
        f'{meta_names[0]} x {meta_names[1]} balanced (n = {target_n}) contrast done!')
