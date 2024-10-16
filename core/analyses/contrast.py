from pathlib import Path
import nibabel as nb
import numpy as np
from joblib import Parallel, delayed
from core.utils.compute import compute_ale, compute_null_diff, compute_sig_diff, plot_and_save
from core.utils.template import BRAIN_ARRAY_SHAPE


def contrast(project_path, meta_names, significance_threshold=0.05, null_repeats=10000, nprocesses=4):

    ma_group1 = np.load(project_path /
                        f"Results/MainEffect/{meta_names[0]}_ma.npy")
    ale_group1 = compute_ale(ma_group1)
    n_exp_group1 = ma_group1.shape[0]

    ma_group2 = np.load(project_path /
                        f"Results/MainEffect/{meta_names[1]}_ma.npy")
    ale_group2 = compute_ale(ma_group1)
    n_exp_group2 = ma_group2.shape[0]

    # Check if contrast has already been calculated
    if Path(project_path /
            f"Results/Contrast/Full/{meta_names[0]}_vs_{meta_names[1]}.nii").exists():
        print(f"{meta_names[0]} x {meta_names[1]} - Loading contrast.")
        contrast_arr = nb.load(
            project_path /
            f"Results/Contrast/Full/{meta_names[0]}_vs_{meta_names[1]}.nii").get_fdata()
    else:
        print(f"{meta_names[0]} x {meta_names[1]} - Computing positive contrast.")  # noqa
        group1_main_effect = nb.load(
            project_path /
            f"Results/MainEffect/Full/Volumes/{meta_names[0]}_cFWE05.nii").get_fdata()
        mask = group1_main_effect > 0
        if mask.sum() > 0:
            stacked_masked_ma = np.vstack(
                (ma_group1[:, mask], ma_group2[:, mask])
            )

            ale_difference1 = ale_group1 - ale_group2
            # estimate null distribution of difference values if studies
            # would be randomly assigned to either meta analysis
            null_difference1 = Parallel(n_jobs=nprocesses)(delayed(compute_null_diff)(
                stacked_masked_ma, n_exp_group1) for i in range(null_repeats))
            z1, sig_idxs1 = compute_sig_diff(
                ale_difference1[mask], null_difference1, significance_threshold
            )

        else:
            print(f"{meta_names[0]}: No significant indices!")
            z1, sig_idxs1 = [], []

        print(
            f"{meta_names[1]} x {meta_names[0]} - Computing negative contrast.")
        group2_main_effect = nb.load(
            project_path /
            f"Results/MainEffect/Full/Volumes/{meta_names[1]}_cFWE05.nii").get_fdata()
        mask = group2_main_effect > 0
        if mask.sum() > 0:
            stacked_masked_ma = np.vstack(
                (ma_group1[:, mask], ma_group2[:, mask])
            )
            ale_difference2 = ale_group2 - ale_group1
            null_difference2 = Parallel(n_jobs=nprocesses)(delayed(compute_null_diff)(
                stacked_masked_ma, n_exp_group2) for i in range(null_repeats))
            z2, sig_idxs2 = compute_sig_diff(
                ale_difference2[mask], null_difference2, significance_threshold
            )

        else:
            print(f"{meta_names[1]}: No significant indices!")
            z2, sig_idxs2 = [], []

        print(f"{meta_names[0]} x {meta_names[1]} - Inference and printing.")
        contrast_arr = np.zeros(BRAIN_ARRAY_SHAPE)
        contrast_arr[tuple(sig_idxs1)] = z1
        contrast_arr[tuple(sig_idxs2)] = -z2
        contrast_arr = plot_and_save(
            contrast_arr,
            nii_folder=project_path /
            f"Results/Contrast/Full/{meta_names[0]}_vs_{meta_names[1]}_cFWE.nii")

    # Check if conjunction has already been calculated
    if Path(project_path /
            f"Results/Contrast/Full/{meta_names[0]}_AND_{meta_names[1]}_cFWE.nii").exists():
        print(f"{meta_names[0]} & {meta_names[1]} - Loading conjunction.")
        conj_arr = nb.load(
            project_path /
            f"Results/Contrast/Full/{meta_names[0]}_AND_{meta_names[1]}_cFWE.nii").get_fdata()
    else:
        print(f"{meta_names[0]} & {meta_names[1]} - Computing conjunction.")
        conj_arr = np.minimum(group1_main_effect, group2_main_effect)
        if conj_arr is not None:
            conj_arr = plot_and_save(
                conj_arr,
                nii_folder=project_path /
                f"Results/Contrast/Full/{meta_names[0]}_AND_{meta_names[1]}_cFWE.nii")

    print(f"{meta_names[0]} & {meta_names[1]} - done!")
