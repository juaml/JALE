import logging
import pickle
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from jale.core.utils.compute import (
    compute_ma,
    compute_monte_carlo_null,
    compute_sub_ale,
    generate_unique_subsamples,
)
from jale.core.utils.folder_setup import folder_setup
from jale.core.utils.kernel import create_kernel_array
from jale.core.utils.plot_and_save import plot_and_save

logger = logging.getLogger("ale_logger")


def probabilistic_ale(
    project_path,
    exp_df,
    meta_name,
    bin_steps=0.0001,
    cluster_forming_threshold=0.001,
    monte_carlo_iterations=5000,
    target_n=None,
    sample_n=2500,
    nprocesses=2,
):
    """
    Compute and save probabilistic ALE maps for a given meta-analysis.

    Parameters
    ----------
    project_path : str or Path
        Path to the project directory containing the "Results" folder.
    exp_df : pandas.DataFrame
        DataFrame containing experiment data, including coordinates and number of foci.
    meta_name : str
        Name of the meta-analysis, used for naming saved files.
    tfce_enabled : bool, optional
        Whether to compute TFCE-corrected maps, by default True.
    cutoff_predict_enabled : bool, optional
        If True, predicts statistical thresholds using ML models, by default True.
    bin_steps : float, optional
        Step size for defining histogram bins, by default 0.0001.
    cluster_forming_threshold : float, optional
        Threshold for forming clusters in ALE, by default 0.001.
    monte_carlo_iterations : int, optional
        Number of Monte Carlo iterations for null distribution simulation, by default 5000.
    target_n : int, optional
        Target number of subsamples for probabilistic ALE, by default None (uses full sample).
    sample_n : int, optional
        Number of subsamples to generate if `target_n` is specified, by default 2500.
    nprocesses : int, optional
        Number of parallel processes for computations, by default 2.

    Returns
    -------
    None
        The function performs computations and saves the results as NIfTI files in the
        specified `project_path` directory.
    """
    logger.info(
        f"{meta_name} : {exp_df.shape[0]} experiments; average of {exp_df.Subjects.mean():.2f} subjects per experiment"
    )

    folder_setup(project_path, "Probabilistic")
    # set cv results folder as path
    project_path = (Path(project_path) / "Results/Probabilistic/").resolve()

    # calculate smoothing kernels for each experiment
    kernels = create_kernel_array(exp_df)
    np.save(project_path / f"{meta_name}_kernels", kernels)

    # calculate maximum possible ale value to set boundaries for histogram bins
    max_ma = np.prod([1 - np.max(kernel) for kernel in kernels])

    # define bins for histogram
    bin_edges = np.arange(0.00005, 1 - max_ma + 0.001, bin_steps)
    bin_centers = np.arange(0, 1 - max_ma + 0.001, bin_steps)
    step = int(1 / bin_steps)

    # Save included experiments for provenance tracking
    print_df = pd.DataFrame(
        {
            "Experiment": exp_df.Articles.values,
            "Number of Foci": exp_df.NumberOfFoci.values,
        }
    )
    print_df.to_csv(
        project_path / f"{meta_name}_included_experiments.csv", index=False, sep="\t"
    )

    # MA calculation
    ma = compute_ma(exp_df.Coordinates.values, kernels)
    np.savez_compressed(project_path / f"{meta_name}_ma", ma)

    # subsampling or probabilistic ALE
    logger.info(f"{meta_name} - entering probabilistic ALE routine.")
    # Check whether monte-carlo cutoff has been calculated before
    if Path(
        project_path / f"NullDistributions/{meta_name}_montecarlo_{target_n}.pickle"
    ).exists():
        logger.info(f"{meta_name} - loading cv cluster cut-off.")
        with open(
            project_path
            / f"NullDistributions/{meta_name}_montecarlo_{target_n}.pickle",
            "rb",
        ) as f:
            cfwe_null = pickle.load(f)
            subsampling_cfwe_threshold = np.percentile(cfwe_null, 95)
    else:
        logger.info(f"{meta_name} - computing cv cluster cut-off.")
        _, cfwe_null, _ = zip(
            *Parallel(n_jobs=nprocesses, verbose=2)(
                delayed(compute_monte_carlo_null)(
                    num_foci=exp_df.NumberOfFoci,
                    kernels=kernels,
                    bin_edges=bin_edges,
                    bin_centers=bin_centers,
                    step=step,
                    cluster_forming_threshold=cluster_forming_threshold,
                    target_n=target_n,
                    tfce_enabled=False,
                )
                for i in range(monte_carlo_iterations)
            )
        )

        subsampling_cfwe_threshold = np.percentile(cfwe_null, 95)
        with open(
            project_path
            / f"NullDistributions/{meta_name}_montecarlo_{target_n}.pickle",
            "wb",
        ) as f:
            pickle.dump(cfwe_null, f)
    if Path(project_path / f"Volumes/{meta_name}_sub_ale_{target_n}.nii").exists():
        logger.info(f"{meta_name} - loading cv ale")

        ale_mean = nb.load(
            project_path / f"Volumes/{meta_name}_sub_ale_{target_n}.nii"
        ).get_fdata()

    else:
        logger.info(f"{meta_name} - computing cv ale.")

        samples = generate_unique_subsamples(
            total_n=exp_df.shape[0], target_n=target_n, sample_n=sample_n
        )
        ale_mean = compute_sub_ale(
            samples,
            ma,
            subsampling_cfwe_threshold,
            bin_edges,
            bin_centers,
            step,
            cluster_forming_threshold,
        )
        plot_and_save(project_path, f"{meta_name}_sub_ale_{target_n}", ale_mean)

        logger.info(f"{meta_name} - probabilistic ALE done!")
