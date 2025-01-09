import logging
import pickle
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from jale.core.utils.compute import (
    compute_ale,
    compute_clusters,
    compute_hx,
    compute_hx_conv,
    compute_ma,
    compute_monte_carlo_null,
    compute_tfce,
    compute_z,
    illustrate_foci,
)
from jale.core.utils.cutoff_prediction import predict_cutoff
from jale.core.utils.folder_setup import folder_setup
from jale.core.utils.kernel import create_kernel_array
from jale.core.utils.plot_and_save import plot_and_save
from jale.core.utils.template import GM_PRIOR

logger = logging.getLogger("ale_logger")


def main_effect(
    project_path,
    exp_df,
    meta_name,
    tfce_enabled=True,
    cutoff_predict_enabled=True,
    gm_masking=True,
    bin_steps=0.0001,
    cluster_forming_threshold=0.001,
    monte_carlo_iterations=5000,
    nprocesses=2,
):
    """
    Compute and save the main effect map for a given meta-analysis.

    This function calculates the main effect ALE, going from a coordinate dataframe all the way to thresholded brain maps.
    Uses either monte-carlo simulation or machine learning prediction for multiple comparison correction (vFWE, cFWE or TFCE).

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
        f"{meta_name} : {exp_df.shape[0]} experiments; {exp_df.Subjects.sum()}"
        f" total subjects - average of {exp_df.Subjects.mean():.2f} subjects per experiment"
    )

    # set main_effect results folder as path
    folder_setup(project_path, "MainEffect")
    project_path = (Path(project_path) / "Results/MainEffect/").resolve()

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

    # Foci illustration
    if not Path(project_path / f"Volumes/{meta_name}_foci.nii").exists():
        logger.info(f"{meta_name} - illustrate Foci")
        # take all peaks of included studies and save them in a Nifti
        foci_arr = illustrate_foci(exp_df.Coordinates.values)
        plot_and_save(project_path, f"{meta_name}_foci", foci_arr)

    # ALE calculation
    if Path(project_path / f"NullDistributions/{meta_name}.pickle").exists():
        logger.info(f"{meta_name} - loading ALE")
        logger.info(f"{meta_name} - loading null PDF")
        ale = nb.loadsave.load(
            project_path / f"Volumes/{meta_name}_ale.nii"
        ).get_fdata()  # type: ignore
        with open(project_path / f"NullDistributions/{meta_name}.pickle", "rb") as f:
            hx_conv, _ = pickle.load(f)

    else:
        logger.info(f"{meta_name} - computing ALE and null PDF")
        ale = compute_ale(ma)
        if gm_masking is True:
            ale[GM_PRIOR == 0] = 0
        plot_and_save(project_path, f"{meta_name}_ale", ale)

        # Calculate histogram and use it to estimate a null probability density function
        hx = compute_hx(ma, bin_edges)
        hx_conv = compute_hx_conv(hx, bin_centers, step)

        pickle_object = (hx_conv, hx)
        with open(
            project_path / f"NullDistributions/{meta_name}_histogram.pickle",
            "wb",
        ) as f:
            pickle.dump(pickle_object, f)

    # z- and tfce-map calculation
    if Path(project_path / f"Volumes/{meta_name}_z.nii").exists():
        logger.info(f"{meta_name} - loading z-values & TFCE")
        z = nb.loadsave.load(project_path / f"Volumes/{meta_name}_z.nii").get_fdata()  # type: ignore

    else:
        logger.info(f"{meta_name} - computing p-values & TFCE")
        z = compute_z(ale, hx_conv, step)
        plot_and_save(project_path, f"{meta_name}_z", z)
    if tfce_enabled is True:
        if Path(project_path / f"Volumes/{meta_name}_tfce_uncorrected.nii").exists():
            tfce = nb.loadsave.load(
                project_path / f"Volumes/{meta_name}_tfce_uncorrected.nii"
            ).get_fdata()  # type: ignore
        else:
            tfce = compute_tfce(z)
            plot_and_save(project_path, f"{meta_name}_tfce_uncorrected", tfce)

    # monte-carlo simulation for multiple comparison corrected thresholds
    if cutoff_predict_enabled:
        # using ml models to predict thresholds
        vfwe_treshold, cfwe_threshold, tfce_threshold = predict_cutoff(exp_df=exp_df)
    else:
        if Path(
            project_path / f"NullDistributions/{meta_name}_montecarlo.pickle"
        ).exists():
            logger.info(f"{meta_name} - loading null")
            with open(
                project_path / f"NullDistributions/{meta_name}_montecarlo.pickle",
                "rb",
            ) as f:
                vfwe_null, cfwe_null, tfce_null = pickle.load(f)
        else:
            logger.info(f"{meta_name} - simulating null")
            vfwe_null, cfwe_null, tfce_null = zip(
                *Parallel(n_jobs=nprocesses, verbose=2)(
                    delayed(compute_monte_carlo_null)(
                        num_foci=exp_df.NumberOfFoci,
                        kernels=kernels,
                        bin_edges=bin_edges,
                        bin_centers=bin_centers,
                        step=step,
                        cluster_forming_threshold=cluster_forming_threshold,
                        tfce_enabled=tfce_enabled,
                    )
                    for i in range(monte_carlo_iterations)
                )
            )

            simulation_pickle = (vfwe_null, cfwe_null, tfce_null)
            with open(
                project_path / f"NullDistributions/{meta_name}_montecarlo.pickle",
                "wb",
            ) as f:
                pickle.dump(simulation_pickle, f)

        vfwe_treshold = np.percentile(vfwe_null, 95)
        cfwe_threshold = np.percentile(cfwe_null, 95)
        tfce_threshold = np.percentile(tfce_null, 95)

    # Tresholding maps with vFWE, cFWE, TFCE thresholds
    if not Path(project_path / f"Volumes/{meta_name}_vFWE.nii").exists():
        logger.info(f"{meta_name} - inference and printing")
        # voxel wise family wise error correction
        vfwe_map = ale * (ale > vfwe_treshold)
        plot_and_save(project_path, f"{meta_name}_vFWE", vfwe_map)
        if np.max(ale) > vfwe_treshold:
            logger.info("vFWE: significant effect found.")
        else:
            logger.info("vFWE: no significant effect found.")

        # cluster wise family wise error correction
        cfwe_map, max_clust = compute_clusters(
            z, cluster_forming_threshold, cfwe_threshold
        )
        plot_and_save(project_path, f"{meta_name}_cFWE", cfwe_map)
        if max_clust > cfwe_threshold:
            logger.info("cFWE: significant effect found.")
        else:
            logger.info("cFWE: no significant effect found.")

        # tfce error correction
        if tfce_enabled:
            tfce_map = tfce * (tfce > tfce_threshold)
            plot_and_save(project_path, f"{meta_name}_tfce", tfce_map)
            if np.max(tfce) > tfce_threshold:
                logger.info("TFCE: significant effect found.")
            else:
                logger.info("TFCE: no significant effect found.")

    else:
        pass
        logger.info(f"{meta_name} - done!")
