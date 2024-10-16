from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nb
import pickle
from joblib import Parallel, delayed
from core.utils.kernel import create_kernel_array
from core.utils.cutoff_prediction import predict_cutoff
from core.utils.compute import (
    compute_ma, compute_ale, compute_z, compute_tfce, compute_clusters,
    compute_hx, compute_hx_conv, compute_monte_carlo_null,
    compute_sub_ale, generate_unique_subsamples, plot_and_save, illustrate_foci
)


def main_effect(project_path,
                exp_df,
                meta_name,
                tfce_enabled=True,
                cutoff_predict_enabled=True,
                bin_steps=0.0001,
                cluster_forming_threshold=0.001,
                monte_carlo_iterations=5000,
                target_n=None,
                sample_n=2500,
                nprocesses=2):

    # set main_effect results folder as path
    project_path = (Path(project_path) / "Results/MainEffect").resolve()

    # calculate smoothing kernels for each experiment
    kernels = create_kernel_array(exp_df)

    # calculate maximum possible ale value to set boundaries for histogram bins
    max_ma = np.prod([1 - np.max(kernel) for kernel in kernels])

    # define bins for histogram
    bin_edges = np.arange(0.00005, 1 - max_ma + 0.001, bin_steps)
    bin_centers = np.arange(0, 1 - max_ma + 0.001, bin_steps)
    step = int(1 / bin_steps)

    # Save included experiments for provenance tracking
    print_df = pd.DataFrame({
        "Experiment": exp_df.Articles.values,
        "Number of Foci": exp_df.NumberOfFoci.values
    })
    print_df.to_csv(project_path /
                    f"{meta_name}_included_experiments.csv", index=None, sep="\t")

    ma = compute_ma(exp_df.Coordinates.values, kernels)
    np.save(project_path / f'{meta_name}_ma', ma)

    if target_n:
        # subsampling or probabilistic ALE
        print(f"{meta_name} - entering probabilistic ALE routine.")
        # Check whether monte-carlo cutoff has been calculated before
        if Path(project_path /
                f"CV/NullDistributions/{meta_name}_montecarlo_{target_n}.pickle").exists():
            print(f"{meta_name} - loading cv cluster cut-off.")
            with open(project_path /
                      f"/CV/NullDistributions/{meta_name}_montecarlo_{target_n}.pickle", "rb") as f:
                cfwe_null = pickle.load(f)
        else:
            print(f"{meta_name} - computing cv cluster cut-off.")
            _, cfwe_null, _ = zip(
                *Parallel(
                    n_jobs=nprocesses,
                    verbose=2
                )(
                    delayed(compute_monte_carlo_null)(
                        num_foci=exp_df.NumberOfFoci,
                        kernels=kernels,
                        bin_edges=bin_edges,
                        bin_centers=bin_centers,
                        step=step,
                        cluster_forming_threshold=cluster_forming_threshold,
                        target_n=target_n,
                        tfce_enabled=False
                    ) for i in range(monte_carlo_iterations)
                )
            )

            subsampling_cfwe_threshold = np.percentile(cfwe_null, 95)
            with open(project_path /
                      f"CV/NullDistributions/{meta_name}_montecarlo_{target_n}.pickle", "wb") as f:
                pickle.dump(cfwe_null, f)

        print(f"{meta_name} - computing cv ale.")

        samples = generate_unique_subsamples(total_n=exp_df.shape[0],
                                             target_n=target_n,
                                             sample_n=sample_n
                                             )
        ale_mean = compute_sub_ale(samples,
                                   ma,
                                   subsampling_cfwe_threshold,
                                   bin_edges,
                                   bin_centers,
                                   step,
                                   cluster_forming_threshold)
        plot_and_save(ale_mean,
                      nii_path=project_path /
                      f"CV/Volumes/{meta_name}_sub_ale_{target_n}.nii")

        print(f"{meta_name} - probabilistic ALE done!")
        return

    else:
        # Full ALE
        # Foci illustration
        if not Path(project_path /
                    f"/Full/Volumes/Foci/{meta_name}.nii").exists():
            print(f"{meta_name} - illustrate Foci")
            # take all peaks of included studies and save them in a Nifti
            foci_arr = illustrate_foci(exp_df.Coordinates.values)
            plot_and_save(foci_arr,
                          nii_path=project_path /
                          f"Full/Volumes/{meta_name}_foci.nii")

        # ALE calculation
        if Path(project_path /
                f"Full/NullDistributions/{meta_name}.pickle").exists():
            print(f"{meta_name} - loading ALE")
            print(f"{meta_name} - loading null PDF")
            ale = nb.load(project_path /
                          f"/Full/Volumes/{meta_name}_ale.nii").get_fdata()
            with open(project_path /
                      f"/Full/NullDistributions/{meta_name}.pickle", "rb") as f:
                hx_conv, _ = pickle.load(f)

        else:
            print(f"{meta_name} - computing ALE and null PDF")
            ale = compute_ale(ma)
            plot_and_save(ale,
                          nii_path=project_path /
                          f"Full/Volumes/{meta_name}_ale.nii")

            # Calculate histogram and use it to estimate a null probability density function
            hx = compute_hx(ma, bin_edges)
            hx_conv = compute_hx_conv(hx, bin_centers, step)

            pickle_object = (hx_conv, hx)
            with open(project_path /
                      f"Full/NullDistributions/{meta_name}_histogram.pickle", "wb") as f:
                pickle.dump(pickle_object, f)

        # z- and tfce-map calculation
        if Path(project_path /
                f"Full/Volumes/{meta_name}_z.nii").exists():
            print(f"{meta_name} - loading z-values & TFCE")
            z = nb.load(project_path /
                        f"Full/Volumes/{meta_name}_z.nii").get_fdata()

        else:
            print(f"{meta_name} - computing p-values & TFCE")
            z = compute_z(ale, hx_conv, step)
            plot_and_save(z, nii_path=project_path /
                          f"Full/Volumes/{meta_name}_z.nii")
        if tfce_enabled is True:
            if Path(project_path /
                    f"Full/Volumes/{meta_name}_tfce_uncorrected.nii").exists():
                tfce = nb.load(project_path /
                               f"Full/Volumes/{meta_name}_tfce_uncorrected.nii").get_fdata()
            else:
                tfce = compute_tfce(z)
                plot_and_save(tfce,
                              nii_path=project_path /
                              f"Full/Volumes/{meta_name}_tfce_uncorrected.nii")

        # monte-carlo simulation for multiple comparison corrected thresholds
        if cutoff_predict_enabled:
            # using ml models to predict thresholds
            vfwe_treshold, cfwe_threshold, tfce_threshold = predict_cutoff(
                exp_df=exp_df)
        else:
            if Path(project_path /
                    f"Full/NullDistributions/{meta_name}_montecarlo.pickle").exists():
                print(f"{meta_name} - loading null")
                with open(project_path /
                          f"Full/NullDistributions/{meta_name}_montecarlo.pickle", "rb") as f:
                    vfwe_null, cfwe_null, tfce_null = pickle.load(f)
            else:
                print(f"{meta_name} - simulating null")
                vfwe_null, cfwe_null, tfce_null = zip(
                    *Parallel(
                        n_jobs=nprocesses,
                        verbose=2
                    )(
                        delayed(compute_monte_carlo_null)(
                            num_foci=exp_df.NumberOfFoci,
                            kernels=kernels,
                            bin_edges=bin_edges,
                            bin_centers=bin_centers,
                            step=step,
                            cluster_forming_threshold=cluster_forming_threshold,
                            tfce_enabled=tfce_enabled
                        ) for i in range(monte_carlo_iterations)
                    )
                )
                simulation_pickle = (vfwe_null, cfwe_null, tfce_null)
                with open(project_path /
                          f"Full/NullDistributions/{meta_name}_montecarlo.pickle", "wb") as f:
                    pickle.dump(simulation_pickle, f)

            vfwe_treshold = np.percentile(vfwe_null, 95)
            cfwe_threshold = np.percentile(cfwe_null, 95)
            tfce_threshold = np.percentile(tfce_null, 95)

        # Tresholding maps with vFWE, cFWE, TFCE thresholds
        if not Path(project_path /
                    f"Full/Volumes/{meta_name}_vfwe.nii").exists():
            print(f"{meta_name} - inference and printing")
            # voxel wise family wise error correction
            vfwe_map = ale*(ale > vfwe_treshold)
            plot_and_save(vfwe_map,
                          nii_path=project_path /
                          f"Full/Volumes/{meta_name}_vFWE05.nii")
            print(
                f"Min p-value for FWE: {sum(vfwe_null > np.max(ale)) / len(vfwe_null)}"
            )

            # cluster wise family wise error correction
            cfwe_map, max_clust = compute_clusters(z,
                                                   cluster_forming_threshold,
                                                   cfwe_threshold)
            plot_and_save(cfwe_map,
                          nii_path=project_path /
                          f"Full/Volumes/{meta_name}_cFWE05.nii")
            print(
                f"Min p-value for cFWE:{sum(cfwe_null>max_clust)/len(cfwe_null)}")

            # tfce error correction
            if tfce_enabled:
                tfce_map = tfce*(tfce > tfce_threshold)
                plot_and_save(tfce_map,
                              nii_path=project_path /
                              f"Full/Volumes/{meta_name}_TFCE05.nii")
                print(
                    f"Min p-value for TFCE:{sum(tfce_null>np.max(tfce))/len(tfce_null)}")

        else:
            pass
            print(f"{meta_name} - done!")
