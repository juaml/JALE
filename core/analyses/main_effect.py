from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nb
import pickle
from joblib import Parallel, delayed
from core.utils.cutoff_prediction import predict_cutoff
from core.utils.kernel import kernel_calc
from core.utils.compute import (
    compute_ma, compute_ale, compute_z, compute_tfce, compute_clusters,
    compute_hx, compute_hx_conv, compute_null_cutoffs,
    generate_unique_subsamples, compute_sub_ale,
    plot_and_save, illustrate_foci
)


def main_effect(project_path,
                exp_df,
                exp_name,
                tfce_enabled=True,
                cutoff_predict_enabled=True,
                bin_steps=0.0001,
                cluster_thresh=0.001,
                null_repeats=5000,
                target_n=None,
                sample_n=None,
                nprocesses=2):

    # set main_effect results folder as path
    project_path = project_path / "Results/MainEffect/"
    project_path = Path(project_path).resolve()

    # calculate smoothing kernels for each experiment
    kernels = np.empty((exp_df.shape[0], 31, 31, 31))
    template_uncertainty = 5.7/(2*np.sqrt(2/np.pi)) * np.sqrt(8*np.log(2))
    for i, n_subjects in enumerate(exp_df.Subjects.values):
        subj_uncertainty = (11.6/(2*np.sqrt(2/np.pi)) *
                            np.sqrt(8*np.log(2))) / np.sqrt(n_subjects)
        smoothing = np.sqrt(template_uncertainty**2 + subj_uncertainty**2)
        kernels[i, :, :] = kernel_calc(smoothing, 31)

    # calculate maximum possible ale value to set boundaries for histogram bins
    max_ma = 1
    for kernel in kernels:
        max_ma = max_ma*(1-np.max(kernel))

    # define bins for histogram
    bin_edges = np.arange(0.00005, 1-max_ma+0.001, bin_steps)
    bin_centers = np.arange(0, 1-max_ma+0.001, bin_steps)
    step = int(1/bin_steps)

    # save included experiments for provenance tracking
    print_df = pd.DataFrame(
        [exp_df.Author.values, exp_df.NumberofFoci.values]).transpose()
    print_df.columns = ["Experiment", "Number of Foci"]
    print_df.to_csv(
        project_path / f"{exp_name}_included_experiments.csv",
        index=None,
        mode="w",
        sep="\t"
    )

    ma = compute_ma(exp_df.Coordinates.values, kernels)
    hx = compute_hx(ma, bin_edges)

    if target_n:
        # Probabilistic or cross-validated ALE

        print(f"{exp_name} - entering probabilistic ALE routine.")

        # Check whether monte-carlo cutoff has been calculated before
        if Path(project_path /
                f"CV/NullDistributions/{exp_name}_ccut_{target_n}.pickle").exists():
            print(f"{exp_name} - loading cv cluster cut-off.")
            with open(project_path /
                      f"/CV/NullDistributions/{exp_name}_ccut_{target_n}.pickle", "rb") as f:
                cut_cluster = pickle.load(f)
        else:
            print(f"{exp_name} - computing cv cluster cut-off.")
            vfwe_null, cfwe_null, _ = zip(
                *Parallel(
                    n_jobs=nprocesses,
                    verbose=2
                )(
                    delayed(compute_null_cutoffs)(
                        num_foci=exp_df.Coordinates,
                        kernels=exp_df.Kernels,
                        step=step,
                        cluster_thresh=cluster_thresh,
                        bin_centers=bin_centers,
                        bin_edges=bin_edges,
                        target_n=target_n,
                        tfce_enabled=False
                    ) for i in range(null_repeats)
                )
            )

            cut_cluster = np.percentile(cfwe_null, 95)
            with open(project_path /
                      f"CV/NullDistributions/{exp_name}_ccut_{target_n}.pickle", "wb") as f:
                pickle.dump(cut_cluster, f)

        print(f"{exp_name} - computing cv ale.")
        samples = generate_unique_subsamples(total_n=exp_df.shape[0],
                                             target_n=target_n,
                                             sample_n=sample_n)
        ale_mean = np.zeros((91, 109, 91))
        for idx, sample in enumerate(samples):
            if (idx % 500) == 0:
                print(f"Calculated {idx} subsample ALEs")
            ale_mean += compute_sub_ale(sample,
                                        ma,
                                        hx,
                                        bin_centers,
                                        cut_cluster,
                                        thresh=cluster_thresh)
        ale_mean = ale_mean / len(samples)
        plot_and_save(ale_mean,
                      nii_path=project_path /
                      "CV/Volumes/{exp_name}_sub_ale_{target_n}.nii")

        print(f"{exp_name} - probabilistic ALE done!")
        return

    else:
        # Full ALE

        # Foci illustration
        if not Path(project_path /
                    f"/Full/Volumes/Foci/{exp_name}.nii").exists():
            print(f"{exp_name} - illustrate Foci")
            # take all peaks of included studies and save them in a Nifti
            foci_arr = illustrate_foci(exp_df.Coordinates.values)
            plot_and_save(foci_arr,
                          nii_path=project_path /
                          f"Full/Volumes/Foci/{exp_name}.nii")

        # ALE calculation
        if Path(project_path /
                f"Full/NullDistributions/{exp_name}.pickle").exists():
            print(f"{exp_name} - loading ALE")
            print(f"{exp_name} - loading null PDF")
            ale = nb.load(project_path /
                          f"/Full/Volumes/{exp_name}_ale.nii").get_fdata()
            with open(project_path /
                      f"/Full/NullDistributions/{exp_name}.pickle", "rb") as f:
                hx_conv, _ = pickle.load(f)

        else:
            print(f"{exp_name} - computing ALE and null PDF")
            ale = compute_ale(ma)
            plot_and_save(ale,
                          nii_path=project_path /
                          f"/Full/Volumes/{exp_name}_ale.nii")

            # Use the histograms from above to estimate a null probability density function
            hx_conv = compute_hx_conv(hx, bin_centers, step)

            pickle_object = (hx_conv, hx)
            with open(project_path /
                      f"Full/NullDistributions/{exp_name}_histogram.pickle", "wb") as f:
                pickle.dump(pickle_object, f)

        # z- and tfce-map calculation
        if Path(project_path /
                f"/Full/Volumes/{exp_name}_z.nii"):
            print(f"{exp_name} - loading p-values & TFCE")
            z = nb.load(project_path /
                        f"/Full/Volumes/{exp_name}_z.nii").get_fdata()

        else:
            print(f"{exp_name} - computing p-values & TFCE")
            z = compute_z(ale, hx_conv, step)
            plot_and_save(z, nii_path=project_path /
                          f"Full/Volumes/{exp_name}_z.nii")
        if tfce_enabled is True:
            if Path(project_path /
                    f"/Full/Volumes/{exp_name}_tfce.nii"):
                tfce = nb.load(project_path /
                               f"/Full/Volumes/{exp_name}_tfce.nii").get_fdata()
            else:
                tfce = compute_tfce(z)
                plot_and_save(tfce,
                              nii_path=project_path /
                              f"Full/Volumes/{exp_name}_tfce.nii")

        # monte-carlo simulation for multiple comparison corrected thresholds
        if cutoff_predict_enabled:
            # using ml models to predict thresholds
            vfwe_treshold, cfwe_threshold, tfce_threshold = predict_cutoff(
                exp_df=exp_df)
        else:
            if Path(project_path /
                    f"Full/NullDistributions/{exp_name}_montecarlo.pickle"):
                print(f"{exp_name} - loading null")
                with open(project_path /
                          f"Full/NullDistributions/{exp_name}_montecarlo.pickle", "rb") as f:
                    vfwe_null, cfwe_null, tfce_null = pickle.load(f)
            else:
                print(f"{exp_name} - simulating null")
                vfwe_null, cfwe_null, tfce_null = zip(
                    *Parallel(
                        n_jobs=nprocesses,
                        verbose=2
                    )(
                        delayed(compute_null_cutoffs)(
                            num_foci=exp_df.Coordinates,
                            kernels=exp_df.Kernels,
                            step=step,
                            cluster_thresh=cluster_thresh,
                            bin_centers=bin_centers,
                            bin_edges=bin_edges,
                            tfce_enabled=tfce_enabled
                        ) for i in range(null_repeats)
                    )
                )
                simulation_pickle = (vfwe_null, cfwe_null, tfce_null)
                with open(project_path /
                          f"Full/NullDistributions/{exp_name}_montecarlo.pickle", "wb") as f:
                    pickle.dump(simulation_pickle, f)

                vfwe_treshold = np.percentile(vfwe_null, 95)
                cfwe_threshold = np.percentile(cfwe_null, 95)
                tfce_threshold = np.percentile(tfce_null, 95)

        # Tresholding maps with vFWE, cFWE, TFCE
        if not Path(project_path /
                    f"/Full/Volumes/{exp_name}_vfwe.nii"):
            print(f"{exp_name} - inference and printing")
            # voxel wise family wise error correction
            vfwe_map = ale*(ale > vfwe_treshold)
            plot_and_save(vfwe_map,
                          nii_path=project_path /
                          f"Full/Volumes/{exp_name}_vFWE05.nii")
            print(
                f"Min p-value for FWE: {sum(vfwe_null > np.max(ale)) / len(vfwe_null)}"
            )

            # cluster wise family wise error correction
            cfwe_map, max_clust = compute_clusters(z,
                                                   cluster_thresh=cluster_thresh,
                                                   cut_cluster=cfwe_threshold)
            plot_and_save(cfwe_map,
                          nii_path=project_path /
                          f"Full/Volumes/Corrected/{exp_name}_cFWE05.nii")
            print(
                f"Min p-value for cFWE:{sum(cfwe_null>max_clust)/len(cfwe_null)}")

            # tfce error correction
            if tfce_enabled:
                tfce_map = tfce*(tfce > tfce_threshold)
                plot_and_save(tfce_map,
                              nii_path=project_path /
                              f"Full/Volumes/Corrected/{exp_name}_TFCE05.nii")
                print(
                    f"Min p-value for TFCE:{sum(tfce_null>np.max(tfce))/len(tfce_null)}")

        else:
            pass

        print(f"{exp_name} - done!")
