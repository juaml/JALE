import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy, spearmanr
from sklearn.metrics import calinski_harabasz_score, mutual_info_score, silhouette_score
from sklearn_extra.cluster import KMedoids

from jale.core.utils.compute import compute_ma, generate_unique_subsamples
from jale.core.utils.template import GM_PRIOR

logger = logging.getLogger("ale_logger")


def kmedoids_clustering_pipeline(
    project_path,
    exp_df,
    meta_name,
    kernels,
    correlation_type,
    linkage_method,
    correlation_matrix,
    max_clusters,
    subsample_fraction,
    sampling_iterations,
    null_iterations,
    use_pooled_std,
):
    logger = logging.getLogger("ale_logger")
    logger.info(f"{meta_name} - starting subsampling")
    (
        silhouette_scores,
        calinski_harabasz_scores,
        variation_of_information,
        cluster_labels,
    ) = compute_kmedoids_subsampling(
        correlation_matrix=correlation_matrix,
        max_clusters=max_clusters,
        subsample_fraction=subsample_fraction,
        sampling_iterations=sampling_iterations,
    )
    logger.info(f"{meta_name} - starting null calculation")
    (
        null_silhouette_scores,
        null_calinski_harabasz_scores,
    ) = compute_kmedoids_null(
        exp_df=exp_df,
        kernels=kernels,
        correlation_type=correlation_type,
        max_clusters=max_clusters,
        null_iterations=null_iterations,
        subsample_fraction=subsample_fraction,
    )
    silhouette_scores_z, calinski_harabasz_scores_z = compute_kmedoids_metrics_z(
        silhouette_scores=silhouette_scores,
        calinski_harabasz_scores=calinski_harabasz_scores,
        null_silhouette_scores=null_silhouette_scores,
        null_calinski_harabasz_scores=null_calinski_harabasz_scores,
        use_pooled_std=use_pooled_std,
    )

    logger.info(f"{meta_name} - calculating final cluster labels")
    hamming_distance_cluster_labels = compute_hamming_distance_kmedoids(
        cluster_labels=cluster_labels,
        max_clusters=max_clusters,
    )
    logger.info(f"{meta_name} - creating output and saving")
    save_kmedoids_labels(
        project_path=project_path,
        meta_name=meta_name,
        exp_df=exp_df,
        cluster_labels=hamming_distance_cluster_labels,
        correlation_type=correlation_type,
        max_clusters=max_clusters,
    )
    save_kmedoids_metrics(
        project_path=project_path,
        meta_name=meta_name,
        silhouette_scores=silhouette_scores,
        silhouette_scores_z=silhouette_scores_z,
        calinski_harabasz_scores=calinski_harabasz_scores,
        calinski_harabasz_scores_z=calinski_harabasz_scores_z,
        vi_scores=variation_of_information,
        correlation_type=correlation_type,
    )
    plot_kmedoids_metrics(
        project_path=project_path,
        meta_name=meta_name,
        silhouette_scores=silhouette_scores,
        null_silhouette_scores=null_silhouette_scores,
        silhouette_scores_z=silhouette_scores_z,
        calinski_harabasz_scores=calinski_harabasz_scores,
        null_calinski_harabasz_scores=null_calinski_harabasz_scores,
        calinski_harabasz_scores_z=calinski_harabasz_scores_z,
        vi_scores=variation_of_information,
        correlation_type=correlation_type,
        max_clusters=max_clusters,
    )


def compute_kmedoids_subsampling(
    correlation_matrix, max_clusters, subsample_fraction, sampling_iterations
):
    correlation_distance = 1 - correlation_matrix
    np.fill_diagonal(correlation_distance, 0)

    silhouette_scores = np.empty((max_clusters - 1, sampling_iterations))
    calinski_harabasz_scores = np.empty((max_clusters - 1, sampling_iterations))
    variation_of_information = np.empty((max_clusters - 2, sampling_iterations))
    cluster_labels = np.full(
        (max_clusters - 1, correlation_matrix.shape[0], sampling_iterations), np.nan
    )

    subsamples = generate_unique_subsamples(
        total_n=correlation_matrix.shape[0],
        target_n=int(subsample_fraction * correlation_matrix.shape[0]),
        sample_n=sampling_iterations,
    )
    # Iterate over different values of k, compute cluster metrics
    for i in range(sampling_iterations):
        for k in range(2, max_clusters + 1):
            # Resample indices for subsampling
            resampled_indices = subsamples[i]
            resampled_correlation = correlation_matrix[
                np.ix_(resampled_indices, resampled_indices)
            ]

            (
                silhouette_score,
                calinski_harabasz_score,
                cluster_label,
            ) = compute_kmedoids_clustering(
                correlation_matrix=resampled_correlation,
                k=k,
            )
            silhouette_scores[k - 2, i] = silhouette_score
            calinski_harabasz_scores[k - 2, i] = calinski_harabasz_score
            cluster_labels[k - 2, resampled_indices, i] = cluster_label.astype(np.int64)

            if k >= 3:
                variation_of_information[k - 3, i] = compute_variation_of_information(
                    cluster_labels_prev=cluster_labels[
                        k - 3, resampled_indices, i
                    ].astype(int),
                    cluster_labels_current=cluster_labels[
                        k - 2, resampled_indices, i
                    ].astype(int),
                )

    return (
        silhouette_scores,
        calinski_harabasz_scores,
        variation_of_information,
        cluster_labels,
    )


def compute_kmedoids_clustering(correlation_matrix, k):
    """
    Perform k-medoids clustering on the given correlation matrix and compute clustering metrics.

    Parameters:
    correlation_matrix (ndarray): A symmetric matrix representing the correlation between data points.
    k (int): The number of clusters to form.

    Returns:
    tuple: A tuple containing the silhouette score, Calinski-Harabasz index, and cluster labels.
    """

    distance_matrix = 1 - correlation_matrix
    np.fill_diagonal(distance_matrix, 0)

    kmedoids = KMedoids(
        n_clusters=k,
        metric="precomputed",
        method="pam",
        random_state=0,
    )
    cluster_labels = kmedoids.fit_predict(distance_matrix)

    silhouette = silhouette_score(distance_matrix, cluster_labels, metric="precomputed")
    calinski_harabasz = calinski_harabasz_score(1 - distance_matrix, cluster_labels)

    return silhouette, calinski_harabasz, cluster_labels


def compute_variation_of_information(cluster_labels_prev, cluster_labels_current):
    """
    Computes the variation of information (VI) between two clustering solutions at k-1 and k.
    """
    mutual_info = mutual_info_score(cluster_labels_prev, cluster_labels_current)
    entropy_k = entropy(np.bincount(cluster_labels_prev))
    entropy_k_next = entropy(np.bincount(cluster_labels_current))

    return entropy_k + entropy_k_next - 2 * mutual_info


def compute_kmedoids_null(
    exp_df,
    kernels,
    correlation_type,
    max_clusters,
    null_iterations,
    subsample_fraction,
):
    null_silhouette_scores = np.empty((max_clusters - 1, null_iterations))
    null_calinski_harabasz_scores = np.empty((max_clusters - 1, null_iterations))

    subsamples = generate_unique_subsamples(
        total_n=exp_df.shape[0],
        target_n=int(subsample_fraction * exp_df.shape[0]),
        sample_n=null_iterations,
    )
    for n in range(null_iterations):
        # Create an index array for subsampling

        # Subsample exp_df and kernels using the sampled indices
        sampled_exp_df = exp_df.iloc[subsamples[n]].reset_index(drop=True)
        sampled_kernels = [kernels[idx] for idx in subsamples[n]]

        coords_stacked = np.vstack(sampled_exp_df.Coordinates.values)
        shuffled_coords = []

        for exp in range(len(sampled_exp_df)):
            K = sampled_exp_df.iloc[exp]["NumberOfFoci"]
            # Step 1: Randomly sample K unique row indices
            sample_indices = np.random.choice(
                coords_stacked.shape[0], size=K, replace=False
            )
            # Step 2: Extract the sampled rows using the sampled indices
            sampled_rows = coords_stacked[sample_indices]
            shuffled_coords.append(sampled_rows)
            # Step 3: Delete the sampled rows from the original array
            coords_stacked = np.delete(coords_stacked, sample_indices, axis=0)

        # Compute the meta-analysis result with subsampled kernels
        null_ma = compute_ma(shuffled_coords, sampled_kernels)
        ma_gm_masked = null_ma[:, GM_PRIOR]
        # Set entire row to np.nan only if all values in the row are zero
        ma_gm_masked_nan = ma_gm_masked.copy()
        row_is_zero = np.all(ma_gm_masked_nan == 0, axis=1)
        ma_gm_masked_nan[row_is_zero, :] = np.nan
        if correlation_type == "spearman":
            correlation_matrix, _ = spearmanr(ma_gm_masked_nan, axis=1)
        elif correlation_type == "pearson":
            correlation_matrix = np.corrcoef(ma_gm_masked_nan)
        correlation_matrix = np.nan_to_num(
            correlation_matrix, nan=0, posinf=0, neginf=0
        )
        np.fill_diagonal(correlation_matrix, 0)

        for k in range(2, max_clusters + 1):
            (
                silhouette_score,
                calinski_harabasz_score,
                cluster_label,
            ) = compute_kmedoids_clustering(
                correlation_matrix=correlation_matrix,
                k=k,
            )
            null_silhouette_scores[k - 2, n] = silhouette_score
            null_calinski_harabasz_scores[k - 2, n] = calinski_harabasz_score

    return (null_silhouette_scores, null_calinski_harabasz_scores)


def compute_kmedoids_metrics_z(
    silhouette_scores,
    calinski_harabasz_scores,
    null_silhouette_scores,
    null_calinski_harabasz_scores,
    use_pooled_std=False,
):
    def pooled_std(sample1, sample2):
        """Compute the pooled standard deviation of two samples."""
        n1, n2 = sample1.shape[1], sample2.shape[1]
        var1, var2 = np.var(sample1, axis=1, ddof=1), np.var(sample2, axis=1, ddof=1)
        return np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    silhouette_scores_avg = np.average(silhouette_scores, axis=1)
    null_silhouette_scores_avg = np.average(null_silhouette_scores, axis=1)

    if use_pooled_std:
        silhouette_std = pooled_std(silhouette_scores, null_silhouette_scores)
    else:
        silhouette_std = np.std(null_silhouette_scores, axis=1, ddof=1)

    silhouette_z = (silhouette_scores_avg - null_silhouette_scores_avg) / silhouette_std

    calinski_harabasz_scores_avg = np.average(calinski_harabasz_scores, axis=1)
    null_calinski_harabasz_scores_avg = np.average(
        null_calinski_harabasz_scores, axis=1
    )

    if use_pooled_std:
        calinski_harabasz_std = pooled_std(
            calinski_harabasz_scores, null_calinski_harabasz_scores
        )
    else:
        calinski_harabasz_std = np.std(null_calinski_harabasz_scores, axis=1, ddof=1)

    calinski_harabasz_z = (
        calinski_harabasz_scores_avg - null_calinski_harabasz_scores_avg
    ) / calinski_harabasz_std

    return silhouette_z, calinski_harabasz_z


def compute_hamming_distance_kmedoids(cluster_labels, max_clusters):
    hamming_distance_cluster_labels = np.empty(
        (max_clusters - 1, cluster_labels.shape[1])
    )

    for k in range(2, max_clusters + 1):
        hamming_distance = compute_hamming_with_nan(cluster_labels[k - 2])

        # Fit KMedoids on the precomputed distance matrix
        kmedoids = KMedoids(
            n_clusters=k, metric="precomputed", method="pam", random_state=0
        )
        labels = kmedoids.fit_predict(hamming_distance)

        hamming_distance_cluster_labels[k - 2] = labels

    return hamming_distance_cluster_labels


def compute_hamming_with_nan(cluster_labels):
    # Precompute valid masks
    valid_masks = ~np.isnan(cluster_labels)

    # Initialize matrix for results
    n = cluster_labels.shape[0]
    hamming_matrix = np.full((n, n), np.nan)

    # Iterate through pairs using broadcasting
    for i in range(n):
        valid_i = valid_masks[i]
        for j in range(i + 1, n):
            valid_j = valid_masks[j]
            valid_mask = valid_i & valid_j
            total_valid = np.sum(valid_mask)
            if total_valid > 0:
                mismatches = np.sum(
                    cluster_labels[i, valid_mask] != cluster_labels[j, valid_mask]
                )
                hamming_matrix[i, j] = mismatches / total_valid
                hamming_matrix[j, i] = hamming_matrix[i, j]
            else:
                print(i, j)

    np.fill_diagonal(hamming_matrix, 0)
    return hamming_matrix


def save_kmedoids_labels(
    project_path,
    exp_df,
    meta_name,
    cluster_labels,
    correlation_type,
    max_clusters,
):
    # Generate dynamic header from k=2 to k=max_clusters
    header = ["Experiment"] + [f"k={k}" for k in range(2, max_clusters + 1)]

    # Create DataFrame
    cluster_labels_df = pd.DataFrame(
        np.column_stack([exp_df.Articles.values, cluster_labels.T]), columns=header
    )

    # Save as CSV
    cluster_labels_df.to_csv(
        project_path
        / f"Results/MA_Clustering/labels/{meta_name}_cluster_labels_{correlation_type}_kmedoids.csv",
        index=False,
        header=header,
    )


def save_kmedoids_metrics(
    project_path,
    meta_name,
    silhouette_scores,
    silhouette_scores_z,
    calinski_harabasz_scores,
    calinski_harabasz_scores_z,
    vi_scores,
    correlation_type,
):
    max_k = len(silhouette_scores) + 1
    metrics_df = pd.DataFrame(
        {
            "Number of Clusters": range(2, max_k + 1),
            "Silhouette Scores": np.average(silhouette_scores, axis=1),
            "Silhouette Scores SD": np.std(silhouette_scores, axis=1),
            "Silhouette Scores Z": silhouette_scores_z,
            "Calinski-Harabasz Scores": np.average(calinski_harabasz_scores, axis=1),
            "Calinski-Harabasz Scores SD": np.std(calinski_harabasz_scores, axis=1),
            "Calinski-Harabasz Scores Z": calinski_harabasz_scores_z,
            "Variation of Information": np.concatenate(
                [[0], np.average(vi_scores, axis=1)]
            ),
        }
    )
    metrics_df.to_csv(
        project_path
        / f"Results/MA_Clustering/{meta_name}_clustering_metrics_{correlation_type}_kmedoids.csv",
        index=False,
    )

    pd.DataFrame(silhouette_scores.T).to_csv(
        project_path
        / f"Results/MA_Clustering/metrics/{meta_name}_silhouette_scores_{correlation_type}_kmedoids.csv",
        index=False,
        header=[f"k={k}" for k in range(2, max_k + 1)],
    )

    pd.DataFrame(calinski_harabasz_scores.T).to_csv(
        project_path
        / f"Results/MA_Clustering/metrics/{meta_name}_calinski_harabasz_scores_{correlation_type}_kmedoids.csv",
        index=False,
        header=[f"k={k}" for k in range(2, max_k + 1)],
    )

    pd.DataFrame(vi_scores.T).to_csv(
        project_path
        / f"Results/MA_Clustering/metrics/{meta_name}_vi_scores_{correlation_type}_kmedoids.csv",
        index=False,
        header=[f"k={k}" for k in range(3, max_k + 1)],
    )


def plot_kmedoids_metrics(
    project_path,
    meta_name,
    silhouette_scores,
    null_silhouette_scores,
    silhouette_scores_z,
    calinski_harabasz_scores,
    null_calinski_harabasz_scores,
    calinski_harabasz_scores_z,
    vi_scores,
    correlation_type,
    max_clusters,
):
    k_range = np.array(list(range(2, max_clusters + 1)))

    # Calculate average scores, handling NaNs
    avg_silhouette_scores = np.nanmean(silhouette_scores, axis=1)
    avg_calinski_harabasz_scores = np.nanmean(calinski_harabasz_scores, axis=1)

    # Set up Matplotlib figure and subplots
    # Increased figsize for better readability and spacing
    fig, axes = plt.subplots(
        4, 1, figsize=(10, 18), sharex=True
    )  # Share x-axis for better comparison

    # Apply a nicer Seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("viridis")  # A visually appealing color palette

    # --- Plot 1: Average Silhouette Scores ---
    ax = axes[0]
    ax.plot(
        k_range,
        avg_silhouette_scores,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        color=sns.color_palette("viridis")[0],
    )
    ax.set_title("Average Silhouette Scores", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Score", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    # Add minor ticks for better readability
    ax.minorticks_on()
    ax.grid(which="minor", linestyle=":", linewidth="0.5", color="gray", alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=10)

    # --- Plot 2: Silhouette Scores Z ---
    ax = axes[1]
    ax.plot(
        k_range,
        silhouette_scores_z,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        color=sns.color_palette("viridis")[1],
    )
    ax.set_title("Silhouette Scores Z-scores", fontsize=14, fontweight="bold")
    ax.set_ylabel("Z-Score", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    # Add a horizontal line for common significance threshold (e.g., Z=1.96 for p<0.05, two-tailed)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)  # Line at Z=0
    ax.axhline(
        1.96,
        color="darkgreen",
        linestyle=":",
        linewidth=1.5,
        label="Z=1.96 (p<0.05)",
        alpha=0.7,
    )
    ax.axhline(-1.96, color="darkgreen", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.legend(loc="best", fontsize=10)
    ax.minorticks_on()
    ax.grid(which="minor", linestyle=":", linewidth="0.5", color="gray", alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=10)

    # --- Plot 3: Average Calinski-Harabasz Scores ---
    ax = axes[2]
    ax.plot(
        k_range,
        avg_calinski_harabasz_scores,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        color=sns.color_palette("viridis")[2],
    )
    ax.set_title("Average Calinski-Harabasz Scores", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Score", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.minorticks_on()
    ax.grid(which="minor", linestyle=":", linewidth="0.5", color="gray", alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=10)

    # --- Plot 4: Calinski-Harabasz Scores Z ---
    ax = axes[3]
    ax.plot(
        k_range,
        calinski_harabasz_scores_z,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        color=sns.color_palette("viridis")[3],
    )
    ax.set_title("Calinski-Harabasz Scores Z-scores", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Z-Score", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    # Add a horizontal line for common significance threshold
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)  # Line at Z=0
    ax.axhline(
        1.96,
        color="darkgreen",
        linestyle=":",
        linewidth=1.5,
        label="Z=1.96 (p<0.05)",
        alpha=0.7,
    )
    ax.axhline(
        -1.96, color="darkgreen", linestyle=":", linewidth=1.5, alpha=0.7
    )  # Only if negative Z-scores are meaningful
    ax.legend(loc="best", fontsize=10)
    ax.minorticks_on()
    ax.grid(which="minor", linestyle=":", linewidth="0.5", color="gray", alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Set common x-axis ticks and limits
    # Ensure x-axis ticks are integers for number of clusters
    axes[-1].set_xticks(k_range)
    axes[-1].set_xlim(
        min(k_range) - 0.5, max(k_range) + 0.5
    )  # Add a small buffer around min/max k

    plt.suptitle(
        f"Clustering Metrics for {meta_name} ({correlation_type} correlation)",
        y=0.99,
        fontsize=16,
        fontweight="bold",
    )  # Overall title for the figure
    plt.tight_layout(
        rect=[0, 0.03, 1, 0.96]
    )  # Adjust layout to make space for suptitle
    plt.savefig(
        project_path
        / f"Results/MA_Clustering/{meta_name}_clustering_metrics_{correlation_type}_kmedoids.png"
    )
    plt.close()

    # --- Distribution Plots ---

    num_k = len(k_range)

    # --- Calculate Global X-axis Limits for Silhouette Scores ---
    all_silhouette_scores = np.concatenate(
        (silhouette_scores.flatten(), null_silhouette_scores.flatten())
    )
    all_silhouette_scores_cleaned = all_silhouette_scores[
        ~np.isnan(all_silhouette_scores)
    ]

    if all_silhouette_scores_cleaned.size > 0:
        min_sil_x = np.nanmin(all_silhouette_scores_cleaned)
        max_sil_x = np.nanmax(all_silhouette_scores_cleaned)
        # Add a small buffer to the limits
        sil_x_buffer = (max_sil_x - min_sil_x) * 0.05
        fixed_sil_xlim = [min_sil_x - sil_x_buffer, max_sil_x + sil_x_buffer]
    else:
        # Default limits if no valid data
        fixed_sil_xlim = [-0.1, 1.1]  # Silhouette scores are typically -1 to 1

    # --- Calculate Global X-axis Limits for Calinski-Harabasz Scores ---
    all_ch_scores = np.concatenate(
        (calinski_harabasz_scores.flatten(), null_calinski_harabasz_scores.flatten())
    )
    all_ch_scores_cleaned = all_ch_scores[~np.isnan(all_ch_scores)]

    if all_ch_scores_cleaned.size > 0:
        min_ch_x = np.nanmin(all_ch_scores_cleaned)
        max_ch_x = np.nanmax(all_ch_scores_cleaned)
        # Add a small buffer to the limits, ensure positive range for CH which is non-negative
        ch_x_buffer = (max_ch_x - min_ch_x) * 0.05
        fixed_ch_xlim = [max(0, min_ch_x - ch_x_buffer), max_ch_x + ch_x_buffer]
    else:
        # Default limits if no valid data
        fixed_ch_xlim = [
            0,
            1000,
        ]  # Calinski-Harabasz can vary widely, this is a placeholder

    # Determine figure size dynamically based on the number of k values
    fig_height = max(6, num_k * 3)  # Minimum height of 6, 3 inches per row
    fig, axes = plt.subplots(
        num_k, 2, figsize=(14, fig_height), squeeze=False
    )  # 2 columns for Silhouette/CH

    for i, k in enumerate(k_range):
        # --- Silhouette Score Plots ---
        ax_sil = axes[i, 0]

        # Filter out NaN values for plotting KDEs
        observed_sil = silhouette_scores[k - 2, :]
        null_sil = null_silhouette_scores[k - 2, :]

        observed_sil_cleaned = observed_sil[~np.isnan(observed_sil)]
        null_sil_cleaned = null_sil[~np.isnan(null_sil)]

        if observed_sil_cleaned.size > 1:  # KDE requires at least 2 data points
            sns.kdeplot(
                observed_sil_cleaned,
                fill=True,
                color="red",
                ax=ax_sil,
                label="Observed",
            )
        else:
            ax_sil.text(
                0.5,
                0.5,
                "Not enough observed data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax_sil.transAxes,
                color="gray",
            )

        if null_sil_cleaned.size > 1:
            sns.kdeplot(
                null_sil_cleaned, fill=True, color="skyblue", ax=ax_sil, label="Null"
            )
        else:
            ax_sil.text(
                0.5,
                0.3,
                "Not enough null data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax_sil.transAxes,
                color="gray",
            )

        ax_sil.set_title(f"k={k} Silhouette Scores")
        ax_sil.set_ylabel("Density")
        ax_sil.grid(True, linestyle="--", alpha=0.7)
        ax_sil.legend()
        ax_sil.set_xlim(fixed_sil_xlim)  # Apply fixed x-axis limits

        # --- Calinski-Harabasz Score Plots ---
        ax_ch = axes[i, 1]

        # Filter out NaN values for plotting KDEs
        observed_ch = calinski_harabasz_scores[k - 2, :]
        null_ch = null_calinski_harabasz_scores[k - 2, :]

        observed_ch_cleaned = observed_ch[~np.isnan(observed_ch)]
        null_ch_cleaned = null_ch[~np.isnan(null_ch)]

        if observed_ch_cleaned.size > 1:
            sns.kdeplot(
                observed_ch_cleaned, fill=True, color="red", ax=ax_ch, label="Observed"
            )
        else:
            ax_ch.text(
                0.5,
                0.5,
                "Not enough observed data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax_ch.transAxes,
                color="gray",
            )

        if null_ch_cleaned.size > 1:
            sns.kdeplot(
                null_ch_cleaned, fill=True, color="skyblue", ax=ax_ch, label="Null"
            )
        else:
            ax_ch.text(
                0.5,
                0.3,
                "Not enough null data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax_ch.transAxes,
                color="gray",
            )

        ax_ch.set_title(f"k={k} Calinski-Harabasz Scores")
        # Only set x-label on the bottom row
        if i == num_k - 1:
            ax_sil.set_xlabel("Score Value")
            ax_ch.set_xlabel("Score Value")
        ax_ch.set_ylabel("Density")
        ax_ch.grid(True, linestyle="--", alpha=0.7)
        ax_ch.legend()
        ax_ch.set_xlim(fixed_ch_xlim)  # Apply fixed x-axis limits

        plt.tight_layout()
    plt.close
    plt.savefig(
        project_path
        / f"Results/MA_Clustering/{meta_name}_clustering_metrics_{correlation_type}_kmedoids_distributions.png"
    )

    # --- Laird/Riedel Metrics Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    k_range_special = range(3, max_clusters + 1)

    color = "tab:blue"
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Variation of Information", color=color)
    ax.plot(k_range_special, np.average(vi_scores, axis=1), "o-", color=color)
    ax.tick_params(axis="y", labelcolor=color)
    ax.grid(True)

    plt.title("Variation of Information Across Cluster Numbers")
    fig.tight_layout()
    plt.savefig(
        project_path
        / f"Results/MA_Clustering/{meta_name}_variation_of_information_{correlation_type}_kmedoids.png"
    )
    plt.close()
