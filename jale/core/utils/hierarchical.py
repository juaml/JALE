import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import (
    dendrogram,
    fcluster,
    linkage,
    optimal_leaf_ordering,
)
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import (
    calinski_harabasz_score,
    silhouette_score,
)

from jale.core.utils.compute import compute_ma, generate_unique_subsamples
from jale.core.utils.template import GM_PRIOR


def hierarchical_clustering_pipeline(
    project_path,
    meta_name,
    exp_df,
    kernels,
    correlation_type,
    correlation_matrix,
    linkage_method,
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
        exp_separation_density,
        cluster_labels,
    ) = compute_hc_subsampling(
        correlation_matrix=correlation_matrix,
        max_clusters=max_clusters,
        subsample_fraction=subsample_fraction,
        sampling_iterations=sampling_iterations,
        linkage_method=linkage_method,
    )
    logger.info(f"{meta_name} - starting null calculation")
    (
        null_silhouette_scores,
        null_calinski_harabasz_scores,
    ) = compute_hc_null(
        exp_df=exp_df,
        kernels=kernels,
        correlation_type=correlation_type,
        linkage_method=linkage_method,
        max_clusters=max_clusters,
        null_iterations=null_iterations,
        subsample_fraction=subsample_fraction,
    )
    silhouette_scores_z, calinski_harabasz_scores_z = compute_hc_metrics_z(
        silhouette_scores=silhouette_scores,
        calinski_harabasz_scores=calinski_harabasz_scores,
        null_silhouette_scores=null_silhouette_scores,
        null_calinski_harabasz_scores=null_calinski_harabasz_scores,
        use_pooled_std=use_pooled_std,
    )
    logger.info(f"{meta_name} - calculating final cluster labels")
    hamming_distance_cluster_labels = compute_hamming_distance_hc(
        cluster_labels=cluster_labels,
        linkage_method=linkage_method,
        max_clusters=max_clusters,
    )
    logger.info(f"{meta_name} - creating output and saving")
    save_hc_labels(
        project_path=project_path,
        exp_df=exp_df,
        meta_name=meta_name,
        cluster_labels=hamming_distance_cluster_labels,
        correlation_type=correlation_type,
        linkage_method=linkage_method,
        max_clusters=max_clusters,
    )
    save_hc_metrics(
        project_path=project_path,
        meta_name=meta_name,
        silhouette_scores=silhouette_scores,
        null_silhouette_scores=null_silhouette_scores,
        silhouette_scores_z=silhouette_scores_z,
        calinski_harabasz_scores=calinski_harabasz_scores,
        null_calinski_harabasz_scores=null_calinski_harabasz_scores,
        calinski_harabasz_scores_z=calinski_harabasz_scores_z,
        exp_separation_density=exp_separation_density,
        correlation_type=correlation_type,
        linkage_method=linkage_method,
    )
    plot_hc_metrics(
        project_path=project_path,
        meta_name=meta_name,
        silhouette_scores=silhouette_scores,
        null_silhouette_scores=null_silhouette_scores,
        silhouette_scores_z=silhouette_scores_z,
        calinski_harabasz_scores=calinski_harabasz_scores,
        null_calinski_harabasz_scores=null_calinski_harabasz_scores,
        calinski_harabasz_scores_z=calinski_harabasz_scores_z,
        exp_separation_density=exp_separation_density,
        correlation_type=correlation_type,
        linkage_method=linkage_method,
        max_clusters=max_clusters,
    )
    plot_sorted_dendrogram(
        project_path=project_path,
        meta_name=meta_name,
        correlation_matrix=correlation_matrix,
        correlation_type=correlation_type,
        linkage_method=linkage_method,
        max_clusters=max_clusters,
    )


def compute_hc_subsampling(
    correlation_matrix,
    max_clusters,
    subsample_fraction,
    sampling_iterations,
    linkage_method,
):
    silhouette_scores = np.empty((max_clusters - 1, sampling_iterations))
    calinski_harabasz_scores = np.empty((max_clusters - 1, sampling_iterations))
    exp_separation_density = np.empty((max_clusters - 2, sampling_iterations))
    cluster_labels = np.full(
        (max_clusters - 1, correlation_matrix.shape[0], sampling_iterations), np.nan
    )

    subsamples = generate_unique_subsamples(
        total_n=correlation_matrix.shape[0],
        target_n=int(subsample_fraction * correlation_matrix.shape[0]),
        sample_n=sampling_iterations,
    )

    for i in range(sampling_iterations):
        resampled_indices = subsamples[i]
        resampled_correlation = correlation_matrix[
            np.ix_(resampled_indices, resampled_indices)
        ]

        # Perform hierarchical clustering once per subsample
        distance_matrix = 1 - resampled_correlation
        np.fill_diagonal(distance_matrix, 0)
        condensed_distance = squareform(distance_matrix, checks=False)
        Z = linkage(condensed_distance, method=linkage_method)

        # Calculate relative difference in cophenetic distance for the whole hierarchy
        # rel_diff_cophenetic[:, i] = calculate_rel_diff_cophenetic(Z, max_clusters)

        for k in range(2, max_clusters + 1):
            cluster_label = fcluster(Z, k, criterion="maxclust")

            # Silhouette Score
            silhouette = silhouette_score(
                distance_matrix, cluster_label, metric="precomputed"
            )

            # Calinski-Harabasz Index
            calinski_harabasz = calinski_harabasz_score(
                resampled_correlation, cluster_label
            )

            silhouette_scores[k - 2, i] = silhouette
            calinski_harabasz_scores[k - 2, i] = calinski_harabasz
            cluster_labels[k - 2, resampled_indices, i] = cluster_label

            # Calculate experiment separation density for transition from k to k+1
            if k < max_clusters:
                esd = calculate_exp_separation_density(Z, k)
                exp_separation_density[k - 2, i] = esd

    return (
        silhouette_scores,
        calinski_harabasz_scores,
        exp_separation_density,
        cluster_labels,
    )


def calculate_exp_separation_density(linkage_matrix, k):
    """Calculates the experiment separation density for the transition from k to k+1 clusters."""
    if k + 1 > linkage_matrix.shape[0] + 1:
        return np.nan  # Cannot form k+1 clusters

    clusters_k = fcluster(linkage_matrix, k, criterion="maxclust")
    clusters_k_plus_1 = fcluster(linkage_matrix, k + 1, criterion="maxclust")

    split_cluster_id = -1
    new_labels = []

    # Find which cluster in the k-cluster solution was split
    for cluster_id in range(1, k + 1):
        member_indices = np.where(clusters_k == cluster_id)[0]
        sub_labels = np.unique(clusters_k_plus_1[member_indices])
        if len(sub_labels) > 1:
            split_cluster_id = cluster_id
            new_labels = sub_labels
            break

    if split_cluster_id != -1:
        low_size = np.sum(clusters_k == split_cluster_id)
        high1_size = np.sum(clusters_k_plus_1 == new_labels[0])
        high2_size = np.sum(clusters_k_plus_1 == new_labels[1])

        if low_size > 0:
            return max(high1_size, high2_size) / low_size

    return np.nan  # Should not be reached in a valid split


def compute_hc_null(
    exp_df,
    kernels,
    correlation_type,
    linkage_method,
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
        # Replace any NaNs in the correlation matrix with zero for further calculations
        correlation_matrix = np.nan_to_num(
            correlation_matrix, nan=0, posinf=0, neginf=0
        )
        np.fill_diagonal(correlation_matrix, 0)

        for k in range(2, max_clusters + 1):
            (
                silhouette_score,
                calinski_harabasz_score,
                cluster_label,
            ) = compute_hierarchical_clustering(
                correlation_matrix=correlation_matrix,
                k=k,
                linkage_method=linkage_method,
            )
            null_silhouette_scores[k - 2, n] = silhouette_score
            null_calinski_harabasz_scores[k - 2, n] = calinski_harabasz_score

    return (
        null_silhouette_scores,
        null_calinski_harabasz_scores,
    )


def compute_hierarchical_clustering(correlation_matrix, k, linkage_method):
    distance_matrix = 1 - correlation_matrix
    np.fill_diagonal(distance_matrix, 0)
    condensed_distance = squareform(distance_matrix, checks=False)
    # Perform hierarchical clustering
    Z = linkage(condensed_distance, method=linkage_method)
    cluster_labels = fcluster(Z, k, criterion="maxclust")

    # Safeguard: If number of cluster labels is smaller than k, set metrics to np.nan
    if len(np.unique(cluster_labels)) < 2:
        silhouette = np.nan
        calinski_harabasz = np.nan
    else:
        silhouette = silhouette_score(
            distance_matrix,
            cluster_labels,
            metric="precomputed",
        )
        calinski_harabasz = calinski_harabasz_score(correlation_matrix, cluster_labels)

    return (
        silhouette,
        calinski_harabasz,
        cluster_labels,
    )


def compute_hc_metrics_z(
    silhouette_scores,
    calinski_harabasz_scores,
    null_silhouette_scores,
    null_calinski_harabasz_scores,
    use_pooled_std=False,
):
    def pooled_std(sample1, sample2):
        """Compute the pooled standard deviation of two samples."""
        n1, n2 = sample1.shape[1], sample2.shape[1]
        var1, var2 = np.nanvar(sample1, axis=1, ddof=1), np.nanvar(sample2, axis=1, ddof=1)
        return np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    silhouette_scores_avg = np.nanmean(silhouette_scores, axis=1)
    null_silhouette_scores_avg = np.nanmean(null_silhouette_scores, axis=1)

    if use_pooled_std:
        silhouette_std = pooled_std(silhouette_scores, null_silhouette_scores)
    else:
        silhouette_std = np.nanstd(null_silhouette_scores, axis=1, ddof=1)

    silhouette_z = (silhouette_scores_avg - null_silhouette_scores_avg) / silhouette_std

    calinski_harabasz_scores_avg = np.nanmean(calinski_harabasz_scores, axis=1)
    null_calinski_harabasz_scores_avg = np.nanmean(
        null_calinski_harabasz_scores, axis=1
    )

    if use_pooled_std:
        calinski_harabasz_std = pooled_std(
            calinski_harabasz_scores, null_calinski_harabasz_scores
        )
    else:
        calinski_harabasz_std = np.nanstd(null_calinski_harabasz_scores, axis=1, ddof=1)

    calinski_harabasz_z = (
        calinski_harabasz_scores_avg - null_calinski_harabasz_scores_avg
    ) / calinski_harabasz_std

    return silhouette_z, calinski_harabasz_z


def compute_hamming_distance_hc(cluster_labels, linkage_method, max_clusters):
    hamming_distance_cluster_labels = np.empty(
        (max_clusters - 1, cluster_labels.shape[1])
    )
    for k in range(2, max_clusters + 1):
        hamming_distance = compute_hamming_with_nan(
            cluster_labels=cluster_labels[k - 2]
        )
        condensed_distance = squareform(hamming_distance, checks=False)
        linkage_matrix = linkage(condensed_distance, method=linkage_method)
        hamming_distance_cluster_labels[k - 2] = fcluster(
            linkage_matrix, t=k, criterion="maxclust"
        )

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


def save_hc_labels(
    project_path,
    exp_df,
    meta_name,
    cluster_labels,
    correlation_type,
    linkage_method,
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
        / f"Results/MA_Clustering/labels/{meta_name}_cluster_labels_{correlation_type}_hc_{linkage_method}.csv",
        index=False,
        header=header,
    )


def save_hc_metrics(
    project_path,
    meta_name,
    silhouette_scores,
    null_silhouette_scores,
    silhouette_scores_z,
    calinski_harabasz_scores,
    null_calinski_harabasz_scores,
    calinski_harabasz_scores_z,
    exp_separation_density,
    correlation_type,
    linkage_method,
):
    max_k = len(silhouette_scores) + 1
    metrics_df = pd.DataFrame(
        {
            "Number of Clusters": range(2, max_k + 1),
            "Silhouette Scores": np.average(silhouette_scores, axis=1),
            "Silhouette Scores SD": np.std(silhouette_scores, axis=1),
            "Silhouette Scores NULL": np.nanmean(null_silhouette_scores, axis=1),
            "Silhouette Scores NULL SD": np.nanstd(null_silhouette_scores, axis=1),
            "Silhouette Scores Z": silhouette_scores_z,
            "Calinski-Harabasz Scores": np.average(calinski_harabasz_scores, axis=1),
            "Calinski-Harabasz Scores SD": np.std(calinski_harabasz_scores, axis=1),
            "Calinski-Harabasz Scores NULL": np.nanmean(null_calinski_harabasz_scores, axis=1),
            "Calinski-Harabasz Scores NULL SD": np.nanstd(null_calinski_harabasz_scores, axis=1),
            "Calinski-Harabasz Scores Z": calinski_harabasz_scores_z,
            # Pad with NaN for k=2 as metrics start at k=3
            "Experiment Separation Density": np.concatenate(
                ([np.nan], np.nanmean(exp_separation_density, axis=1))
            ),
        }
    )
    metrics_df.to_csv(
        project_path
        / f"Results/MA_Clustering/{meta_name}_clustering_metrics_{correlation_type}_hc_{linkage_method}.csv",
        index=False,
    )

    pd.DataFrame(silhouette_scores.T).to_csv(
        project_path
        / f"Results/MA_Clustering/metrics/{meta_name}_silhouette_scores_{correlation_type}_hc_{linkage_method}.csv",
        index=False,
        header=[f"k={k}" for k in range(2, max_k + 1)],
    )

    pd.DataFrame(null_silhouette_scores.T).to_csv(
        project_path
        / f"Results/MA_Clustering/metrics/{meta_name}_NULLsilhouette_scores_{correlation_type}_hc_{linkage_method}.csv",
        index=False,
        header=[f"k={k}" for k in range(2, max_k + 1)],
    )

    pd.DataFrame(calinski_harabasz_scores.T).to_csv(
        project_path
        / f"Results/MA_Clustering/metrics/{meta_name}_calinski_harabasz_scores_{correlation_type}_hc_{linkage_method}.csv",
        index=False,
        header=[f"k={k}" for k in range(2, max_k + 1)],
    )

    pd.DataFrame(null_calinski_harabasz_scores.T).to_csv(
        project_path
        / f"Results/MA_Clustering/metrics/{meta_name}_NULLcalinski_harabasz_scores_{correlation_type}_hc_{linkage_method}.csv",
        index=False,
        header=[f"k={k}" for k in range(2, max_k + 1)],
    )

    pd.DataFrame(exp_separation_density.T).to_csv(
        project_path
        / f"Results/MA_Clustering/metrics/{meta_name}_exp_separation_density_{correlation_type}_hc_{linkage_method}.csv",
        index=False,
        header=[f"k={k}" for k in range(3, max_k + 1)],
    )


def plot_hc_metrics(
    project_path,
    meta_name,
    silhouette_scores,
    null_silhouette_scores,
    silhouette_scores_z,
    calinski_harabasz_scores,
    null_calinski_harabasz_scores,
    calinski_harabasz_scores_z,
    exp_separation_density,
    correlation_type,
    linkage_method,
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
        f"Clustering Metrics for {meta_name} ({correlation_type} correlation, {linkage_method} linkage)",
        y=0.99,
        fontsize=16,
        fontweight="bold",
    )  # Overall title for the figure
    plt.tight_layout(
        rect=[0, 0.03, 1, 0.96]
    )  # Adjust layout to make space for suptitle
    plt.savefig(
        project_path
        / f"Results/MA_Clustering/{meta_name}_clustering_metrics_{correlation_type}_hc_{linkage_method}.png"
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
        / f"Results/MA_Clustering/{meta_name}_clustering_metrics_{correlation_type}_hc_{linkage_method}_distributions.png"
    )

    # --- Laird/Riedel Metrics Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    k_range_special = range(3, max_clusters + 1)
    color = "tab:red"
    ax.set_xlabel("Number of Clusters Transitioning To")
    ax.set_ylabel("Experiment Separation Density", color=color)
    ax.plot(
        k_range_special,
        np.nanmean(exp_separation_density, axis=1),
        "s-",
        color=color,
    )
    ax.tick_params(axis="y", labelcolor=color)
    ax.grid(True, axis="x")
    plt.title("Experiment Separation Density")
    fig.tight_layout()
    plt.savefig(
        project_path
        / f"Results/MA_Clustering/{meta_name}_exp_separation_density_{correlation_type}_hc_{linkage_method}_laird_riedel.png"
    )
    plt.close()


def plot_sorted_dendrogram(
    project_path,
    meta_name,
    correlation_type,
    correlation_matrix,
    linkage_method,
    max_clusters,
):
    """
    Creates a dendrogram with optimal leaf ordering for better interpretability.

    Parameters:
        linkage_matrix (ndarray): The linkage matrix from hierarchical clustering.
        data (ndarray): Original data used to compute the distance matrix.

    Returns:
        dict: The dendrogram structure.
    """
    # Apply optimal leaf ordering to the linkage matrix
    distance_matrix = 1 - correlation_matrix
    condensed_distance = squareform(distance_matrix, checks=False)
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance, method=linkage_method)
    ordered_linkage_matrix = optimal_leaf_ordering(linkage_matrix, condensed_distance)
    for k in range(2, max_clusters + 1):
        # Plot the dendrogram
        plt.figure(figsize=(10, 6))
        dendrogram(
            ordered_linkage_matrix,
            leaf_rotation=90,
            leaf_font_size=10,
            color_threshold=linkage_matrix[-(k - 1), 2],  # Highlight k-clusters
        )
        plt.title("Optimal Leaf Ordered Dendrogram")
        plt.xlabel("Experiments")
        plt.ylabel("Distance")
        plt.xticks([])

        plt.savefig(
            project_path
            / f"Results/MA_Clustering/dendograms/{meta_name}_dendogram_{correlation_type}_hc_{linkage_method}_{k}.png",
        )
        plt.close()  # Close figure to avoid displaying it in notebooks
