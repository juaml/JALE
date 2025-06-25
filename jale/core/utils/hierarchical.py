import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        rel_diff_cophenetic,
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
        silhouette_scores_z=silhouette_scores_z,
        calinski_harabasz_scores=calinski_harabasz_scores,
        calinski_harabasz_scores_z=calinski_harabasz_scores_z,
        rel_diff_cophenetic=rel_diff_cophenetic,
        exp_separation_density=exp_separation_density,
        correlation_type=correlation_type,
        linkage_method=linkage_method,
    )
    plot_hc_metrics(
        project_path=project_path,
        meta_name=meta_name,
        silhouette_scores=silhouette_scores,
        silhouette_scores_z=silhouette_scores_z,
        calinski_harabasz_scores=calinski_harabasz_scores,
        calinski_harabasz_scores_z=calinski_harabasz_scores_z,
        rel_diff_cophenetic=rel_diff_cophenetic,
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
    rel_diff_cophenetic = np.empty((max_clusters - 2, sampling_iterations))
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
        rdc = calculate_rel_diff_cophenetic(Z, max_clusters)
        # Store results, aligning index with k (k=3 to max_clusters)
        rel_diff_cophenetic[:, i] = [
            rdc.get(k, np.nan) for k in range(3, max_clusters + 1)
        ]

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
        rel_diff_cophenetic,
        exp_separation_density,
        cluster_labels,
    )


def calculate_rel_diff_cophenetic(linkage_matrix, max_clusters):
    """Calculates the relative difference in cophenetic distance."""
    cophenetic_distances = linkage_matrix[:, 2]
    rev_cophenetic_distances = cophenetic_distances[::-1]

    rel_diff_cophenetic = {}
    # Linkage matrix has n-1 rows for n samples.
    # We can evaluate transitions up to n-1 clusters.
    num_possible_clusters = len(rev_cophenetic_distances)

    for i in range(num_possible_clusters - 1):
        num_clusters_to = (
            i + 3
        )  # Corresponds to transition to k+1 clusters in the paper's plot
        if num_clusters_to > max_clusters:
            break

        # The denominator is the cophenetic distance when merging from k+1 to k clusters
        denominator = rev_cophenetic_distances[i]
        if denominator > 1e-12:  # Avoid division by zero
            # The numerator is the difference between k+1->k and k->k-1 merges
            numerator = rev_cophenetic_distances[i] - rev_cophenetic_distances[i + 1]
            dc = numerator / denominator
            rel_diff_cophenetic[num_clusters_to] = dc
        else:
            rel_diff_cophenetic[num_clusters_to] = 0

    return rel_diff_cophenetic


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
        if correlation_type == "spearman":
            correlation_matrix, _ = spearmanr(ma_gm_masked, axis=1)
        elif correlation_type == "pearson":
            correlation_matrix = np.corrcoef(ma_gm_masked)
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

    # Silhouette Score
    silhouette = silhouette_score(
        distance_matrix,
        cluster_labels,
        metric="precomputed",
    )

    # Calinski-Harabasz Index
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
    silhouette_scores_z,
    calinski_harabasz_scores,
    calinski_harabasz_scores_z,
    rel_diff_cophenetic,
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
            "Silhouette Scores Z": silhouette_scores_z,
            "Calinski-Harabasz Scores": np.average(calinski_harabasz_scores, axis=1),
            "Calinski-Harabasz Scores SD": np.std(calinski_harabasz_scores, axis=1),
            "Calinski-Harabasz Scores Z": calinski_harabasz_scores_z,
            # Pad with NaN for k=2 as metrics start at k=3
            "Relative Difference Cophenetic": np.concatenate(
                ([np.nan], np.nanmean(rel_diff_cophenetic, axis=1))
            ),
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


def plot_hc_metrics(
    project_path,
    meta_name,
    silhouette_scores,
    silhouette_scores_z,
    calinski_harabasz_scores,
    calinski_harabasz_scores_z,
    rel_diff_cophenetic,
    exp_separation_density,
    correlation_type,
    linkage_method,
    max_clusters,
):
    # --- Standard Metrics Plot ---
    plt.figure(figsize=(12, 15))
    k_range = range(2, max_clusters + 1)

    plt.subplot(4, 1, 1)
    plt.plot(k_range, np.average(silhouette_scores, axis=1), marker="o")
    plt.title("Silhouette Scores")
    plt.ylabel("Score")
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(k_range, silhouette_scores_z, marker="o")
    plt.title("Silhouette Scores Z")
    plt.ylabel("Z-Score")
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(k_range, np.average(calinski_harabasz_scores, axis=1), marker="o")
    plt.title("Calinski-Harabasz Scores")
    plt.ylabel("Score")
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(k_range, calinski_harabasz_scores_z, marker="o")
    plt.title("Calinski-Harabasz Scores Z")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Z-Score")
    plt.grid()

    plt.tight_layout()
    plt.savefig(
        project_path
        / f"Results/MA_Clustering/{meta_name}_clustering_metrics_{correlation_type}_hc_{linkage_method}.png"
    )
    plt.close()

    # --- Laird/Riedel Metrics Plot ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    k_range_special = range(3, max_clusters + 1)

    color = "tab:blue"
    ax1.set_xlabel("Number of Clusters Transitioning To")
    ax1.set_ylabel("Relative Difference in Cophenetic Distances", color=color)
    ax1.plot(
        k_range_special, np.nanmean(rel_diff_cophenetic, axis=1), "o-", color=color
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, axis="x")

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Experiment Separation Density", color=color)
    # The density metric is for the transition from k to k+1, so it aligns with k_range_special
    ax2.plot(
        k_range_special,
        np.nanmean(exp_separation_density, axis=1),
        "s-",
        color=color,
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Cluster Evaluation Metrics")
    fig.tight_layout()
    plt.savefig(
        project_path
        / f"Results/MA_Clustering/{meta_name}_clustering_metrics_{correlation_type}_hc_{linkage_method}_laird_riedel.png"
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
