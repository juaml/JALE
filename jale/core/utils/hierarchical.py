import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import (
    cophenet,
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
        seperation_densities,
        cophenet_scores,
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
        seperation_densities=seperation_densities,
        cophenet_scores=cophenet_scores,
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
        seperation_densities=seperation_densities,
        cophenet_scores=cophenet_scores,
        correlation_type=correlation_type,
        linkage_method=linkage_method,
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
    correlation_distance = 1 - correlation_matrix
    np.fill_diagonal(correlation_distance, 0)

    silhouette_scores = np.empty((max_clusters - 1, sampling_iterations))
    calinski_harabasz_scores = np.empty((max_clusters - 1, sampling_iterations))
    seperation_densities = np.empty((max_clusters - 1, sampling_iterations))
    cluster_labels = np.full(
        (max_clusters - 1, correlation_matrix.shape[0], sampling_iterations), np.nan
    )

    subsamples = generate_unique_subsamples(
        total_n=correlation_matrix.shape[0],
        target_n=int(subsample_fraction * correlation_matrix.shape[0]),
        sample_n=sampling_iterations,
    )
    # Iterate over different values of k, compute cluster metrics
    cophenet_distance = []
    for k in range(2, max_clusters + 1):
        for i in range(sampling_iterations):
            # Resample indices for subsampling
            resampled_indices = subsamples[i]
            resampled_correlation = correlation_matrix[
                np.ix_(resampled_indices, resampled_indices)
            ]
            resampled_distance = correlation_distance[
                np.ix_(resampled_indices, resampled_indices)
            ]

            # Ensure diagonal is zero for distance matrix
            np.fill_diagonal(resampled_distance, 0)

            (
                silhouette_score,
                calinski_harabasz_score,
                cluster_label,
            ) = compute_hierarchical_clustering(
                correlation_matrix=resampled_correlation,
                k=k,
                linkage_method=linkage_method,
            )
            silhouette_scores[k - 2, i] = silhouette_score
            calinski_harabasz_scores[k - 2, i] = calinski_harabasz_score
            cluster_labels[k - 2, resampled_indices, i] = cluster_label

            seperation_densities[k - 2, i] = compute_seperation_density(
                cluster_labels=cluster_label
            )

        # compute within cluster cophenetic distance
        within_cluster_cophonet = compute_within_cluster_cophenet(
            cluster_labels=cluster_labels[k - 2],
            linkage_method=linkage_method,
            distance_matrix=correlation_distance,
        )
        cophenet_distance.append(within_cluster_cophonet)

        # compute seperation density

    # Compute relative cophenetic distances
    relative_cophenetic_distances = [0]
    for i in range(max_clusters - 3):
        c_current = cophenet_distance[i]
        c_next = cophenet_distance[i + 1]
        relative_dc = (c_next - c_current) / c_next if c_next != 0 else 0
        relative_cophenetic_distances.append(relative_dc)
    relative_cophenetic_distances.append(0)

    return (
        silhouette_scores,
        calinski_harabasz_scores,
        seperation_densities,
        relative_cophenetic_distances,
        cluster_labels,
    )


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


def compute_within_cluster_cophenet(cluster_labels, linkage_method, distance_matrix):
    within_cluster_cophenet = []
    for cluster_id in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cluster_id)[0]
        if len(indices) > 1:  # Only calculate if there are at least two points
            sub_distances = squareform(
                distance_matrix[np.ix_(indices, indices)], checks=False
            )
            sub_Z = linkage(sub_distances, method=linkage_method)
            cophenet_score = cophenet(sub_Z, sub_distances)[0]
            within_cluster_cophenet.append(cophenet_score)
    within_cluster_cophenet = np.average(within_cluster_cophenet)
    return within_cluster_cophenet


def compute_seperation_density(cluster_labels):
    # Compute cluster sizes manually
    unique_labels = np.unique(cluster_labels)
    cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]

    # Sort cluster sizes in descending order
    cluster_sizes.sort(reverse=True)
    separation_density = cluster_sizes[0] / sum(cluster_sizes)  # n1 / n0

    return separation_density


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
    seperation_densities,
    cophenet_scores,
    correlation_type,
    linkage_method,
):
    metrics_df = pd.DataFrame(
        {
            "Number of Clusters": range(2, len(silhouette_scores) + 2),
            "Silhouette Scores": np.average(silhouette_scores, axis=1),
            "Silhouette Scores SD": np.std(silhouette_scores, axis=1),
            "Silhouette Scores Z": silhouette_scores_z,
            "Calinski-Harabasz Scores": np.average(calinski_harabasz_scores, axis=1),
            "Calinski-Harabasz Scores SD": np.std(calinski_harabasz_scores, axis=1),
            "Calinski-Harabasz Scores Z": calinski_harabasz_scores_z,
            "Seperation Density": np.average(seperation_densities, axis=1),
            "Cophenet Scores": cophenet_scores,
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
    seperation_densities,
    cophenet_scores,
    correlation_type,
    linkage_method,
):
    plt.figure(figsize=(12, 15))

    # Plot Silhouette Scores
    plt.subplot(6, 1, 1)
    plt.plot(np.average(silhouette_scores, axis=1), marker="o")
    plt.title("Silhouette Scores")
    plt.xlabel("Number of Clusters")
    plt.xticks(
        ticks=range(len(silhouette_scores)),
        labels=range(2, len(silhouette_scores) + 2),
    )
    plt.ylabel("Score")
    plt.grid()

    # Plot Silhouette Scores Z
    plt.subplot(6, 1, 2)
    plt.plot(silhouette_scores_z, marker="o")
    plt.title("Silhouette Scores Z")
    plt.xlabel("Number of Clusters")
    plt.xticks(
        ticks=range(len(silhouette_scores_z)),
        labels=range(2, len(silhouette_scores_z) + 2),
    )
    plt.ylabel("Z-Score")
    plt.grid()

    # Plot Calinski-Harabasz Scores
    plt.subplot(6, 1, 3)
    plt.plot(np.average(calinski_harabasz_scores, axis=1), marker="o")
    plt.title("Calinski-Harabasz Scores")
    plt.xlabel("Number of Clusters")
    plt.xticks(
        ticks=range(len(calinski_harabasz_scores)),
        labels=range(2, len(calinski_harabasz_scores) + 2),
    )
    plt.ylabel("Score")
    plt.grid()

    # Plot Calinski-Harabasz Scores Z
    plt.subplot(6, 1, 4)
    plt.plot(calinski_harabasz_scores_z, marker="o")
    plt.title("Calinski-Harabasz Scores Z")
    plt.xlabel("Number of Clusters")
    plt.xticks(
        ticks=range(len(calinski_harabasz_scores_z)),
        labels=range(2, len(calinski_harabasz_scores_z) + 2),
    )
    plt.ylabel("Z-Score")
    plt.grid()

    # Plot Seperation Density
    plt.subplot(6, 1, 5)
    plt.plot(np.average(seperation_densities, axis=1), marker="o")
    plt.title("Seperation Density")
    plt.xlabel("Number of Clusters")
    plt.xticks(
        ticks=range(len(seperation_densities)),
        labels=range(2, len(seperation_densities) + 2),
    )
    plt.ylabel("Density")

    # Plot Cophenet Scores
    plt.subplot(6, 1, 6)
    plt.plot(cophenet_scores, marker="o")
    plt.title("Cophenet Scores")
    plt.xlabel("Number of Clusters")
    plt.xticks(
        ticks=range(len(cophenet_scores)),
        labels=range(2, len(cophenet_scores) + 2),
    )
    plt.ylabel("Score")
    plt.grid()

    plt.tight_layout()
    plt.savefig(
        project_path
        / f"Results/MA_Clustering/{meta_name}_clustering_metrics_{correlation_type}_hc_{linkage_method}.png"
    )


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
