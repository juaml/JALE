import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, mutual_info_score, silhouette_score

from jale.core.utils.compute import compute_ma, generate_unique_subsamples
from jale.core.utils.template import GM_PRIOR

logger = logging.getLogger("ale_logger")


def kmeans_clustering_pipeline(
    project_path,
    exp_df,
    meta_name,
    kernels,
    correlation_type,
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
        cluster_labels,
    ) = compute_kmeans_subsampling(
        correlation_matrix=correlation_matrix,
        max_clusters=max_clusters,
        subsample_fraction=subsample_fraction,
        sampling_iterations=sampling_iterations,
    )
    logger.info(f"{meta_name} - starting null calculation")
    (
        null_silhouette_scores,
        null_calinski_harabasz_scores,
    ) = compute_kmeans_null(
        exp_df=exp_df,
        kernels=kernels,
        correlation_type=correlation_type,
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

    vi_scores, hierarchy_indices = compute_kmeans_comparative_metrics(
        correlation_matrix=correlation_matrix,
        max_clusters=max_clusters,
    )

    logger.info(f"{meta_name} - calculating final cluster labels")
    hamming_distance_cluster_labels = compute_hamming_distance_kmeans(
        cluster_labels=cluster_labels,
        max_clusters=max_clusters,
    )
    logger.info(f"{meta_name} - creating output and saving")
    save_kmeans_labels(
        project_path=project_path,
        meta_name=meta_name,
        exp_df=exp_df,
        cluster_labels=hamming_distance_cluster_labels,
        correlation_type=correlation_type,
        max_clusters=max_clusters,
    )
    save_kmeans_metrics(
        project_path=project_path,
        meta_name=meta_name,
        silhouette_scores=silhouette_scores,
        silhouette_scores_z=silhouette_scores_z,
        calinski_harabasz_scores=calinski_harabasz_scores,
        calinski_harabasz_scores_z=calinski_harabasz_scores_z,
        vi_scores=vi_scores,
        hierarchy_indices=hierarchy_indices,
        correlation_type=correlation_type,
    )
    plot_kmeans_metrics(
        project_path=project_path,
        meta_name=meta_name,
        silhouette_scores=silhouette_scores,
        silhouette_scores_z=silhouette_scores_z,
        calinski_harabasz_scores=calinski_harabasz_scores,
        calinski_harabasz_scores_z=calinski_harabasz_scores_z,
        vi_scores=vi_scores,
        hierarchy_indices=hierarchy_indices,
        correlation_type=correlation_type,
    )


def compute_kmeans_subsampling(
    correlation_matrix,
    max_clusters,
    subsample_fraction,
    sampling_iterations,
):
    correlation_distance = 1 - correlation_matrix
    np.fill_diagonal(correlation_distance, 0)

    silhouette_scores = np.empty((max_clusters - 1, sampling_iterations))
    calinski_harabasz_scores = np.empty((max_clusters - 1, sampling_iterations))
    cluster_labels = np.full(
        (max_clusters - 1, correlation_matrix.shape[0], sampling_iterations), np.nan
    )

    subsamples = generate_unique_subsamples(
        total_n=correlation_matrix.shape[0],
        target_n=int(subsample_fraction * correlation_matrix.shape[0]),
        sample_n=sampling_iterations,
    )
    # Iterate over different values of k, compute cluster metrics
    for k in range(2, max_clusters + 1):
        for i in range(sampling_iterations):
            # Resample indices for subsampling
            resampled_indices = subsamples[i]
            resampled_correlation = correlation_matrix[
                np.ix_(resampled_indices, resampled_indices)
            ]

            (
                silhouette_score,
                calinski_harabasz_score,
                cluster_label,
            ) = compute_kmeans_clustering(
                correlation_matrix=resampled_correlation,
                k=k,
            )
            silhouette_scores[k - 2, i] = silhouette_score
            calinski_harabasz_scores[k - 2, i] = calinski_harabasz_score
            cluster_labels[k - 2, resampled_indices, i] = cluster_label

    return (
        silhouette_scores,
        calinski_harabasz_scores,
        cluster_labels,
    )


def compute_kmeans_null(
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
            ) = compute_kmeans_clustering(
                correlation_matrix=correlation_matrix,
                k=k,
            )
            null_silhouette_scores[k - 2, n] = silhouette_score
            null_calinski_harabasz_scores[k - 2, n] = calinski_harabasz_score

    return (null_silhouette_scores, null_calinski_harabasz_scores)


def compute_kmeans_clustering(correlation_matrix, k):
    # Perform hierarchical clustering
    kmeans = KMeans(n_clusters=k).fit(correlation_matrix)
    cluster_labels = kmeans.labels_

    # Silhouette Score
    silhouette = silhouette_score(
        correlation_matrix,
        cluster_labels,
        metric="euclidean",
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


def compute_variation_of_information(cluster_labels_prev, cluster_labels_current):
    """
    Computes the variation of information (VI) between two clustering solutions at k-1 and k.
    """
    mutual_info = mutual_info_score(cluster_labels_prev, cluster_labels_current)
    entropy_k = entropy(np.bincount(cluster_labels_prev))
    entropy_k_next = entropy(np.bincount(cluster_labels_current))

    return entropy_k + entropy_k_next - 2 * mutual_info


def compute_hierarchy_index(cluster_labels_prev, cluster_labels_current):
    """
    Computes the hierarchy index, measuring how clusters split when increasing k.

    Parameters:
    cluster_labels_prev: array - Cluster assignments for k-1 clusters
    cluster_labels_current: array - Cluster assignments for k clusters
    """
    unique_prev_clusters = np.unique(cluster_labels_prev)

    split_counts = []
    for cluster in unique_prev_clusters:
        mask = cluster_labels_prev == cluster
        unique_splits = len(np.unique(cluster_labels_current[mask]))
        split_counts.append(unique_splits)

    return np.mean(split_counts)


def compute_kmeans_comparative_metrics(correlation_matrix, max_clusters):
    """
    Performs KMeans clustering on the full correlation matrix for each k and computes
    clustering metrics: variation of information (VI) and hierarchy index.

    Parameters:
    correlation_matrix: 2D numpy array - Correlation matrix
    max_clusters: int - Maximum number of clusters to evaluate

    Returns:
    Dictionary containing computed metrics for each k.
    """
    correlation_distance = 1 - correlation_matrix
    np.fill_diagonal(correlation_distance, 0)

    vi_scores = [0]
    hierarchy_indices = [0]
    prev_labels = None
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k).fit(correlation_distance)
        cluster_labels = kmeans.labels_

        if prev_labels is not None:
            vi_score = compute_variation_of_information(prev_labels, cluster_labels)
            hierarchy_index = compute_hierarchy_index(prev_labels, cluster_labels)
            vi_scores.append(vi_score)
            hierarchy_indices.append(hierarchy_index)

        prev_labels = cluster_labels

    return vi_scores, hierarchy_indices


def compute_hamming_distance_kmeans(cluster_labels, max_clusters):
    hamming_distance_cluster_labels = np.empty(
        (max_clusters - 1, cluster_labels.shape[1])
    )
    for k in range(2, max_clusters + 1):
        hamming_distance = compute_hamming_with_nan(
            cluster_labels=cluster_labels[k - 2]
        )
        kmeans = KMeans(n_clusters=k).fit(hamming_distance)
        hamming_distance_cluster_labels[k - 2, :] = kmeans.labels_

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


def save_kmeans_labels(
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
        / f"Results/MA_Clustering/labels/{meta_name}_cluster_labels_{correlation_type}_kmeans.csv",
        index=False,
        header=header,
    )


def save_kmeans_metrics(
    project_path,
    meta_name,
    silhouette_scores,
    silhouette_scores_z,
    calinski_harabasz_scores,
    calinski_harabasz_scores_z,
    vi_scores,
    hierarchy_indices,
    correlation_type,
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
            "Variation of Information": vi_scores,
            "Hierarchy Index": hierarchy_indices,
        }
    )
    metrics_df.to_csv(
        project_path
        / f"Results/MA_Clustering/{meta_name}_clustering_metrics_{correlation_type}_kmeans.csv",
        index=False,
    )


def plot_kmeans_metrics(
    project_path,
    meta_name,
    silhouette_scores,
    silhouette_scores_z,
    calinski_harabasz_scores,
    calinski_harabasz_scores_z,
    vi_scores,
    hierarchy_indices,
    correlation_type,
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
        ticks=range(len(calinski_harabasz_scores_z)),
        labels=range(2, len(calinski_harabasz_scores_z) + 2),
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

    # Plot Variation of Information
    plt.subplot(6, 1, 5)
    plt.plot(vi_scores, marker="o")
    plt.title("Variation of Information")
    plt.xlabel("Number of Clusters")
    plt.xticks(
        ticks=range(len(vi_scores)),
        labels=range(2, len(vi_scores) + 2),
    )
    plt.ylabel("Score")

    # Plot Hierarchy Index
    plt.subplot(6, 1, 6)
    plt.plot(hierarchy_indices, marker="o")
    plt.title("Hierarchy Index")
    plt.xlabel("Number of Clusters")
    plt.xticks(
        ticks=range(len(hierarchy_indices)),
        labels=range(2, len(hierarchy_indices) + 2),
    )
    plt.ylabel("Score")

    plt.tight_layout()
    plt.savefig(
        project_path
        / f"Results/MA_Clustering/{meta_name}_clustering_metrics_{correlation_type}_kmeans.png"
    )
