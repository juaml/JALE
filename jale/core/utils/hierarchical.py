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
from scipy.stats import entropy, spearmanr
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
        relative_coph_dists,
        cluster_similarity_entropy,
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
        relative_coph_dists=relative_coph_dists,
        matrix_similarity_entropy=cluster_similarity_entropy,
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
        relative_coph_dists=relative_coph_dists,
        matrix_similarity_entropy=cluster_similarity_entropy,
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
    relative_coph_dists = np.empty((max_clusters - 2, sampling_iterations))
    cluster_similarity_entropy = np.empty((max_clusters - 2, sampling_iterations))
    cluster_labels = np.full(
        (max_clusters - 1, correlation_matrix.shape[0], sampling_iterations), np.nan
    )

    subsamples = generate_unique_subsamples(
        total_n=correlation_matrix.shape[0],
        target_n=int(subsample_fraction * correlation_matrix.shape[0]),
        sample_n=sampling_iterations,
    )

    for i in range(sampling_iterations):
        # Resample indices for subsampling
        resampled_indices = subsamples[i]

        # Extract subsampled matrices once
        resampled_correlation = correlation_matrix[
            np.ix_(resampled_indices, resampled_indices)
        ]
        resampled_distance = correlation_distance[
            np.ix_(resampled_indices, resampled_indices)
        ]
        np.fill_diagonal(resampled_distance, 0)

        between_cluster_cophenet = []
        for k in range(2, max_clusters + 1):
            # Compute clustering
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

            between_cluster_cophenet_k = compute_between_cluster_cophenet(
                distance_matrix=resampled_distance,
                linkage_method=linkage_method,
                k=k,
                agg="mean",
            )

            between_cluster_cophenet.append(between_cluster_cophenet_k)

            if k > 2:
                relative_coph_dist = (
                    between_cluster_cophenet[k - 2] - between_cluster_cophenet[k - 3]
                ) / (between_cluster_cophenet[k - 2] + 1e-12)
                relative_coph_dists[k - 3, i] = relative_coph_dist

                entropy_dict = compute_cluster_similarity_entropy(
                    labels_k=cluster_labels[k - 3, resampled_indices, i],
                    labels_kplus1=cluster_labels[k - 2, resampled_indices, i],
                )
                cluster_similarity_entropy[k - 3, i] = entropy_dict["mean_entropy"]

    return (
        silhouette_scores,
        calinski_harabasz_scores,
        relative_coph_dists,
        cluster_similarity_entropy,
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


def compute_between_cluster_cophenet(
    distance_matrix, linkage_method="average", k=2, agg="mean"
):
    """
    Compute average cophenetic distance between clusters at model order k.

    Parameters
    ----------
    distance_matrix : ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix (e.g., 1 - correlation).

    linkage_method : str
        Linkage method for hierarchical clustering.

    k : int
        Number of clusters to cut tree into.

    agg : str
        Aggregation method: "mean", "min", or "max"

    Returns
    -------
    avg_between_dist : float
        Mean cophenetic distance between different clusters.
    """

    condensed_dist = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_dist, method=linkage_method)
    coph_dist_matrix = squareform(cophenet(Z, condensed_dist)[1])

    cluster_labels = fcluster(Z, t=k, criterion="maxclust")

    between_dists = []

    for i in range(k):
        indices_i = np.where(cluster_labels == (i + 1))[0]
        for j in range(i + 1, k):
            indices_j = np.where(cluster_labels == (j + 1))[0]
            # Compute all pairwise distances between clusters i and j
            sub_dists = coph_dist_matrix[np.ix_(indices_i, indices_j)]
            if agg == "mean":
                val = np.mean(sub_dists)
            elif agg == "min":
                val = np.min(sub_dists)
            elif agg == "max":
                val = np.max(sub_dists)
            else:
                raise ValueError("Invalid agg method")
            between_dists.append(val)

    return np.mean(between_dists) if between_dists else np.nan


def compute_cluster_similarity_entropy(labels_k, labels_kplus1):
    """
    Computes a similarity matrix between cluster assignments at two model orders (k and k+1),
    along with entropy-based fragmentation scores.

    This function measures how each cluster at model order `k` distributes its members across
    clusters at model order `k+1`. It uses entropy to quantify how "fragmented" or "stable"
    each cluster's assignment becomes when increasing the number of clusters.

    Parameters
    ----------
    labels_k : array-like of shape (n_samples,)
        Cluster labels for each sample at model order k.

    labels_kplus1 : array-like of shape (n_samples,)
        Cluster labels for each sample at model order k+1.

    Returns
    -------
    result : dict with keys
        - "similarity_matrix" : ndarray of shape (n_k, n_kplus1)
            The normalized matrix where each row shows the distribution of a cluster at k
            over clusters at k+1.

        - "entropies" : ndarray of shape (n_k,)
            Entropy values (in bits) for each cluster at k, indicating fragmentation.

        - "mean_entropy" : float
            Average entropy across all clusters at model order k — a global fragmentation score.

    Notes
    -----
    - Entropy is computed with base 2.
    - A small epsilon is added to avoid log(0) in entropy calculation.
    - This method assumes a fixed number of samples with cluster labels at both model orders.
    """
    labels_k = np.array(labels_k)
    labels_kplus1 = np.array(labels_kplus1)

    # Unique clusters
    clusters_k = np.unique(labels_k)
    clusters_kplus1 = np.unique(labels_kplus1)

    n_k = len(clusters_k)
    n_kplus1 = len(clusters_kplus1)

    # Build the similarity matrix S: rows = clusters_k, cols = clusters_kplus1
    S = np.zeros((n_k, n_kplus1))

    # Mapping from cluster label to index
    k_index = {label: i for i, label in enumerate(clusters_k)}
    kplus1_index = {label: j for j, label in enumerate(clusters_kplus1)}

    for label in clusters_k:
        indices = np.where(labels_k == label)[0]
        next_labels = labels_kplus1[indices]
        counts = np.bincount(
            [kplus1_index[label] for label in next_labels], minlength=n_kplus1
        )
        S[k_index[label], :] = counts / len(indices)  # Normalize to make row sum to 1

    # Compute entropy per cluster at model order k
    entropies = np.array(
        [entropy(row + 1e-12, base=2) for row in S]
    )  # small epsilon to avoid log(0)
    mean_entropy = np.mean(entropies)

    return {
        "similarity_matrix": S,
        "entropies": entropies,
        "mean_entropy": mean_entropy,
    }


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
    relative_coph_dists,
    cluster_similarity_entropy,
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
            "Cophenet Scores": [0, np.average(relative_coph_dists, axis=1)],
            "Cluster Similarity Entropy": [
                0,
                np.average(cluster_similarity_entropy, axis=1),
            ],
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
    relative_coph_dists,
    cluster_similarity_entropy,
    correlation_type,
    linkage_method,
):
    plt.figure(figsize=(12, 15))

    # Plot Silhouette Scores
    plt.subplot(5, 1, 1)
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
    plt.subplot(5, 1, 2)
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
    plt.subplot(5, 1, 3)
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
    plt.subplot(5, 1, 4)
    plt.plot(calinski_harabasz_scores_z, marker="o")
    plt.title("Calinski-Harabasz Scores Z")
    plt.xlabel("Number of Clusters")
    plt.xticks(
        ticks=range(len(calinski_harabasz_scores_z)),
        labels=range(2, len(calinski_harabasz_scores_z) + 2),
    )
    plt.ylabel("Z-Score")
    plt.grid()

    plt.tight_layout()
    plt.savefig(
        project_path
        / f"Results/MA_Clustering/{meta_name}_clustering_metrics_{correlation_type}_hc_{linkage_method}.png"
    )

    plt.figure(figsize=(12, 6))
    # Plot Cophenet Scores
    plt.subplot(2, 1, 1)
    plt.plot(relative_coph_dists, marker="o")
    plt.title("Relative Cophenet Scores (compared to K-1)")
    plt.xlabel("Number of Clusters")
    plt.xticks(
        ticks=range(len(relative_coph_dists)),
        labels=range(3, len(relative_coph_dists) + 3),
    )
    plt.ylabel("Score")
    plt.grid()

    # Plot Cluster Similarity Entropy
    plt.subplot(2, 1, 2)
    plt.plot(cluster_similarity_entropy, marker="o")
    plt.title("Cluster Similarity Entropy")
    plt.xlabel("Number of Clusters")
    plt.xticks(
        ticks=range(len(cluster_similarity_entropy)),
        labels=range(3, len(cluster_similarity_entropy) + 3),
    )
    plt.ylabel("Entropy")
    plt.grid()

    plt.tight_layout()
    plt.savefig(
        project_path
        / f"Results/MA_Clustering/{meta_name}_clustering_metrics_{correlation_type}_hc_{linkage_method}_coph_entropy.png"
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
