import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import (
    dendrogram,
    fcluster,
    leaves_list,
    linkage,
    optimal_leaf_ordering,
)
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    rand_score,
    silhouette_score,
)
from sklearn.metrics.cluster import entropy, mutual_info_score
from sklearn.utils import resample

from jale.core.utils.compute import compute_ma
from jale.core.utils.folder_setup import folder_setup
from jale.core.utils.kernel import create_kernel_array
from jale.core.utils.template import GM_PRIOR

logger = logging.getLogger("ale_logger")


def clustering(
    project_path,
    exp_df,
    meta_name,
    correlation_type="spearman",  # spearman or pearson
    clustering_method="hierarchical",  # hierarchical or k-means
    linkage_method="complete",  # complete or average
    max_clusters=10,
    subsample_fraction=0.9,
    sampling_iterations=500,
    null_iterations=1000,
):
    folder_setup(project_path, "MA_Clustering")

    # Save included experiments for provenance tracking
    print_df = pd.DataFrame(
        {
            "Experiment": exp_df.Articles.values,
            "Number of Foci": exp_df.NumberOfFoci.values,
        }
    )
    print_df.to_csv(
        project_path / f"Results/MA_Clustering/{meta_name}_included_experiments.csv",
        index=False,
        sep="\t",
    )

    kernels = create_kernel_array(exp_df)

    ma = compute_ma(exp_df.Coordinates.values, kernels)
    ma_gm_masked = ma[:, GM_PRIOR]

    if correlation_type == "spearman":
        correlation_matrix, _ = spearmanr(ma_gm_masked, axis=1)
    elif correlation_type == "pearson":
        correlation_matrix, _ = pearsonr(ma_gm_masked, axis=1)
    else:
        raise ValueError("Invalid correlation_type. Choose 'spearman' or 'pearson'.")

    plot_cor_matrix(
        project_path=project_path,
        correlation_matrix=correlation_matrix,
        correlation_type=correlation_type,
        linkage_method=linkage_method,
    )

    if subsample_fraction == 1:
        sampling_iterations = 1
    logger.info(f"{meta_name} - calculate true clustering")
    (
        silhouette_scores,
        calinski_harabasz_scores,
        rand_index,
        variation_of_information,
        hamming_distance,
        cluster_labels,
    ) = compute_clustering(
        meta_name,
        project_path,
        correlation_matrix,
        correlation_type=correlation_type,
        linkage_method=linkage_method,
        clustering_method=clustering_method,
        max_clusters=max_clusters,
        subsample_fraction=subsample_fraction,
        sampling_iterations=sampling_iterations,
    )
    logger.info(f"{meta_name} - calculate null clustering")
    null_silhouette_scores, null_calinski_harabasz_scores = compute_permute_clustering(
        meta_name,
        project_path,
        exp_df,
        kernels,
        correlation_type,
        clustering_method=clustering_method,
        linkage_method=linkage_method,
        max_clusters=max_clusters,
        null_iterations=null_iterations,
        subsample_fraction=subsample_fraction,
    )
    logger.info(f"{meta_name} - calculate z metrics")
    silhouette_z, calinski_harabasz_z = compute_metrics_z(
        silhouette_scores=silhouette_scores,
        calinski_harabasz_scores=calinski_harabasz_scores,
        null_silhouette_scores=null_silhouette_scores,
        null_calinski_harabasz_scores=null_calinski_harabasz_scores,
    )
    logger.info(f"{meta_name} - plot metrics")
    plot_clustering_metrics(
        project_path,
        silhouette_scores=silhouette_scores,
        silhouette_scores_z=silhouette_z,
        calinski_harabasz_scores=calinski_harabasz_scores,
        calinski_harabasz_scores_z=calinski_harabasz_z,
        rand_index=rand_index,
        variation_of_information=variation_of_information,
        correlation_type=correlation_type,
        clustering_method=clustering_method,
        linkage_method=linkage_method,
    )
    logger.info(f"{meta_name} - save metrics & labels")
    save_clustering_metrics(
        project_path,
        silhouette_scores=silhouette_scores,
        silhouette_scores_z=silhouette_z,
        calinski_harabasz_scores=calinski_harabasz_scores,
        calinski_harabasz_scores_z=calinski_harabasz_z,
        rand_index=rand_index,
        variation_of_information=variation_of_information,
        correlation_type=correlation_type,
        clustering_method=clustering_method,
        linkage_method=linkage_method,
    )

    save_cluster_labels(
        project_path,
        exp_df,
        cluster_labels=cluster_labels,
        correlation_type=correlation_type,
        clustering_method=clustering_method,
        linkage_method=linkage_method,
        max_clusters=max_clusters,
    )


def compute_clustering(
    meta_name,
    project_path,
    correlation_matrix,
    correlation_type,
    clustering_method,
    linkage_method,
    max_clusters,
    subsample_fraction,
    sampling_iterations,
):
    # Convert correlation matrix to correlation distance (1 - r)
    correlation_distance = 1 - correlation_matrix
    np.fill_diagonal(correlation_distance, 0)

    silhouette_scores = np.empty((max_clusters - 1, sampling_iterations))
    calinski_harabasz_scores = np.empty((max_clusters - 1, sampling_iterations))
    rand_index = np.empty((max_clusters - 1, sampling_iterations))
    variation_of_information = np.empty((max_clusters - 1, sampling_iterations))
    cluster_labels = np.empty((correlation_matrix.shape[0], max_clusters - 1))

    # Iterate over different values of k, compute cluster metrics
    for k in range(2, max_clusters + 1):
        tmp_hamming_distance = np.full(
            (correlation_matrix.shape[0], sampling_iterations), np.nan
        )
        for i in range(sampling_iterations):
            # Resample indices for subsampling
            resampled_indices = resample(
                np.arange(correlation_matrix.shape[0]),
                replace=False,
                n_samples=int(subsample_fraction * correlation_matrix.shape[0]),
            )
            resampled_correlation = correlation_matrix[
                np.ix_(resampled_indices, resampled_indices)
            ]
            resampled_distance = correlation_distance[
                np.ix_(resampled_indices, resampled_indices)
            ]

            # Ensure diagonal is zero for distance matrix
            np.fill_diagonal(resampled_distance, 0)

            if clustering_method == "hierarchical":
                # Convert to condensed form for hierarchical clustering
                condensed_resampled_distance = squareform(
                    resampled_distance, checks=False
                )
                # Perform hierarchical clustering
                Z = linkage(condensed_resampled_distance, method=linkage_method)
                cluster_labels_tmp = fcluster(Z, k, criterion="maxclust")
            elif clustering_method == "kmeans":
                # Perform K-Means clustering
                kmeans = KMeans(n_clusters=k, random_state=i).fit(resampled_correlation)
                cluster_labels_tmp = kmeans.labels_
            else:
                raise ValueError(
                    "Invalid clustering_method. Choose 'hierarchical' or 'kmeans'."
                )

            tmp_hamming_distance[resampled_indices, i] = cluster_labels_tmp

            # Silhouette Score
            silhouette = silhouette_score(
                resampled_correlation
                if clustering_method == "kmeans"
                else resampled_distance,
                cluster_labels_tmp,
                metric="euclidean" if clustering_method == "kmeans" else "precomputed",
            )
            silhouette_scores[k - 2, i] = silhouette

            # Calinski-Harabasz Index
            calinski_harabasz = calinski_harabasz_score(
                resampled_correlation, cluster_labels_tmp
            )
            calinski_harabasz_scores[k - 2, i] = calinski_harabasz

            # Random clustering for comparison labels in rand and variation of information
            random_labels = np.random.randint(0, k, size=resampled_distance.shape[0])

            # Rand Score
            rand_avg = rand_score(cluster_labels_tmp, random_labels)
            rand_index[k - 2, i] = rand_avg

            # Compute Variation of Information
            vi_score = compute_variation_of_information(
                cluster_labels_tmp, random_labels
            )
            variation_of_information[k - 2, i] = vi_score

        hamming_distance = compute_hamming_with_nan(tmp_hamming_distance)

        cluster_labels[:, k - 2] = compute_final_cluster_labels(
            project_path,
            hamming_distance,
            correlation_type,
            clustering_method,
            linkage_method,
            k,
        )

        if clustering_method == "hierarchical":
            condensed_distance = squareform(hamming_distance, checks=False)
            linkage_matrix = linkage(condensed_distance, method=linkage_method)
            plot_sorted_dendrogram(
                project_path,
                linkage_matrix=linkage_matrix,
                distance_matrix=condensed_distance,
                correlation_type=correlation_type,
                clustering_method=clustering_method,
                linkage_method=linkage_method,
                k=k,
            )

    # Save results
    np.save(
        project_path
        / f"Results/MA_Clustering/tmp/{meta_name}_silhouette_scores_{correlation_type}_{clustering_method}_{linkage_method}.npy",
        silhouette_scores,
    )
    np.save(
        project_path
        / f"Results/MA_Clustering/tmp/{meta_name}_calinski_harabasz_scores_{correlation_type}_{clustering_method}_{linkage_method}.npy",
        calinski_harabasz_scores,
    )
    np.save(
        project_path
        / f"Results/MA_Clustering/tmp/{meta_name}_rand_index_{correlation_type}_{clustering_method}_{linkage_method}.npy",
        rand_index,
    )
    np.save(
        project_path
        / f"Results/MA_Clustering/tmp/{meta_name}_variation_of_information_{correlation_type}_{clustering_method}_{linkage_method}.npy",
        variation_of_information,
    )

    return (
        silhouette_scores,
        calinski_harabasz_scores,
        rand_index,
        variation_of_information,
        hamming_distance,
        cluster_labels,
    )


def compute_permute_clustering(
    meta_name,
    project_path,
    exp_df,
    kernels,
    correlation_type,
    clustering_method,
    linkage_method,
    max_clusters,
    null_iterations,
    subsample_fraction,
):
    null_silhouette_scores = np.empty((max_clusters - 1, null_iterations))
    null_calinski_harabasz_scores = np.empty((max_clusters - 1, null_iterations))

    for n in range(null_iterations):
        # Create an index array for subsampling
        sampled_indices = np.random.choice(
            exp_df.index, size=int(subsample_fraction * len(exp_df)), replace=False
        )

        # Subsample exp_df and kernels using the sampled indices
        sampled_exp_df = exp_df.iloc[sampled_indices].reset_index(drop=True)
        sampled_kernels = [kernels[idx] for idx in sampled_indices]

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
        correlation_matrix, _ = spearmanr(ma_gm_masked, axis=1)
        correlation_matrix = np.nan_to_num(
            correlation_matrix, nan=0, posinf=0, neginf=0
        )
        correlation_distance = 1 - correlation_matrix

        if clustering_method == "hierarchical":
            condensed_distance = squareform(correlation_distance, checks=False)
            Z = linkage(condensed_distance, method=linkage_method)
        elif clustering_method == "kmeans":
            pass  # No preprocessing needed for K-Means

        for k in range(2, max_clusters + 1):
            if clustering_method == "hierarchical":
                # Step 5: Extract clusters for k clusters
                cluster_labels = fcluster(Z, k, criterion="maxclust")
            elif clustering_method == "kmeans":
                kmeans = KMeans(n_clusters=k, random_state=n).fit(correlation_matrix)
                cluster_labels = kmeans.labels_
            else:
                raise ValueError(
                    "Invalid clustering_method. Choose 'hierarchical' or 'kmeans'."
                )

            # Silhouette Score
            null_silhouette = silhouette_score(
                correlation_distance
                if clustering_method == "hierarchical"
                else correlation_matrix,
                cluster_labels,
                metric="precomputed"
                if clustering_method == "hierarchical"
                else "euclidean",
            )
            null_silhouette_scores[k - 2, n] = null_silhouette

            # Calinski-Harabasz Index
            null_calinski_harabasz = calinski_harabasz_score(
                correlation_matrix, cluster_labels
            )
            null_calinski_harabasz_scores[k - 2, n] = null_calinski_harabasz

    # Save results
    np.save(
        project_path
        / f"Results/MA_Clustering/tmp/{meta_name}_null_silhouette_scores_{correlation_type}_{clustering_method}_{linkage_method}.npy",
        null_silhouette_scores,
    )
    np.save(
        project_path
        / f"Results/MA_Clustering/tmp/{meta_name}_null_calinski_harabasz_scores_{correlation_type}_{clustering_method}_{linkage_method}.npy",
        null_calinski_harabasz_scores,
    )

    return (
        null_silhouette_scores,
        null_calinski_harabasz_scores,
    )


def compute_metrics_z(
    silhouette_scores,
    calinski_harabasz_scores,
    null_silhouette_scores,
    null_calinski_harabasz_scores,
):
    silhouette_scores_avg = np.average(silhouette_scores, axis=1)
    null_silhouette_scores_avg = np.average(null_silhouette_scores, axis=1)
    calinski_harabasz_scores_avg = np.average(calinski_harabasz_scores, axis=1)
    null_calinski_harabasz_scores_avg = np.average(
        null_calinski_harabasz_scores, axis=1
    )

    silhouette_z = (silhouette_scores_avg - null_silhouette_scores_avg) / np.std(
        null_silhouette_scores
    )
    calinski_harabasz_z = (
        calinski_harabasz_scores_avg - null_calinski_harabasz_scores_avg
    ) / np.std(null_calinski_harabasz_scores)

    return silhouette_z, calinski_harabasz_z


def compute_final_cluster_labels(
    project_path,
    hamming_distance,
    correlation_type,
    clustering_method,
    linkage_method,
    k,
):
    if clustering_method == "hierarchical":
        condensed_distance = squareform(hamming_distance, checks=False)
        linkage_matrix = linkage(condensed_distance, method=linkage_method)
        cluster_labels = fcluster(linkage_matrix, t=k, criterion="maxclust")
    elif clustering_method == "kmeans":
        kmeans = KMeans(n_clusters=k, random_state=0).fit(hamming_distance)
        cluster_labels = kmeans.labels_

    # Save cluster labels
    np.savetxt(
        project_path
        / f"Results/MA_Clustering/labels/cluster_labels_{correlation_type}_{clustering_method}_{linkage_method}_{k}.txt",
        cluster_labels.astype(int),
        fmt="%d",
    )

    return cluster_labels


def compute_variation_of_information(labels_true, labels_pred):
    """
    Compute the Variation of Information (VI) metric.

    Parameters:
    labels_true (array-like): Ground truth cluster labels.
    labels_pred (array-like): Predicted cluster labels.

    Returns:
    float: VI score.
    """
    # Compute entropy for each clustering
    H_true = entropy(np.bincount(labels_true))
    H_pred = entropy(np.bincount(labels_pred))

    # Compute mutual information
    I_uv = mutual_info_score(labels_true, labels_pred)

    # Compute Variation of Information
    return H_true + H_pred - 2 * I_uv


def compute_hamming_with_nan(data):
    # Precompute valid masks
    valid_masks = ~np.isnan(data)

    # Initialize matrix for results
    n = data.shape[0]
    hamming_matrix = np.full((n, n), np.nan)

    # Iterate through pairs using broadcasting
    for i in range(n):
        valid_i = valid_masks[i]
        for j in range(i + 1, n):
            valid_j = valid_masks[j]
            valid_mask = valid_i & valid_j
            total_valid = np.sum(valid_mask)
            if total_valid > 0:
                mismatches = np.sum(data[i, valid_mask] != data[j, valid_mask])
                hamming_matrix[i, j] = mismatches / total_valid
                hamming_matrix[j, i] = hamming_matrix[i, j]
            else:
                print(i, j)

    np.fill_diagonal(hamming_matrix, 0)
    return hamming_matrix


def save_cluster_labels(
    project_path,
    exp_df,
    cluster_labels,
    correlation_type,
    clustering_method,
    linkage_method,
    max_clusters,
):
    # Generate dynamic header from k=2 to k=max_clusters
    header = ["Experiment"] + [f"k={k}" for k in range(2, max_clusters + 1)]

    # Create DataFrame
    cluster_labels_df = pd.DataFrame(
        np.column_stack([exp_df.Articles.values, cluster_labels]), columns=header
    )

    # Save as CSV
    cluster_labels_df.to_csv(
        project_path
        / f"Results/MA_Clustering/labels/cluster_labels_{correlation_type}_{clustering_method}_{linkage_method}.csv",
        index=False,
        header=header,
    )


def save_clustering_metrics(
    project_path,
    silhouette_scores,
    silhouette_scores_z,
    calinski_harabasz_scores,
    calinski_harabasz_scores_z,
    rand_index,
    variation_of_information,
    correlation_type,
    clustering_method,
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
            "Rand Index": np.average(rand_index, axis=1),
            "Variation of Information": np.average(variation_of_information, axis=1),
        }
    )
    metrics_df.to_csv(
        project_path
        / f"Results/MA_Clustering/clustering_metrics_{correlation_type}_{clustering_method}_{linkage_method}.csv",
        index=False,
    )


def plot_cor_matrix(project_path, correlation_matrix, correlation_type, linkage_method):
    # Perform hierarchical clustering
    linkage_matrix = linkage(correlation_matrix, method=linkage_method)

    # Get the ordering of rows/columns
    ordered_indices = leaves_list(linkage_matrix)

    # Reorder the correlation matrix
    sorted_correlation_matrix = correlation_matrix[ordered_indices][:, ordered_indices]
    plt.figure(figsize=(8, 6))
    sns.heatmap(sorted_correlation_matrix, cmap="RdBu_r", center=0, vmin=-1, vmax=1)

    # Add title and labels
    plt.title("Correlation Matrix with Custom Colormap")
    plt.xlabel("Experiments")
    plt.xticks(ticks=[])
    plt.ylabel("Experiments")
    plt.yticks(ticks=[])

    plt.savefig(
        project_path
        / f"Results/MA_Clustering/correlation_matrix_{correlation_type}_{linkage_method}.png"
    )


def plot_clustering_metrics(
    project_path,
    silhouette_scores,
    silhouette_scores_z,
    calinski_harabasz_scores,
    calinski_harabasz_scores_z,
    rand_index,
    variation_of_information,
    correlation_type,
    clustering_method,
    linkage_method,
):
    plt.figure(figsize=(12, 20))

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

    # Plot Rand Index
    plt.subplot(6, 1, 5)
    plt.plot(np.average(rand_index, axis=1), marker="o")
    plt.title("Rand Index")
    plt.xlabel("Number of Clusters")
    plt.xticks(
        ticks=range(len(rand_index)),
        labels=range(2, len(rand_index) + 2),
    )
    plt.ylabel("RI-Score")
    plt.grid()

    # Plot Variation of Information
    plt.subplot(6, 1, 6)
    plt.plot(np.average(variation_of_information, axis=1), marker="o")
    plt.title("Variation of Information")
    plt.xlabel("Number of Clusters")
    plt.xticks(
        ticks=range(len(variation_of_information)),
        labels=range(2, len(variation_of_information) + 2),
    )
    plt.ylabel("VI-Score")
    plt.grid()

    plt.tight_layout()
    plt.savefig(
        project_path
        / f"Results/MA_Clustering/clustering_metrics_{correlation_type}_{clustering_method}_{linkage_method}.png"
    )


def plot_sorted_dendrogram(
    project_path,
    linkage_matrix,
    distance_matrix,
    correlation_type,
    clustering_method,
    linkage_method,
    k,
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
    ordered_linkage_matrix = optimal_leaf_ordering(linkage_matrix, distance_matrix)

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
        / f"Results/MA_Clustering/dendograms/dendogram_{correlation_type}_{clustering_method}_{linkage_method}_{k}.png",
    )
