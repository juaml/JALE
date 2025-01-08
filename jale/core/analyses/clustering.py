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
    adjusted_rand_score,
    calinski_harabasz_score,
    silhouette_score,
)
from sklearn.metrics.cluster import entropy, mutual_info_score
from sklearn.utils import resample

from jale.core.utils.compute import compute_ma
from jale.core.utils.folder_setup import folder_setup
from jale.core.utils.kernel import create_kernel_array
from jale.core.utils.template import GM_PRIOR


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
        project_path,
        correlation_matrix,
        correlation_type,
        linkage_method=linkage_method,
    )

    (
        silhouette_scores,
        calinski_harabasz_scores,
        adjusted_rand_index,
        variation_of_information,
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

    silhouette_z, calinski_harabasz_z = compute_metrics_z(
        silhouette_scores,
        calinski_harabasz_scores,
        null_silhouette_scores,
        null_calinski_harabasz_scores,
    )

    plot_clustering_metrics(
        project_path,
        silhouette_scores_z=silhouette_z,
        calinski_harabasz_scores_z=calinski_harabasz_z,
        adjusted_rand_index=adjusted_rand_index,
        variation_of_information=variation_of_information,
        correlation_type=correlation_type,
        clustering_method=clustering_method,
        linkage_method=linkage_method,
    )

    save_clustering_metrics(
        project_path,
        silhouette_scores=silhouette_scores,
        silhouette_scores_z=silhouette_z,
        calinski_harabasz_scores=calinski_harabasz_scores,
        calinski_harabasz_scores_z=calinski_harabasz_z,
        adjusted_rand_index=adjusted_rand_index,
        variation_of_information=variation_of_information,
        correlation_type=correlation_type,
        clustering_method=clustering_method,
        linkage_method=linkage_method,
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
    condensed_distance = squareform(correlation_distance, checks=False)

    silhouette_scores = np.empty((max_clusters - 1, sampling_iterations))
    calinski_harabasz_scores = np.empty((max_clusters - 1, sampling_iterations))
    adjusted_rand_index = np.empty((max_clusters - 1, sampling_iterations))
    variation_of_information = np.empty((max_clusters - 1, sampling_iterations))

    # Iterate over different values of k, compute cluster metrics
    for k in range(2, max_clusters + 1):
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
                cluster_labels = fcluster(Z, k, criterion="maxclust")
            elif clustering_method == "kmeans":
                # Perform K-Means clustering
                kmeans = KMeans(n_clusters=k, random_state=i).fit(resampled_correlation)
                cluster_labels = kmeans.labels_
            else:
                raise ValueError(
                    "Invalid clustering_method. Choose 'hierarchical' or 'kmeans'."
                )

            # Silhouette Score
            silhouette_avg = silhouette_score(
                resampled_correlation
                if clustering_method == "kmeans"
                else resampled_distance,
                cluster_labels,
                metric="euclidean" if clustering_method == "kmeans" else "precomputed",
            )
            silhouette_scores[k - 2, i] = silhouette_avg

            # Calinski-Harabasz Index
            calinski_harabasz_avg = calinski_harabasz_score(
                resampled_correlation, cluster_labels
            )
            calinski_harabasz_scores[k - 2, i] = calinski_harabasz_avg

            # Random clustering for comparison labels in adjusted rand and variation of information
            random_labels = np.random.randint(0, k, size=resampled_distance.shape[0])
            vof_labels = random_labels

            # Adjusted Rand Score
            adjusted_rand_avg = adjusted_rand_score(cluster_labels, vof_labels)
            adjusted_rand_index[k - 2, i] = adjusted_rand_avg

            # Compute Variation of Information
            vi_score = compute_variation_of_information(cluster_labels, vof_labels)
            variation_of_information[k - 2, i] = vi_score

        plot_sorted_dendrogram(
            project_path,
            linkage_matrix=linkage(correlation_matrix, method=linkage_method),
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
        / f"Results/MA_Clustering/tmp/{meta_name}_adjusted_rand_index_{correlation_type}_{clustering_method}_{linkage_method}.npy",
        adjusted_rand_index,
    )
    np.save(
        project_path
        / f"Results/MA_Clustering/tmp/{meta_name}_variation_of_information_{correlation_type}_{clustering_method}_{linkage_method}.npy",
        variation_of_information,
    )

    silhouette_scores = np.average(silhouette_scores, axis=1)
    calinski_harabasz_scores = np.average(calinski_harabasz_scores, axis=1)
    adjusted_rand_index = np.average(adjusted_rand_index, axis=1)
    variation_of_information = np.average(variation_of_information, axis=1)

    return (
        silhouette_scores,
        calinski_harabasz_scores,
        adjusted_rand_index,
        variation_of_information,
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
            silhouette_avg = silhouette_score(
                correlation_distance
                if clustering_method == "hierarchical"
                else correlation_matrix,
                cluster_labels,
                metric="precomputed"
                if clustering_method == "hierarchical"
                else "euclidean",
            )
            null_silhouette_scores[k - 2, n] = silhouette_avg

            # Calinski-Harabasz Index
            calinski_harabasz_avg = calinski_harabasz_score(
                correlation_matrix, cluster_labels
            )
            null_calinski_harabasz_scores[k - 2, n] = calinski_harabasz_avg

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


def compute_metrics_z(
    silhouette_scores,
    calinski_harabasz_scores,
    null_silhouette_scores,
    null_calinski_harabasz_scores,
):
    null_silhouette_scores_avg = np.average(null_silhouette_scores, axis=1)
    null_calinski_harabasz_scores_avg = np.average(
        null_calinski_harabasz_scores, axis=1
    )

    silhouette_z = (silhouette_scores - null_silhouette_scores_avg) / np.std(
        null_silhouette_scores
    )
    alinski_harabasz_z = (
        calinski_harabasz_scores - null_calinski_harabasz_scores_avg
    ) / np.std(null_calinski_harabasz_scores)

    return silhouette_z, alinski_harabasz_z


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
    silhouette_scores_z,
    calinski_harabasz_scores_z,
    adjusted_rand_index,
    variation_of_information,
    correlation_type,
    clustering_method,
    linkage_method,
):
    plt.figure(figsize=(12, 8))

    # Plot Silhouette Scores
    plt.subplot(4, 1, 1)
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
    plt.subplot(4, 1, 2)
    plt.plot(calinski_harabasz_scores_z, marker="o")
    plt.title("Calinski-Harabasz Scores Z")
    plt.xlabel("Number of Clusters")
    plt.xticks(
        ticks=range(len(calinski_harabasz_scores_z)),
        labels=range(2, len(calinski_harabasz_scores_z) + 2),
    )
    plt.ylabel("Z-Score")
    plt.grid()

    # Plot Adjusted Rand Index
    plt.subplot(4, 1, 3)
    plt.plot(adjusted_rand_index, marker="o")
    plt.title("Adjusted Rand Index")
    plt.xlabel("Number of Clusters")
    plt.xticks(
        ticks=range(len(adjusted_rand_index)),
        labels=range(2, len(adjusted_rand_index) + 2),
    )
    plt.ylabel("aRI-Score")
    plt.grid()

    # Plot Variation of Information
    plt.subplot(4, 1, 4)
    plt.plot(variation_of_information, marker="o")
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


def save_clustering_metrics(
    project_path,
    silhouette_scores,
    silhouette_scores_z,
    calinski_harabasz_scores,
    calinski_harabasz_scores_z,
    adjusted_rand_index,
    variation_of_information,
    correlation_type,
    clustering_method,
    linkage_method,
):
    metrics_df = pd.DataFrame(
        {
            "Number of Clusters": range(2, len(silhouette_scores) + 2),
            "Silhouette Scores": silhouette_scores,
            "Silhouette Scores Z": silhouette_scores_z,
            "Calinski-Harabasz Scores": calinski_harabasz_scores,
            "Calinski-Harabasz Scores Z": calinski_harabasz_scores_z,
            "Adjusted Rand Index": adjusted_rand_index,
            "Variation of Information": variation_of_information,
        }
    )
    metrics_df.to_csv(
        project_path
        / f"Results/MA_Clustering/clustering_metrics_{correlation_type}_{clustering_method}_{linkage_method}.csv",
        index=False,
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
    dendro = dendrogram(
        ordered_linkage_matrix,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=linkage_matrix[-(k - 1), 2],  # Highlight k-clusters
    )
    plt.title("Optimal Leaf Ordered Dendrogram")
    plt.xlabel("Samples")
    plt.ylabel("Distance")

    plt.savefig(
        project_path
        / f"Results/MA_Clustering/dendograms/dendogram_{correlation_type}_{clustering_method}_{linkage_method}_{k}.png",
    )
