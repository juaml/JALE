import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
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
from sklearn_extra.cluster import KMedoids

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
    sampling_iterations=1000,
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

    plot_cor_matrix(project_path, correlation_matrix, linkage_method=linkage_method)

    (
        silhouette_scores,
        calinski_harabasz_scores,
        adjusted_rand_index,
        variation_of_information,
    ) = compute_clustering(
        meta_name,
        project_path,
        correlation_matrix,
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
        max_clusters=max_clusters,
        null_iterations=null_iterations,
    )

    silhouette_z, alinski_harabasz_z = compute_metrics_z(
        silhouette_scores,
        calinski_harabasz_scores,
        null_silhouette_scores,
        null_calinski_harabasz_scores,
    )

    plot_clustering_metrics(
        project_path,
        silhouette_scores_z=silhouette_z,
        calinski_harabasz_scores_z=alinski_harabasz_z,
        adjusted_rand_index_z=adjusted_rand_index,
        voi_scores_z=variation_of_information,
    )

    save_clustering_metrics(
        project_path,
        silhouette_scores=silhouette_scores,
        silhouette_scores_z=silhouette_z,
        calinski_harabasz_scores=calinski_harabasz_scores,
        calinski_harabasz_scores_z=alinski_harabasz_z,
        adjusted_rand_index_z=adjusted_rand_index,
        voi_scores_z=variation_of_information,
    )


def compute_clustering(
    meta_name,
    project_path,
    correlation_matrix,
    clustering_method="hierarchical",
    linkage_method="complete",
    max_clusters=10,
    subsample_fraction=0.9,
    sampling_iterations=500,
):
    # Convert correlation matrix to correlation distance (1 - r)
    correlation_distance = 1 - correlation_matrix

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

            # K-Medoids for comparison labels in adjusted rand and variation of information
            kmedoids = KMedoids(n_clusters=k, metric="precomputed").fit(
                resampled_distance
            )
            vof_labels = kmedoids.labels_

            # Adjusted Rand Score
            adjusted_rand_avg = adjusted_rand_score(cluster_labels, vof_labels)
            adjusted_rand_index[k - 2, i] = adjusted_rand_avg

            # Compute Variation of Information
            vi_score = compute_variation_of_information(cluster_labels, vof_labels)
            variation_of_information[k - 2, i] = vi_score

    # Save results
    np.save(
        project_path / f"Results/MA_Clustering/{meta_name}_silhouette_scores.npy",
        silhouette_scores,
    )
    np.save(
        project_path
        / f"Results/MA_Clustering/{meta_name}_calinski_harabasz_scores.npy",
        calinski_harabasz_scores,
    )
    np.save(
        project_path / f"Results/MA_Clustering/{meta_name}_adjusted_rand_index.npy",
        adjusted_rand_index,
    )
    np.save(
        project_path
        / f"Results/MA_Clustering/{meta_name}_variation_of_information.npy",
        variation_of_information,
    )

    return (
        silhouette_scores,
        calinski_harabasz_scores,
        adjusted_rand_index,
        variation_of_information,
    )


def compute_permute_clustering(
    meta_name, project_path, exp_df, kernels, max_clusters, null_iterations
):
    null_silhouette_scores = np.empty((max_clusters - 1, null_iterations))
    null_calinski_harabasz_scores = np.empty((max_clusters - 1, null_iterations))

    for n in range(null_iterations):
        coords_stacked = np.vstack(exp_df.Coordinates.values)
        shuffled_coords = []
        for exp in np.arange(exp_df.shape[0]):
            K = exp_df.loc[exp, "NumberOfFoci"]
            # Step 1: Randomly sample K unique row indices
            sample_indices = np.random.choice(
                coords_stacked.shape[0], size=K, replace=False
            )
            # Step 2: Extract the sampled rows using the sampled indices
            sampled_rows = coords_stacked[sample_indices]
            shuffled_coords.append(sampled_rows)
            # Step 3: Delete the sampled rows from the original array
            coords_stacked = np.delete(coords_stacked, sample_indices, axis=0)

        null_ma = compute_ma(shuffled_coords, kernels)
        ma_gm_masked = null_ma[:, GM_PRIOR]
        correlation_matrix, _ = spearmanr(ma_gm_masked, axis=1)
        correlation_matrix = np.nan_to_num(
            correlation_matrix, nan=0, posinf=0, neginf=0
        )
        correlation_distance = 1 - correlation_matrix
        condensed_distance = squareform(correlation_distance, checks=False)
        Z = linkage(condensed_distance, method="average")

        for k in range(2, max_clusters + 1):
            # Step 5: Extract clusters for k clusters
            cluster_labels = fcluster(Z, k, criterion="maxclust")

            # Silhouette Score
            silhouette_avg = silhouette_score(
                correlation_distance, cluster_labels, metric="precomputed"
            )
            null_silhouette_scores[k - 2, n] = silhouette_avg

            # Calinski-Harabasz Index
            calinski_harabasz_avg = calinski_harabasz_score(
                correlation_matrix, cluster_labels
            )
            null_calinski_harabasz_scores[k - 2, n] = calinski_harabasz_avg

    np.save(
        project_path / f"Results/MA_Clustering/{meta_name}_null_silhouette_scores.npy",
        null_silhouette_scores,
    )
    np.save(
        project_path
        / f"Results/MA_Clustering/{meta_name}_null_calinski_harabasz_scores.npy",
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
    silhouette_scores_avg = np.average(silhouette_scores, axis=1)
    calinski_harabasz_scores_avg = np.average(calinski_harabasz_scores, axis=1)

    null_silhouette_scores_avg = np.average(null_silhouette_scores, axis=1)
    null_calinski_harabasz_scores_avg = np.average(
        null_calinski_harabasz_scores, axis=1
    )

    silhouette_z = (silhouette_scores_avg - null_silhouette_scores_avg) / np.std(
        null_silhouette_scores
    )
    alinski_harabasz_z = (
        calinski_harabasz_scores_avg - null_calinski_harabasz_scores_avg
    ) / np.std(null_calinski_harabasz_scores)

    return silhouette_z, alinski_harabasz_z


def plot_cor_matrix(project_path, correlation_matrix, linkage_method="average"):
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

    plt.savefig(project_path / "Results/MA_Clustering/correlation_matrix.png")


def plot_clustering_metrics(
    project_path,
    silhouette_scores_z,
    calinski_harabasz_scores_z,
    adjusted_rand_index,
    variation_of_info,
):
    plt.figure(figsize=(12, 8))

    # Plot Silhouette Scores
    plt.subplot(4, 1, 1)
    plt.plot(silhouette_scores_z, marker="o")
    plt.title("Silhouette Scores Z")
    plt.xlabel("Number of Clusters")
    plt.xticks(ticks=range(len(silhouette_scores_z)), labels=range(2, 11))
    plt.ylabel("Z-Score")
    plt.grid()

    # Plot Calinski-Harabasz Scores
    plt.subplot(4, 1, 2)
    plt.plot(calinski_harabasz_scores_z, marker="o")
    plt.title("Calinski-Harabasz Scores Z")
    plt.xlabel("Number of Clusters")
    plt.xticks(ticks=range(len(calinski_harabasz_scores_z)), labels=range(2, 11))
    plt.ylabel("Z-Score")
    plt.grid()

    # Plot Adjusted Rand Index
    plt.subplot(4, 1, 3)
    plt.plot(adjusted_rand_index, marker="o")
    plt.title("Adjusted Rand Index")
    plt.xlabel("Number of Clusters")
    plt.xticks(ticks=range(len(adjusted_rand_index)), labels=range(2, 11))
    plt.ylabel("aRI-Score")
    plt.grid()

    # Plot Variation of Information
    plt.subplot(4, 1, 4)
    plt.plot(variation_of_info, marker="o")
    plt.title("Variation of Information")
    plt.xlabel("Number of Clusters")
    plt.xticks(ticks=range(len(variation_of_info)), labels=range(2, 11))
    plt.ylabel("VI-Score")
    plt.grid()

    plt.tight_layout()
    plt.savefig(project_path / "Results/MA_Clustering/clustering_metrics.png")
    plt.show()


def save_clustering_metrics(
    project_path,
    silhouette_scores,
    silhouette_scores_z,
    calinski_harabasz_scores,
    calinski_harabasz_scores_z,
    adjusted_rand_index,
    variation_of_info,
):
    metrics_df = pd.DataFrame(
        {
            "Number of Clusters": range(2, 11),
            "Silhouette Scores": silhouette_scores,
            "Silhouette Scores Z": silhouette_scores_z,
            "Calinski-Harabasz Scores": calinski_harabasz_scores,
            "Calinski-Harabasz Scores Z": calinski_harabasz_scores_z,
            "Adjusted Rand Index": adjusted_rand_index,
            "Variation of Information": variation_of_info,
        }
    )
    metrics_df.to_csv(
        project_path / "Results/MA_Clustering/clustering_metrics.csv", index=False
    )
