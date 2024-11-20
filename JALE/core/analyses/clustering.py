import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    silhouette_score,
)
from sklearn.utils import resample
from sklearn_extra.cluster import KMedoids

from jale.core.utils.compute import compute_ma
from jale.core.utils.kernel import create_kernel_array
from jale.core.utils.template import GM_PRIOR


def plot_cor_matrix(project_path, correlation_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, cmap="RdBu_r", center=0, vmin=-1, vmax=1)

    # Add title and labels
    plt.title("Correlation Matrix with Custom Colormap")
    plt.xlabel("Experiments")
    plt.ylabel("Experiments")

    plt.savefig(project_path / "Results/MA_Clustering/correlation_matrix.png")


def plot_clustering_metrics(
    project_path,
    silhouette_scores_z,
    calinski_harabasz_scores_z,
    adjusted_rand_scores_z,
):
    plt.figure(figsize=(12, 8))

    # Plot Silhouette Scores
    plt.subplot(3, 1, 1)
    plt.plot(silhouette_scores_z, marker="o")
    plt.title("Silhouette Scores Z")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Z-Score")
    plt.grid()

    # Plot Calinski-Harabasz Scores
    plt.subplot(3, 1, 2)
    plt.plot(calinski_harabasz_scores_z, marker="o")
    plt.title("Calinski-Harabasz Scores Z")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Z-Score")
    plt.grid()

    # Plot Adjusted Rand Scores
    plt.subplot(3, 1, 3)
    plt.plot(adjusted_rand_scores_z, marker="o")
    plt.title("Adjusted Rand Scores Z")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Z-Score")
    plt.grid()

    plt.tight_layout()
    plt.savefig(project_path / "Results/MA_Clustering/clustering_metrics.png")
    plt.show()


def compute_clustering(
    meta_name,
    project_path,
    correlation_matrix,
    max_clusters=10,
    subsample_fraction=0.9,
    sampling_iterations=500,
):
    # Step 2: Convert correlation matrix to correlation distance (1 - r)
    correlation_distance = 1 - correlation_matrix

    # Step 4: Evaluate the optimal number of clusters from 2 to 15 using various metrics

    silhouette_scores = np.empty((max_clusters - 1, sampling_iterations))
    calinski_harabasz_scores = np.empty((max_clusters - 1, sampling_iterations))
    adjusted_rand_scores = np.empty((max_clusters - 1, sampling_iterations))

    # Using the K-medoids algorithm for calculating variation of information and adjusted rand index
    for k in range(2, max_clusters + 1):
        for i in range(sampling_iterations):
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

            # Convert to condensed form for hierarchical clustering
            condensed_resampled_distance = squareform(resampled_distance, checks=False)

            # Step 2: Perform hierarchical clustering on the resampled data
            Z = linkage(condensed_resampled_distance, method="complete")

            # Step 3: Extract clusters for the specified number of clusters (e.g., 5 clusters)
            cluster_labels = fcluster(Z, k, criterion="maxclust")

            # Silhouette Score
            silhouette_avg = silhouette_score(
                resampled_distance, cluster_labels, metric="precomputed"
            )
            silhouette_scores[k - 2, i] = silhouette_avg

            # Calinski-Harabasz Index
            calinski_harabasz_avg = calinski_harabasz_score(
                resampled_correlation, cluster_labels
            )
            calinski_harabasz_scores[k - 2, i] = calinski_harabasz_avg

            # Variation of Information (using K-medoids clustering as baseline)
            kmedoids = KMedoids(n_clusters=k, metric="precomputed").fit(
                resampled_distance
            )
            vof_labels = kmedoids.labels_

            # Variation of Information (adjusted mutual information can be used as a proxy)
            adjusted_rand_avg = adjusted_rand_score(cluster_labels, vof_labels)
            adjusted_rand_scores[k - 2, i] = adjusted_rand_avg

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
        project_path / f"Results/MA_Clustering/{meta_name}_adjusted_rand_scores.npy",
        adjusted_rand_scores,
    )

    return silhouette_scores, calinski_harabasz_scores, adjusted_rand_scores


def compute_permute_clustering(
    meta_name, project_path, exp_df, kernels, max_clusters, null_iterations
):
    null_silhouette_scores = np.empty((max_clusters - 1, null_iterations))
    null_calinski_harabasz_scores = np.empty((max_clusters - 1, null_iterations))
    null_adjusted_rand_scores = np.empty((max_clusters - 1, null_iterations))

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
        correlation_matrix[
            np.isinf(correlation_matrix) | np.isnan(correlation_matrix)
        ] = 0
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

            kmedoids = KMedoids(n_clusters=k, metric="precomputed").fit(
                correlation_distance
            )
            vof_labels = kmedoids.labels_

            adjusted_rand_avg = adjusted_rand_score(cluster_labels, vof_labels)
            null_adjusted_rand_scores[k - 2, n] = adjusted_rand_avg

    np.save(
        project_path / f"Results/MA_Clustering/{meta_name}_null_silhouette_scores.npy",
        null_silhouette_scores,
    )
    np.save(
        project_path
        / f"Results/MA_Clustering/{meta_name}_null_calinski_harabasz_scores.npy",
        null_calinski_harabasz_scores,
    )
    np.save(
        project_path
        / f"Results/MA_Clustering/{meta_name}_null_adjusted_rand_scores.npy",
        null_adjusted_rand_scores,
    )

    return (
        null_silhouette_scores,
        null_calinski_harabasz_scores,
        null_adjusted_rand_scores,
    )


def compute_metrics_z(
    silhouette_scores,
    calinski_harabasz_scores,
    adjusted_rand_scores,
    null_silhouette_scores,
    null_calinski_harabasz_scores,
    null_adjusted_rand_scores,
):
    silhouette_scores_avg = np.average(silhouette_scores, axis=1)
    calinski_harabasz_scores_avg = np.average(calinski_harabasz_scores, axis=1)
    adjusted_rand_scores_avg = np.average(adjusted_rand_scores, axis=1)

    null_silhouette_scores_avg = np.average(null_silhouette_scores, axis=1)
    null_calinski_harabasz_scores_avg = np.average(
        null_calinski_harabasz_scores, axis=1
    )
    null_adjusted_rand_scores_avg = np.average(null_adjusted_rand_scores, axis=1)

    silhouette_z = (silhouette_scores_avg - null_silhouette_scores_avg) / np.std(
        null_silhouette_scores
    )
    alinski_harabasz_z = (
        calinski_harabasz_scores_avg - null_calinski_harabasz_scores_avg
    ) / np.std(null_calinski_harabasz_scores)

    adjusted_rand_z = (
        adjusted_rand_scores_avg - null_adjusted_rand_scores_avg
    ) / np.std(null_adjusted_rand_scores)

    return silhouette_z, alinski_harabasz_z, adjusted_rand_z


def clustering(
    project_path,
    exp_df,
    meta_name,
    max_clusters=10,
    subsample_fraction=0.9,
    sampling_iterations=1000,
    null_iterations=1000,
):
    kernels = create_kernel_array(exp_df)

    ma = compute_ma(exp_df.Coordinates.values, kernels)
    ma_gm_masked = ma[:, GM_PRIOR]

    correlation_matrix, _ = spearmanr(ma_gm_masked, axis=1)
    plot_cor_matrix(project_path, correlation_matrix)

    silhouette_scores, calinski_harabasz_scores, adjusted_rand_scores = (
        compute_clustering(
            meta_name,
            correlation_matrix,
            max_clusters=max_clusters,
            subsample_fraction=0.9,
            sampling_iterations=500,
        )
    )

    null_silhouette_scores, null_calinski_harabasz_scores, null_adjusted_rand_scores = (
        compute_permute_clustering(
            meta_name,
            correlation_matrix,
            max_clusters=max_clusters,
            null_iterations=500,
        )
    )

    silhouette_z, alinski_harabasz_z, adjusted_rand_z = compute_metrics_z(
        silhouette_scores,
        calinski_harabasz_scores,
        adjusted_rand_scores,
        null_silhouette_scores,
        null_calinski_harabasz_scores,
        null_adjusted_rand_scores,
    )

    plot_clustering_metrics(
        project_path,
        silhouette_scores_z=silhouette_z,
        calinski_harabasz_scores_z=alinski_harabasz_z,
        adjusted_rand_scores_z=adjusted_rand_z,
    )
