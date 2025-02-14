import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import spearmanr

from jale.core.utils.compute import compute_ma
from jale.core.utils.folder_setup import folder_setup
from jale.core.utils.hierarchical import hierarchical_clustering_pipeline
from jale.core.utils.kernel import create_kernel_array
from jale.core.utils.kmeans import kmeans_clustering_pipeline
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
    use_pooled_std=False,
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
        correlation_matrix = np.corrcoef(ma_gm_masked)
    else:
        raise ValueError("Invalid correlation_type. Choose 'spearman' or 'pearson'.")

    plot_cor_matrix(
        project_path=project_path,
        meta_name=meta_name,
        correlation_matrix=correlation_matrix,
        correlation_type=correlation_type,
        linkage_method=linkage_method,
    )

    # if no subsampling wanted set iterations to 2; cant be 1 because of later averaging steps
    # this is of course spaghetti code, but for now the easiest solution
    if subsample_fraction == 1:
        sampling_iterations = 2

    if clustering_method == "hierarchical":
        logger.info(f"{meta_name} - running hierarchical clustering")
        hierarchical_clustering_pipeline(
            project_path=project_path,
            exp_df=exp_df,
            meta_name=meta_name,
            kernels=kernels,
            correlation_matrix=correlation_matrix,
            correlation_type=correlation_type,
            linkage_method=linkage_method,
            max_clusters=max_clusters,
            subsample_fraction=subsample_fraction,
            sampling_iterations=sampling_iterations,
            null_iterations=null_iterations,
            use_pooled_std=use_pooled_std,
        )
    elif clustering_method == "kmeans":
        logger.info(f"{meta_name} - kmeans clustering")
        kmeans_clustering_pipeline(
            project_path=project_path,
            exp_df=exp_df,
            meta_name=meta_name,
            kernels=kernels,
            correlation_type=correlation_type,
            correlation_matrix=correlation_matrix,
            max_clusters=max_clusters,
            subsample_fraction=subsample_fraction,
            sampling_iterations=sampling_iterations,
            null_iterations=null_iterations,
            use_pooled_std=use_pooled_std,
        )


def plot_cor_matrix(
    project_path, meta_name, correlation_matrix, correlation_type, linkage_method
):
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
        / f"Results/MA_Clustering/{meta_name}_correlation_matrix_{correlation_type}_{linkage_method}.png"
    )
