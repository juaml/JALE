import numpy as np
from nibabel import loadsave
from scipy import ndimage

from jale.core.utils.compute import compute_ale, compute_ma
from jale.core.utils.kernel import create_kernel_array
from jale.core.utils.template import MNI_AFFINE


def contribution(project_path, exp_df, exp_name, tasks, tfce_enabled=True):
    """
    Analyze contributions of individual studies and tasks to significant clusters in a meta-analysis.

    Parameters
    ----------
    project_path : str or Path
        Path to the project directory.
    exp_df : pandas.DataFrame
        DataFrame containing details of individual experiments, including their coordinates.
    exp_name : str
        Name of the experiment or meta-analysis for file naming.
    tasks : pandas.DataFrame
        DataFrame containing task information, including names and experiment indices.
    tfce_enabled : bool, optional
        If True, includes TFCE correction in the analysis, by default True.

    Returns
    -------
    None
        The function saves results to a text file in the specified `project_path`.
    """
    # Create an array of experiment indices and generate MA maps and smoothing kernels
    exp_idxs = np.arange(exp_df.shape[0])
    kernels = create_kernel_array(exp_df)
    ma = compute_ma(exp_df.Coordinates, kernels)

    # Determine the correction methods based on `tfce_enabled` flag
    corr_methods = ["TFCE", "vFWE", "cFWE"] if tfce_enabled else ["vFWE", "cFWE"]

    # Loop through each correction method to analyze contributions
    for corr_method in corr_methods:
        output_path = (
            project_path
            / f"Results/MainEffect/Full/Contribution/{exp_name}_{corr_method}.txt"
        )
        with open(output_path, "w") as txt:
            write_header(
                txt, exp_name, exp_df
            )  # Write the initial information to the text file

            # Load the ALE results corrected by the current method
            results = load_corrected_results(project_path, exp_name, corr_method)
            if results.any() > 0:  # Proceed only if there are significant results
                labels, cluster_counts = ndimage.label(results)  # type: ignore
                clusters = get_clusters(
                    labels, min_size=5
                )  # Identify clusters with at least 5 voxels

                # Analyze each identified cluster
                for idx, cluster_info in enumerate(clusters):
                    cluster_idxs, center = cluster_info
                    write_cluster_info(
                        txt, idx, cluster_idxs, center
                    )  # Write cluster details

                    # Mask the MA map to the current cluster and calculate contributions
                    ma_cluster_mask = ma[
                        :, cluster_idxs[0], cluster_idxs[1], cluster_idxs[2]
                    ]
                    ale_cluster_mask = compute_ale(ma_cluster_mask)
                    contribution_arr = calculate_contributions(
                        exp_idxs, ma_cluster_mask, ale_cluster_mask
                    )

                    # Write the contributions of individual experiments and tasks to the text file
                    write_experiment_contributions(
                        txt, exp_df, exp_idxs, contribution_arr
                    )
                    write_task_contributions(txt, tasks, exp_idxs, contribution_arr)

            else:
                # Write a message if no significant clusters are found
                txt.write(f"No significant clusters in {corr_method}!\n")


def write_header(txt, exp_name, exp_df):
    """
    Write the header information for the contribution analysis file.

    Parameters
    ----------
    txt : file object
        Opened file object for writing results.
    exp_name : str
        Name of the experiment or meta-analysis.
    exp_df : pandas.DataFrame
        DataFrame containing experiment details.
    """

    txt.write(f"\nStarting with {exp_name}!\n")
    txt.write(
        f"\n{exp_name}: {len(exp_df)} experiments; {exp_df.Subjects.sum()} unique subjects "
        f"(average of {exp_df.Subjects.mean():.1f} per experiment)\n"
    )


def load_corrected_results(project_path, exp_name, corr_method):
    """
    Load corrected ALE results for a specific correction method.

    Parameters
    ----------
    project_path : str or Path
        Path to the project directory.
    exp_name : str
        Name of the experiment or meta-analysis.
    corr_method : str
        Correction method used (e.g., "TFCE", "vFWE", or "cFWE").

    Returns
    -------
    numpy.ndarray
        Array containing corrected ALE results.
    """

    file_path = (
        project_path / f"Results/MainEffect/Full/Volumes/{exp_name}_{corr_method}05.nii"
    )
    results = loadsave.load(file_path).get_fdata()  # type: ignore
    return np.nan_to_num(results)  # Replace NaNs with zeros for further processing


def get_clusters(labels, min_size=5):
    """
    Identify clusters from labeled regions and filter by size.

    Parameters
    ----------
    labels : numpy.ndarray
        Labeled array where each distinct region has a unique label.
    min_size : int, optional
        Minimum cluster size (in voxels), by default 5.

    Returns
    -------
    list of tuple
        List of clusters, each represented by (indices, center coordinates).
    """

    clusters = []
    label_ids, counts = np.unique(labels, return_counts=True)
    for label, count in zip(label_ids[1:], counts[1:]):  # Skip background label
        if count >= min_size:
            cluster_idxs = np.where(labels == label)
            center = compute_cluster_center(cluster_idxs)
            clusters.append((cluster_idxs, center))
    return clusters


def compute_cluster_center(cluster_idxs):
    """
    Compute the center coordinates of a cluster.

    Parameters
    ----------
    cluster_idxs : tuple of arrays
        Indices of the cluster voxels.

    Returns
    -------
    numpy.ndarray
        Center coordinates of the cluster in MNI space.
    """

    return np.median(
        np.dot(MNI_AFFINE, np.vstack([cluster_idxs, np.ones(cluster_idxs[0].size)])),
        axis=1,
    )


def calculate_contributions(exp_idxs, ma_cluster_mask, ale_cluster_mask):
    """
    Calculate the contribution of each experiment to a cluster.

    Parameters
    ----------
    exp_idxs : numpy.ndarray
        Array of experiment indices.
    ma_cluster_mask : numpy.ndarray
        Masked modeled activation maps for the current cluster.
    ale_cluster_mask : numpy.ndarray
        ALE values for the current cluster.

    Returns
    -------
    numpy.ndarray
        Array of contribution values for each experiment.
    """

    contribution_arr = np.zeros((len(exp_idxs), 4))
    for idx in exp_idxs:
        contribution_arr[idx, 0] = np.sum(ma_cluster_mask[idx])  # Sum of activations
        contribution_arr[idx, 1] = (
            100 * contribution_arr[idx, 0] / ma_cluster_mask.shape[1]
        )  # % contribution
        proportion_of_ale = (
            compute_ale(np.delete(ma_cluster_mask, idx, axis=0)) / ale_cluster_mask
        )
        contribution_arr[idx, 2] = 100 * (
            1 - np.mean(proportion_of_ale)
        )  # Mean contribution
        contribution_arr[idx, 3] = 100 * (
            1 - np.min(proportion_of_ale)
        )  # Max contribution
    contribution_arr[:, 2] = (
        contribution_arr[:, 2] / np.sum(contribution_arr[:, 2])
    ) * 100  # Normalize
    return contribution_arr


def write_cluster_info(txt, index, cluster_idxs, center):
    """
    Write information about a significant cluster to the file.

    Parameters
    ----------
    txt : file object
        Opened file object for writing results.
    index : int
        Cluster index.
    cluster_idxs : tuple of arrays
        Indices of the cluster voxels.
    center : numpy.ndarray
        Center coordinates of the cluster.
    """

    txt.write(
        f"\n\nCluster {index + 1}: {cluster_idxs[0].size} voxels "
        f"[Center: {int(center[0])}/{int(center[1])}/{int(center[2])}]\n"
    )


def write_experiment_contributions(txt, exp_df, exp_idxs, contribution_arr):
    """
    Write the contribution of each experiment to the current cluster.

    Parameters
    ----------
    txt : file object
        Opened file object for writing results.
    exp_df : pandas.DataFrame
        DataFrame containing experiment details.
    exp_idxs : numpy.ndarray
        Array of experiment indices.
    contribution_arr : numpy.ndarray
        Array of contribution values for each experiment.
    """

    sorted_idxs = np.argsort(contribution_arr[:, 2])[::-1]
    for idx in sorted_idxs:
        if (
            contribution_arr[idx, 2] > 0.1 or contribution_arr[idx, 3] > 5
        ):  # Filter significant contributions
            article_name = exp_df.Articles[idx].ljust(
                exp_df.Articles.str.len().max() + 2
            )
            txt.write(
                f"{article_name}\t{contribution_arr[idx,0]:.3f}\t"
                f"{contribution_arr[idx,1]:.3f}\t{contribution_arr[idx,2]:.2f}\t"
                f"{contribution_arr[idx,3]:.2f}\t({exp_df.Subjects[idx]})\n"
            )


def write_task_contributions(txt, tasks, exp_idxs, contribution_arr):
    """
    Write the contribution of each task to the current cluster.

    Parameters
    ----------
    txt : file object
        Opened file object for writing results.
    tasks : pandas.DataFrame
        DataFrame containing task information.
    exp_idxs : numpy.ndarray
        Array of experiment indices.
    contribution_arr : numpy.ndarray
        Array of contribution values for each experiment.
    """

    for i, task_name in enumerate(tasks.Name):
        mask = [s in tasks.ExpIndex[i] for s in exp_idxs]
        if any(mask):  # Only include tasks with contributions
            task_contribution = np.sum(contribution_arr[mask], axis=0)
            if task_contribution[0] > 0.01:  # Filter low contributions
                txt.write(
                    f"{task_name.ljust(tasks.Name.str.len().max())}\t"
                    f"{task_contribution[0]:.3f}\t{task_contribution[1]:.3f}\t"
                    f"{task_contribution[2]:.2f}\n"
                )
