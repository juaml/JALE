import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage
from scipy.special import comb
from scipy.stats import norm

from jale.core.utils.kernel import kernel_convolution
from jale.core.utils.template import BRAIN_ARRAY_SHAPE, GM_PRIOR, GM_SAMPLE_SPACE
from jale.core.utils.tfce_par import tfce_par

EPS = np.finfo(float).eps

""" Main Effect Computations """


def illustrate_foci(foci):
    """
    Create a brain array marking the locations of foci.

    Parameters
    ----------
    foci : list of arrays
        Coordinates of foci across experiments.

    Returns
    -------
    numpy.ndarray
        3D brain array with foci locations marked.
    """

    foci_arr = np.zeros(BRAIN_ARRAY_SHAPE)
    # Load all foci associated with study
    foci = np.concatenate(foci)
    # Set all points in foci_arr that are foci for the study to 1
    foci_arr[tuple(foci.T)] += 1

    return foci_arr


def compute_ma(foci, kernels):
    """
    Compute modeled activation (MA) maps by convolving foci with kernels.

    Parameters
    ----------
    foci : list of arrays
        Coordinates of foci for each study.
    kernels : list of numpy.ndarray
        Smoothing kernels for each study.

    Returns
    -------
    numpy.ndarray
        Modeled activation maps (4D array).
    """

    ma = np.zeros(
        (len(kernels), BRAIN_ARRAY_SHAPE[0], BRAIN_ARRAY_SHAPE[1], BRAIN_ARRAY_SHAPE[2])
    )
    for i, kernel in enumerate(kernels):
        ma[i, :] = kernel_convolution(foci=foci[i], kernel=kernel)

    return ma


def compute_hx(ma, bin_edges):
    """
    Calculate histograms for modeled activation (MA) maps.

    Parameters
    ----------
    ma : numpy.ndarray
        Modeled activation maps.
    bin_edges : numpy.ndarray
        Edges for histogram bins.

    Returns
    -------
    numpy.ndarray
        Histogram values for each map.
    """

    hx = np.zeros((ma.shape[0], len(bin_edges)))
    for i in range(ma.shape[0]):
        data = ma[i, :]
        bin_idxs, counts = np.unique(
            np.digitize(data[GM_PRIOR], bin_edges), return_counts=True
        )
        hx[i, bin_idxs] = counts
    return hx


def compute_ale(ma):
    """
    Compute ALE (Activation Likelihood Estimation) values from modeled activation (MA) maps.

    Parameters
    ----------
    ma : numpy.ndarray
        Modeled activation maps.

    Returns
    -------
    numpy.ndarray
        ALE values across all voxels.
    """

    return 1 - np.prod(1 - ma, axis=0)


def compute_hx_conv(hx, bin_centers, step):
    """
    Convolve histograms for ALE threshold estimation from modeled activation maps.

    Parameters
    ----------
    hx : numpy.ndarray
        Histogram values from modeled activation maps.
    bin_centers : numpy.ndarray
        Centers of histogram bins.
    step : int
        Step size for histogram binning.

    Returns
    -------
    numpy.ndarray
        Convolved histogram for ALE estimation.
    """

    ale_hist = hx[0, :] / np.sum(hx[0, :])

    for x in range(1, hx.shape[0]):
        v1 = ale_hist
        # normalize hist
        v2 = hx[x, :] / np.sum(hx[x, :])

        # Get indices of non-zero bins
        da1, da2 = np.where(v1 > 0)[0], np.where(v2 > 0)[0]

        # Compute outer products for probabilities and scores
        p = np.outer(v2[da2], v1[da1])
        score = 1 - (1 - bin_centers[da2])[:, None] * (1 - bin_centers[da1])

        ale_bin = np.round(score * step).astype(int)
        ale_hist = np.zeros(len(bin_centers))

        # Add probabilities to respective bins
        np.add.at(ale_hist, ale_bin, p)

    # Compute cumulative sum up to the last non-zero bin
    last_used = np.max(np.where(ale_hist > 0)[0])
    hx_conv = np.flip(np.cumsum(np.flip(ale_hist[: last_used + 1])))

    return hx_conv


def compute_z(ale, hx_conv, step):
    """
    Calculate z-values from ALE values and a convolved histogram.

    This function computes z-values by determining the p-values corresponding to
    ALE values and then applying the inverse of the normal cumulative distribution function.

    Parameters
    ----------
    ale : numpy.ndarray
        Array of ALE values.
    hx_conv : numpy.ndarray
        Convolved histogram values for ALE thresholding.
    step : int
        Step size for histogram binning.

    Returns
    -------
    numpy.ndarray
        Array of z-values derived from ALE values.
    """

    # computing the corresponding histogram bin for each ale value
    ale_step = np.round(ale * step).astype(int)
    # replacing histogram bin number
    # with corresponding histogram value (= p-value)
    p = np.array([hx_conv[i] for i in ale_step])
    p[p < EPS] = EPS
    # calculate z-values by plugging 1-p into a probability density function
    z = norm.ppf(1 - p)
    z[z < 0] = 0

    return z


def compute_tfce(z, nprocesses=1):
    """
    Calculate TFCE (Threshold-Free Cluster Enhancement) values for a given z-map.

    Uses a parallelized method to compute TFCE scores based on the intensity and
    extent of clusters in the z-map.

    Parameters
    ----------
    z : numpy.ndarray
        Array of z-values for thresholding.
    nprocesses : int, optional
        Number of parallel processes to use, by default 1.

    Returns
    -------
    numpy.ndarray
        Array of TFCE values.
    """

    delta_t = np.max(z) / 100

    tfce = np.zeros(z.shape)
    # calculate tfce values using the parallelized function
    vals, masks = zip(
        *Parallel(n_jobs=nprocesses)(
            delayed(tfce_par)(invol=z, h=h, dh=delta_t)
            for h in np.arange(0, np.max(z), delta_t)
        )
    )
    # Parallelization makes it necessary to integrate the results afterwards
    # Each repition creats it's own mask and
    # an amount of values corresponding to that mask
    for i in range(len(vals)):
        tfce[masks[i]] += vals[i]

    return tfce


def compute_clusters(z, cluster_forming_threshold, cfwe_threshold=None):
    """
    Identify significant clusters in a z-map based on a cluster-forming threshold.

    This function applies thresholding to identify clusters of significant z-values
    and, optionally, filters clusters based on a cluster-wise family-wise error
    (cFWE) threshold.

    Parameters
    ----------
    z : numpy.ndarray
        Array of z-values.
    cluster_forming_threshold : float
        Threshold for forming clusters in the z-map.
    cfwe_threshold : int, optional
        Minimum cluster size for cFWE correction, by default None.

    Returns
    -------
    tuple
        - numpy.ndarray : Thresholded z-map with significant clusters.
        - int : Size of the largest cluster found.
    """

    # Threshold z-values based on the specified cluster threshold
    sig_arr = (z > norm.ppf(1 - cluster_forming_threshold)).astype(int)

    # Find clusters of significant z-values
    labels, cluster_count = ndimage.label(sig_arr)  # type: ignore

    # Determine the size of the largest cluster (if any clusters exist)
    max_clust = np.max(np.bincount(labels[labels > 0])) if cluster_count >= 1 else 0

    # Apply the cluster size cutoff if provided
    if cfwe_threshold:
        significant_clusters = np.bincount(labels[labels > 0]) > cfwe_threshold
        sig_clust_labels = np.where(significant_clusters)[0]
        z = z * np.isin(labels, sig_clust_labels)

    return z, max_clust


def compute_null_ale(num_foci, kernels):
    """
    Generate a null ALE map by randomly assigning foci and computing ALE.

    Creates a null activation map (ALE) by assigning foci locations randomly within
    the sample space and calculating the ALE based on the provided kernels.

    Parameters
    ----------
    num_foci : list of int
        List of foci counts for each experiment.
    kernels : list of numpy.ndarray
        Smoothing kernels for each experiment.

    Returns
    -------
    tuple
        - numpy.ndarray : Null modeled activation maps (MA).
        - numpy.ndarray : Null ALE map.
    """

    null_foci = [
        GM_SAMPLE_SPACE[:, np.random.randint(0, GM_SAMPLE_SPACE.shape[1], num_focus)].T
        for num_focus in num_foci
    ]
    null_ma = compute_ma(null_foci, kernels)
    null_ale = compute_ale(null_ma)

    return null_ma, null_ale


def compute_monte_carlo_null(
    num_foci,
    kernels,
    bin_edges=None,
    bin_centers=None,
    step=10000,
    cluster_forming_threshold=0.001,
    target_n=None,
    hx_conv=None,
    tfce_enabled=True,
):
    """
    Generate a null distribution for ALE using Monte Carlo sampling.

    This function computes vFWE, cFWE and TFCE thresholds based on
    randomly sampled foci.

    Parameters
    ----------
    num_foci : list of int
        Number of foci for each study.
    kernels : list of numpy.ndarray
        Smoothing kernels for each study.
    bin_edges : numpy.ndarray, optional
        Edges for histogram bins, by default None.
    bin_centers : numpy.ndarray, optional
        Centers of histogram bins, by default None.
    step : int, optional
        Step size for histogram binning, by default 10000.
    cluster_forming_threshold : float, optional
        Threshold for forming clusters in ALE, by default 0.001.
    target_n : int, optional
        Specifies if the full dataset is used or a subsample, by default None.
    hx_conv : numpy.ndarray, optional
        Convolved histogram for ALE thresholding, by default None.
    tfce_enabled : bool, optional
        Whether to compute TFCE thresholds, by default True.

    Returns
    -------
    tuple
        - float : Maximum null ALE value.
        - int : Maximum null cluster size.
        - float : Maximum null TFCE value.
    """

    if target_n:
        subsample = np.random.permutation(np.arange(len(num_foci)))
        subsample = subsample[:target_n]
        num_foci = num_foci[subsample]
        kernels = kernels[subsample]
    # compute ALE values based on random peak locations sampled from grey matter
    null_ma, null_ale = compute_null_ale(num_foci, kernels)
    # Peak ALE threshold
    null_max_ale = np.max(null_ale)
    if hx_conv is None:
        null_hx = compute_hx(null_ma, bin_edges)
        hx_conv = compute_hx_conv(null_hx, bin_centers, step)
    null_z = compute_z(null_ale, hx_conv, step)
    # Cluster level threshold
    _, null_max_cluster = compute_clusters(null_z, cluster_forming_threshold)
    null_max_tfce = 0
    if tfce_enabled:
        null_tfce = compute_tfce(null_z)
        # TFCE threshold
        null_max_tfce = np.max(null_tfce)

    return null_max_ale, null_max_cluster, null_max_tfce


""" CV/Subsampling ALE Computations """


def generate_unique_subsamples(total_n, target_n, sample_n):
    """
    Generate unique random subsamples of indices for probabilistic ALE.

    Parameters
    ----------
    total_n : int
        Total number of studies or experiments.
    target_n : int
        Target number of studies for each subsample.
    sample_n : int
        Desired number of unique subsamples to generate.

    Returns
    -------
    list of numpy.ndarray
        List of unique subsample arrays.
    """

    # Calculate the maximum number of unique subsamples (combinations)
    max_combinations = int(comb(total_n, target_n, exact=True))

    # If sample_n exceeds max_combinations, limit it
    if sample_n > max_combinations:
        sample_n = max_combinations

    subsamples = set()

    while len(subsamples) < sample_n:
        # Generate a random subsample of size `target_n` from `total_n`
        subsample = np.sort(np.random.choice(total_n, target_n, replace=False))

        # Add the tuple version of the sorted subsample to ensure uniqueness
        subsamples.add(tuple(subsample))

    # Convert the set of tuples back to a list of NumPy arrays
    return [np.array(subsample) for subsample in subsamples]


def compute_sub_ale_single(
    ma,
    cfwe_threshold,
    bin_edges,
    bin_centers,
    step=10000,
    cluster_forming_threshold=0.001,
):
    """
    Compute a single subsample ALE map with cFWE thresholding.

    This function calculates a thresholded ALE map from modeled activation maps
    for a single subsample, applying voxel-wise and cluster-wise corrections.

    Parameters
    ----------
    ma : numpy.ndarray
        Modeled activation maps for the subsample.
    cfwe_threshold : float
        Threshold for cluster-wise FWE correction.
    bin_edges : numpy.ndarray
        Edges for histogram bins.
    bin_centers : numpy.ndarray
        Centers of histogram bins.
    step : int, optional
        Step size for histogram binning, by default 10000.
    cluster_forming_threshold : float, optional
        Threshold for forming clusters in ALE, by default 0.001.

    Returns
    -------
    numpy.ndarray
        Thresholded ALE map for the subsample.
    """

    hx = compute_hx(ma, bin_edges)
    hx_conv = compute_hx_conv(hx, bin_centers, step)
    ale = compute_ale(ma)
    z = compute_z(ale, hx_conv, step)
    z, _ = compute_clusters(z, cluster_forming_threshold, cfwe_threshold=cfwe_threshold)
    z[z > 0] = 1
    return z


def compute_sub_ale(
    samples,
    ma,
    cfwe_threshold,
    bin_edges,
    bin_centers,
    step=10000,
    cluster_forming_threshold=0.001,
):
    """
    Compute the mean ALE map from multiple subsamples.

    Aggregates thresholded ALE maps from a set of subsamples to produce a mean
    ALE map, applying voxel-wise and cluster-wise corrections.

    Parameters
    ----------
    samples : list of numpy.ndarray
        List of subsamples (each a subset of indices).
    ma : numpy.ndarray
        Modeled activation maps for all studies.
    cfwe_threshold : float
        Threshold for cluster-wise FWE correction.
    bin_edges : numpy.ndarray
        Edges for histogram bins.
    bin_centers : numpy.ndarray
        Centers of histogram bins.
    step : int, optional
        Step size for histogram binning, by default 10000.
    cluster_forming_threshold : float, optional
        Threshold for forming clusters in ALE, by default 0.001.

    Returns
    -------
    numpy.ndarray
        Mean ALE map across all subsamples.
    """

    ale_mean = np.zeros(BRAIN_ARRAY_SHAPE)
    for idx, sample in enumerate(samples):
        if idx % 500 == 0:
            print(f"Calculated {idx} subsample ALEs")
        ale_mean += compute_sub_ale_single(
            ma[sample],
            cfwe_threshold,
            bin_edges,
            bin_centers,
            step,
            cluster_forming_threshold,
        )
    return ale_mean / len(samples)


""" Legacy Contrast Computations"""


def compute_permuted_ale_diff(ma_merge, nexp):
    """
    Calculate the difference in ALE maps based on a random permutation of studies.

    Randomly splits the combined modeled activation maps into two groups and computes
    the ALE difference between these groups.

    Parameters
    ----------
    ma_merge : numpy.ndarray
        Combined modeled activation maps from two groups.
    nexp : int
        Number of experiments in the first group.

    Returns
    -------
    numpy.ndarray
        Permuted difference in ALE values between the two groups.
    """

    permutation = np.random.permutation(np.arange(ma_merge.shape[0]))
    ale_perm1 = compute_ale(ma_merge[permutation[:nexp]])
    ale_perm2 = compute_ale(ma_merge[permutation[nexp:]])
    permuted_diff = ale_perm1 - ale_perm2

    return permuted_diff


def compute_sig_diff(ale_difference, null_difference, significance_threshold=0.05):
    """
    Identify significant differences in ALE values compared to a null distribution.

    This function computes z-scores for differences in ALE values and determines
    significant differences based on a specified threshold.

    Parameters
    ----------
    ale_difference : numpy.ndarray
        Observed ALE difference values.
    null_difference : numpy.ndarray
        Null distribution of ALE differences.
    significance_threshold : float, optional
        Significance level for identifying significant differences, by default 0.05.

    Returns
    -------
    tuple
        - numpy.ndarray : z-scores of significant ALE differences.
        - numpy.ndarray : Indices of significant differences.
    """

    p_diff = np.average((null_difference > ale_difference), axis=0)
    EPS = np.finfo(float).eps
    p_diff[p_diff < EPS] = EPS
    z_diff = norm.ppf(1 - p_diff)
    z_threshold = norm.ppf(1 - significance_threshold)

    if np.max(z_diff) < z_threshold:
        z_diff = 0
        sig_diff_idxs = 0
    else:
        sig_diff_idxs = np.argwhere(z_diff > z_threshold)
        z_diff = z_diff[sig_diff_idxs]

    return z_diff, sig_diff_idxs


""" Balanced Contrast Computations"""


def compute_balanced_ale_diff(ma1, ma2, prior, target_n):
    """
    Compute the ALE difference for balanced subsamples from two groups.

    This function randomly subsamples studies from two groups to calculate a balanced
    ALE difference.

    Parameters
    ----------
    ma1 : numpy.ndarray
        Modeled activation maps for the first group.
    ma2 : numpy.ndarray
        Modeled activation maps for the second group.
    prior : numpy.ndarray
        Mask indicating relevant brain regions (e.g., grey matter).
    target_n : int
        Number of studies to include in each subsample.

    Returns
    -------
    numpy.ndarray
        Difference in ALE values between the two subsamples.
    """

    # subALE1
    subsample1 = np.random.choice(np.arange(ma1.shape[0]), target_n, replace=False)
    ale1 = compute_ale(ma1[subsample1, :][:, prior])

    # subALE2
    subsample2 = np.random.choice(np.arange(ma2.shape[0]), target_n, replace=False)
    ale2 = compute_ale(ma2[subsample2, :][:, prior])

    r_diff = ale1 - ale2

    return r_diff


def compute_balanced_null_diff(
    nfoci1, kernels1, nfoci2, kernels2, prior, target_n, difference_iterations
):
    """
    Generate a null distribution for balanced ALE differences.

    Creates a null distribution of ALE differences between two groups by randomly
    assigning foci locations and computing the ALE difference over multiple iterations.

    Parameters
    ----------
    nfoci1 : list of int
        Number of foci for each study in the first group.
    kernels1 : list of numpy.ndarray
        Smoothing kernels for each study in the first group.
    nfoci2 : list of int
        Number of foci for each study in the second group.
    kernels2 : list of numpy.ndarray
        Smoothing kernels for each study in the second group.
    prior : numpy.ndarray
        Mask indicating relevant brain regions (e.g., grey matter).
    target_n : int
        Number of studies to include in each subsample.
    difference_iterations : int
        Number of iterations to compute the null distribution.

    Returns
    -------
    tuple
        - float : Minimum ALE difference in the null distribution.
        - float : Maximum ALE difference in the null distribution.
    """

    null_foci1 = [
        GM_SAMPLE_SPACE[:, np.random.randint(0, GM_SAMPLE_SPACE.shape[1], nfoci)].T
        for nfoci in nfoci1
    ]
    null_ma1 = compute_ma(null_foci1, kernels1)

    null_foci2 = [
        GM_SAMPLE_SPACE[:, np.random.randint(0, GM_SAMPLE_SPACE.shape[1], nfoci)].T
        for nfoci in nfoci2
    ]
    null_ma2 = compute_ma(null_foci2, kernels2)

    null_diff = np.zeros((np.sum(prior),))
    for _ in range(difference_iterations):
        null_diff += compute_balanced_ale_diff(null_ma1, null_ma2, prior, target_n)
    null_diff = null_diff / difference_iterations

    min_diff, max_diff = np.min(null_diff), np.max(null_diff)

    return min_diff, max_diff
