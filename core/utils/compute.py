import nibabel as nb
import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage
from scipy.special import comb
from scipy.stats import norm

from core.utils.kernel import kernel_convolution
from core.utils.template import BRAIN_ARRAY_SHAPE, GM_PRIOR, GM_SAMPLE_SPACE, MNI_AFFINE
from core.utils.tfce_par import tfce_par

EPS = np.finfo(float).eps

""" Main Effect Computations """


def illustrate_foci(foci):
    foci_arr = np.zeros(BRAIN_ARRAY_SHAPE)
    # Load all foci associated with study
    foci = np.concatenate(foci)
    # Set all points in foci_arr that are foci for the study to 1
    foci_arr[tuple(foci.T)] += 1

    return foci_arr


def compute_ma(foci, kernels):
    ma = np.zeros(
        (len(kernels), BRAIN_ARRAY_SHAPE[0], BRAIN_ARRAY_SHAPE[1], BRAIN_ARRAY_SHAPE[2])
    )
    for i, kernel in enumerate(kernels):
        ma[i, :] = kernel_convolution(foci=foci[i], kernel=kernel)

    return ma


def compute_hx(ma, bin_edges):
    hx = np.zeros((ma.shape[0], len(bin_edges)))
    for i in range(ma.shape[0]):
        data = ma[i, :]
        bin_idxs, counts = np.unique(
            np.digitize(data[GM_PRIOR], bin_edges), return_counts=True
        )
        hx[i, bin_idxs] = counts
    return hx


def compute_ale(ma):
    return 1 - np.prod(1 - ma, axis=0)


def compute_hx_conv(hx, bin_centers, step):
    ale_hist = hx[0, :] / np.sum(hx[0, :])

    for x in range(1, hx.shape[0]):
        v1 = ale_hist
        v2 = hx[x, :] / np.sum(hx[x, :])  # normalize study hist

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
    permutation = np.random.permutation(np.arange(ma_merge.shape[0]))
    ale_perm1 = compute_ale(ma_merge[permutation[:nexp]])
    ale_perm2 = compute_ale(ma_merge[permutation[nexp:]])
    permuted_diff = ale_perm1 - ale_perm2

    return permuted_diff


def compute_sig_diff(ale_difference, null_difference, significance_threshold=0.05):
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
    # subALE1
    subsample1 = np.random.choice(np.arange(ma1.shape[0]), target_n, replace=False)
    ale1 = compute_ale(ma1[subsample1, :][:, prior])

    # subALE2
    subsample2 = np.random.choice(np.arange(ma2.shape[0]), target_n, replace=False)
    ale2 = compute_ale(ma2[subsample2, :][:, prior])

    r_diff = ale1 - ale2

    return r_diff


def compute_balanced_null_diff(
    nfoci1, kernels1, nfoci2, kernels2, prior, sampling_reps, target_n
):
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
    for _ in range(sampling_reps):
        null_diff += compute_balanced_ale_diff(null_ma1, null_ma2, prior, target_n)
    null_diff = null_diff / sampling_reps

    min_diff, max_diff = np.min(null_diff), np.max(null_diff)

    return min_diff, max_diff


""" Plot Utils """


def plot_and_save(arr, nii_path):
    # Function that takes brain array and transforms it to NIFTI1 format
    # Saves it as a Nifti file
    arr_masked = arr
    arr_masked[GM_PRIOR == 0] = 0
    nii_img = nb.nifti1.Nifti1Image(arr_masked, MNI_AFFINE)
    nb.loadsave.save(nii_img, nii_path)
