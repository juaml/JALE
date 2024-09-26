import numpy as np
import nibabel as nb
from scipy import ndimage
from scipy.stats import norm
from joblib import Parallel, delayed
from utils.tfce_par import tfce_par
from utils.kernel import kernel_conv
from scipy.special import comb
from utils.template import BRAIN_ARRAY_SHAPE, GM_PRIOR, GM_SAMPLE_SPACE, MNI_AFFINE

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
        (len(kernels), BRAIN_ARRAY_SHAPE[0], BRAIN_ARRAY_SHAPE[1], BRAIN_ARRAY_SHAPE[2]))
    for i, kernel in enumerate(kernels):
        ma[i, :] = kernel_conv(foci=foci[i],
                               kernel=kernel)

    return ma


def compute_hx(ma, bin_edges):
    hx = np.zeros((ma.shape[0], len(bin_edges)))
    for i in range(ma.shape[0]):
        data = ma[i, :]
        bin_idxs, counts = np.unique(np.digitize(
            data[GM_PRIOR], bin_edges), return_counts=True)
        hx[i, bin_idxs] = counts
    return hx


def compute_ale(ma):
    return 1-np.prod(1-ma, axis=0)


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
    hx_conv = np.flip(np.cumsum(np.flip(ale_hist[:last_used+1])))

    return hx_conv


def compute_z(ale, hx_conv, step):
    # computing the corresponding histogram bin for each ale value
    ale_step = np.round(ale*step).astype(int)
    # replacing histogram bin number
    # with corresponding histogram value (= p-value)
    p = np.array([hx_conv[i] for i in ale_step])
    p[p < EPS] = EPS
    # calculate z-values by plugging 1-p into a probability density function
    z = norm.ppf(1-p)
    z[z < 0] = 0

    return z


def compute_tfce(z, nprocesses=1):
    delta_t = np.max(z)/100

    tfce = np.zeros(z.shape)
    # calculate tfce values using the parallelized function
    vals, masks = zip(*Parallel(n_jobs=nprocesses)
                      (delayed(tfce_par)(invol=z, h=h, dh=delta_t)
                       for h in np.arange(0, np.max(z), delta_t)))
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
    labels, cluster_count = ndimage.label(sig_arr)

    # Determine the size of the largest cluster (if any clusters exist)
    max_clust = np.max(np.bincount(
        labels[labels > 0])) if cluster_count >= 1 else 0

    # Apply the cluster size cutoff if provided
    if cfwe_threshold:
        significant_clusters = np.bincount(labels[labels > 0]) > cfwe_threshold
        sig_clust_labels = np.where(significant_clusters)[0]
        z = z * np.isin(labels, sig_clust_labels)

    return z, max_clust


def compute_null_ale(num_foci, kernels):
    null_foci = np.array([GM_SAMPLE_SPACE[:, np.random.randint(
        0, GM_SAMPLE_SPACE.shape[1], num_peak)].T for num_peak in num_foci],
        dtype=object)
    null_ma = compute_ma(null_foci, kernels)
    null_ale = compute_ale(null_ma)

    return null_ma, null_ale


def compute_null_cutoffs(num_foci, kernels, step=10000,
                         cluster_forming_threshold=0.001, target_n=None,
                         hx_conv=None, bin_edges=None, bin_centers=None,
                         tfce_enabled=True):
    if target_n:
        s0 = np.random.permutation(np.arange(len(num_foci)))
        s0 = s0[:target_n]
        kernels = kernels.loc[s0]
    # compute ALE values based on random peak locations sampled from a give sample_space
    # sample space could be all grey matter or only foci reported in brainmap
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
        tfce = compute_tfce(null_z)
        # TFCE threshold
        null_max_tfce = np.max(tfce)

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


def compute_sub_ale(sample, ma, hx, bin_centers,
                    cfwe_threshold, step=10000, cluster_forming_threshold=0.001):
    hx_conv = compute_hx_conv(hx, bin_centers, step)
    ale = compute_ale(ma[sample])
    z = compute_z(ale, hx_conv, step)
    z, max_cluster = compute_clusters(
        z, cluster_forming_threshold, cfwe_threshold=cfwe_threshold)
    z[z > 0] = 1
    return z


""" New Contrast Computations """


def compute_ale_diff(s, ma_maps, prior, target_n=None):
    ale = np.zeros((2, prior.sum()))
    for xi in (0, 1):
        if target_n:
            s_perm = np.random.permutation(s[xi])
            s_perm = s_perm[:target_n]
            ale[xi, :] = compute_ale(ma_maps[xi][:, prior][s_perm, :])
        else:
            ale[xi, :] = compute_ale(ma_maps[xi][:, prior])
    r_diff = ale[0, :] - ale[1, :]
    return r_diff


def compute_null_diff(s, prior, exp_dfs, target_n=None, diff_repeats=1000):
    prior_idxs = np.argwhere(prior > 0)
    null_ma = []
    for xi in (0, 1):
        null_foci = np.array([prior_idxs[np.random.randint(
            0, prior_idxs.shape[0], exp_dfs[xi].foci[i]), :] for i in s[xi]], dtype=object)
        null_ma.append(compute_ma(null_foci, exp_dfs[xi].Kernels))

    if target_n:
        p_diff = Parallel(n_jobs=4, verbose=1)(delayed(compute_ale_diff)(
            s, null_ma, prior, target_n) for i in range(diff_repeats))
        p_diff = np.mean(p_diff, axis=0)
    else:
        p_diff = compute_ale_diff(s, null_ma, prior)

    min_diff, max_diff = np.min(p_diff), np.max(p_diff)
    return min_diff, max_diff


""" Legacy Contrast Computations"""


def compute_perm_diff(s, masked_ma):
    # make list with range of values with amount of studies in both experiments together
    sr = np.arange(len(s[0])+len(s[1]))
    sr = np.random.permutation(sr)
    # calculate ale difference for this permutation
    perm_diff = (1-np.prod(masked_ma[sr[:len(s[0])]], axis=0)) - \
        (1-np.prod(masked_ma[sr[len(s[0]):]], axis=0))
    return perm_diff


def compute_sig_diff(fx, mask, ale_diff, perm_diff, null_repeats, diff_thresh):
    n_bigger = [np.sum([diff[i] > ale_diff[i] for diff in perm_diff])
                for i in range(mask.sum())]
    prob_bigger = np.array([x / null_repeats for x in n_bigger])

    z_null = norm.ppf(1-prob_bigger)  # z-value
    z_null[np.logical_and(np.isinf(z_null), z_null > 0)] = norm.ppf(1-EPS)
    # for values where the actual difference is consistently higher
    # than the null distribution the minimum will be z and ->
    z = np.minimum(fx[mask], z_null)
    # will most likely be above the threshold of p = 0.05 or z ~ 1.65
    sig_idxs = np.argwhere(z > norm.ppf(1-diff_thresh)).T
    z = z[sig_idxs]
    sig_idxs = np.argwhere(mask is True)[sig_idxs].squeeze().T
    return z, sig_idxs


""" Plot Utils """


def plot_and_save(arr, img_folder=None, nii_folder=None):
    # Function that takes brain array and transforms it to NIFTI1 format
    # Saves it as a Nifti file
    arr_masked = arr
    arr_masked[GM_PRIOR == 0] = 0
    nii_img = nb.Nifti1Image(arr_masked, MNI_AFFINE)
    nb.save(nii_img, nii_folder)
