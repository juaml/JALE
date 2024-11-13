import logging

import matplotlib.pyplot as plt
import numpy as np

from jale.core.utils.compute import compute_ale, compute_ma
from jale.core.utils.template import GM_SAMPLE_SPACE

logger = logging.getLogger("ale_logger")


def roi_ale(project_path, exp_df, meta_name, mask, mask_name, monte_carlo_iterations):
    # Stack the modeled activation (MA) values and compute ALE values
    ma = np.load(project_path / f"MainEffect/{meta_name}_ma.npz")["arr_0"]
    ale = compute_ale(ma)

    # Calculate benchmark values within the mask
    mask_sum = np.sum(ale[mask])
    mask_max = np.max(ale[mask])

    # Initialize null distribution array for masked ALE values
    mask_size = np.sum(mask > 0)
    null_ale_mask = np.zeros((monte_carlo_iterations, mask_size))
    num_peaks = exp_df.NumberOfFoci.values

    # Simulate null distribution with repeated random sampling
    for repeat in range(monte_carlo_iterations):
        if repeat > 0 and repeat % 1000 == 0:
            logger.info(f"Simulated {repeat} iterations of ROI null distribution")

        # Generate random peaks and compute null ALE values
        null_peaks = np.array(
            [
                GM_SAMPLE_SPACE[
                    :, np.random.randint(0, GM_SAMPLE_SPACE.shape[1], num_peak)
                ].T
                for num_peak in num_peaks
            ],
            dtype=object,
        )

        null_ma = compute_ma(null_peaks, exp_df.Kernels)
        null_ale = compute_ale(null_ma)
        null_ale_mask[repeat] = null_ale[mask]

    # Aggregate null distribution statistics
    null_sum = np.sum(null_ale_mask, axis=1)
    null_max = np.max(null_ale_mask, axis=1)

    # Plot the results
    plot_roi_ale(
        project_path,
        meta_name,
        mask_name,
        mask_sum,
        mask_max,
        null_sum,
        null_max,
        monte_carlo_iterations,
    )


def plot_roi_ale(
    project_path,
    meta_name,
    mask_name,
    bench_sum,
    bench_max,
    null_sum,
    null_max,
    monte_carlo_iterations,
):
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    fig.patch.set_facecolor("skyblue")

    def plot_histogram(axis, data, bench_value, title, xlabel):
        weights = np.ones_like(data) / len(data)
        n, _, patches = axis.hist(
            data, bins=np.linspace(0, np.ceil(np.max(data)), 50), weights=weights
        )

        # Plot benchmark line and annotate p-value
        axis.vlines(bench_value, 0, np.max(n) + np.max(n) / 8, colors="r")
        p_value = (data > bench_value).sum() / monte_carlo_iterations
        axis.annotate(
            f"Observed value\np-value = {p_value:.3f}",
            xy=(bench_value, np.max(n) + np.max(n) / 8),
            ha="center",
        )

        # Plot significance lines and percentiles
        percentiles = [
            (95, "darkgreen", "p < 0.05"),
            (99, "green", "p < 0.01"),
            (99.9, "lime", "p < 0.001"),
        ]

        for perc, color, label in percentiles:
            axis.vlines(
                np.percentile(data, perc), 0, np.max(n), colors=color, label=label
            )

        axis.set_xlabel(xlabel)
        axis.set_ylabel(f"Percentage of {monte_carlo_iterations} realizations")
        axis.title.set_text(title)
        axis.legend(loc="upper right")

    # Plot ALE integral histogram
    plot_histogram(
        ax[0],
        null_sum,
        bench_sum,
        f"{mask_name} - ALE integral",
        "ALE integral in ROI mask",
    )

    # Plot max ALE histogram
    ax[1].yaxis.tick_right()
    plot_histogram(
        ax[1],
        null_max,
        bench_max,
        f"{mask_name} - max ALE",
        "Max ALE in ROI mask",
    )

    fig.tight_layout()

    save_path = project_path / f"Results/MainEffect/ROI/{meta_name}_{mask_name}.png"
    fig.savefig(save_path)
    plt.close()
