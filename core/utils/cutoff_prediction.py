import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.stats import kurtosis, skew


def feature_extraction(nexp, nsub, nfoci):
    """
    Extract statistical features from study metadata for machine learning prediction.

    This function calculates various descriptive statistics (e.g., mean, median,
    standard deviation) for the number of subjects and foci across studies, along with
    certain focus-to-subject ratios, used as features for machine learning models.
    It also checks for out-of-distribution values that could affect model accuracy.

    Parameters
    ----------
    nexp : int
        Number of experiments in the dataset.
    nsub : numpy.ndarray
        Array containing the number of subjects for each experiment.
    nfoci : numpy.ndarray
        Array containing the number of foci for each experiment.

    Returns
    -------
    numpy.ndarray
        Array of extracted feature values for model input.
    """

    # Calculate subject-related statistics
    nsub_total = np.sum(nsub)
    nsub_mean = np.mean(nsub)
    nsub_median = np.median(nsub)
    nsub_std = np.std(nsub)
    nsub_max = np.max(nsub)

    # Check for high maximum subject count to detect potential out-of-distribution data
    if nsub_max > 300:
        print(
            "Dataset features parameters that would lead to"
            " Out-Of-Distribution prediction: Accuracy can not be guaranteed."
            " Please disable cutoff prediction!"
        )
        sys.exit()

    nsub_min = np.min(nsub)
    nsub_skew = skew(nsub)
    nsub_kurtosis = kurtosis(nsub)

    # Calculate focus-related statistics
    nfoci_total = np.sum(nfoci)
    nfoci_mean = np.mean(nfoci)
    nfoci_median = np.median(nfoci)
    nfoci_std = np.std(nfoci)
    nfoci_max = np.max(nfoci)

    # Check for high maximum foci count to detect potential out-of-distribution data
    if nfoci_max > 150:
        print(
            "Dataset features parameters that would lead to"
            " Out-Of-Distribution prediction: Accuracy can not be guaranteed."
            " Please disable cutoff prediction!"
        )

    nfoci_min = np.min(nfoci)
    nfoci_skew = skew(nfoci)
    nfoci_kurtosis = kurtosis(nfoci)

    # Calculate foci-to-subject ratios
    ratio_mean = np.mean(nfoci / nsub)
    ratio_std = np.std(nfoci / nsub)
    ratio_max = np.max(nfoci / nsub)
    ratio_min = np.min(nfoci / nsub)

    # Calculate the mean foci per experiment
    nstudies_foci_ratio = nfoci_total / nexp

    # Sum foci counts by subject size categories
    hi_foci, mi_foci, li_foci, vi_foci = 0, 0, 0, 0
    for i in range(nexp):
        if nsub[i] > 20:
            hi_foci += nfoci[i]
        elif 15 < nsub[i] <= 20:
            mi_foci += nfoci[i]
        elif 10 < nsub[i] <= 15:
            li_foci += nfoci[i]
        elif nsub[i] <= 10:
            vi_foci += nfoci[i]

    # Compile all extracted features into a single array
    x = np.c_[
        nexp,
        nsub_total,
        nsub_mean,
        nsub_median,
        nsub_std,
        nsub_max,
        nsub_min,
        nsub_skew,
        nsub_kurtosis,
        nfoci_total,
        nfoci_mean,
        nfoci_median,
        nfoci_std,
        nfoci_max,
        nfoci_min,
        nfoci_skew,
        nfoci_kurtosis,
        ratio_mean,
        ratio_std,
        ratio_max,
        ratio_min,
        nstudies_foci_ratio,
        hi_foci,
        mi_foci,
        li_foci,
        vi_foci,
    ]

    return x


def predict_cutoff(exp_df):
    """
    Predict statistical cutoff values for multiple comparison corrections using ML models.

    This function loads pre-trained machine learning models to predict cutoffs for
    voxel-wise FWE (vFWE), cluster-wise FWE (cFWE), and TFCE based on extracted dataset
    features. It checks for high experiment counts, which may reduce prediction accuracy.

    Parameters
    ----------
    exp_df : pandas.DataFrame
        DataFrame containing experiment data, including subject counts and number of foci.

    Returns
    -------
    tuple
        - float : Predicted vFWE cutoff.
        - float : Predicted cFWE cutoff (rounded).
        - float : Predicted TFCE cutoff.
    """

    # Load pre-trained models for vFWE, cFWE, and TFCE cutoffs
    module_path = Path(__file__).resolve().parents[2]
    xgb_vfwe = xgb.XGBRegressor()
    xgb_vfwe.load_model(module_path / "assets/ml_models/vFWE_model.txt")

    xgb_cfwe = xgb.XGBRegressor()
    xgb_cfwe.load_model(module_path / "assets/ml_models/cFWE_model.txt")

    xgb_tfce = xgb.XGBRegressor()
    xgb_tfce.load_model(module_path / "assets/ml_models/tfce_model.txt")

    # Calculate the number of experiments
    nexp = exp_df.shape[0]

    # Warn and exit if the number of experiments exceeds the model's training range
    if nexp > 150:
        print(
            "Dataset features parameters that would lead to"
            " Out-Of-Distribution prediction: Accuracy can not be guaranteed."
            " Please disable cutoff prediction!"
        )

    # Extract features from the experimental dataset
    nsub = exp_df.Subjects
    nfoci = exp_df.NumberOfFoci
    features = feature_extraction(nexp, nsub, nfoci)

    # Predict cutoffs using the pre-trained models
    vfwe_cutoff = xgb_vfwe.predict(features)
    cfwe_cutoff = np.round(xgb_cfwe.predict(features))
    tfce_cutoff = xgb_tfce.predict(features)

    return vfwe_cutoff, cfwe_cutoff, tfce_cutoff
