import logging
from pathlib import Path

from jale.core.analyses.balanced_contrast import balanced_contrast
from jale.core.analyses.clustering import clustering
from jale.core.analyses.contrast import contrast
from jale.core.analyses.main_effect import main_effect
from jale.core.analyses.probabilistic import probabilistic_ale
from jale.core.analyses.roi import roi_ale
from jale.core.utils.compile_experiments import compile_experiments
from jale.core.utils.contribution import contribution
from jale.core.utils.input import (
    check_for_exp_independence,
    check_params,
    determine_target_n,
    load_config,
    load_dataframes,
    setup_contrast_data,
)
from jale.core.utils.logger import setup_logger


def run_ale(yaml_path=None):
    # Load config and set up paths
    config = load_config(yaml_path)
    project_path = Path(yaml_path).parent
    # Create a logs folder (common across all analysis types)
    (project_path / "logs").mkdir(parents=True, exist_ok=True)
    # Initialize the logger
    logger = setup_logger(project_path)
    logger.info("Logger initialized and project setup complete.")

    params = config.get("parameters", {})
    params = check_params(params)
    clustering_params = config.get("clustering_parameters", {})
    exp_all_df, tasks, analysis_df = load_dataframes(project_path, config)

    # Main loop to process each row in the analysis dataframe
    for row_idx in range(analysis_df.shape[0]):
        # skip empty rows - indicate 2nd effect for contrast analysis
        if not isinstance(analysis_df.iloc[row_idx, 0], str):
            continue

        if analysis_df.iloc[row_idx, 0] == "M":
            run_main_effect(
                analysis_df, row_idx, project_path, params, exp_all_df, tasks
            )
        elif analysis_df.iloc[row_idx, 0][0] == "P":
            run_probabilistic_ale(
                analysis_df, row_idx, project_path, params, exp_all_df, tasks
            )
        elif analysis_df.iloc[row_idx, 0] == "C":
            run_contrast_analysis(
                analysis_df, row_idx, project_path, params, exp_all_df, tasks
            )
        elif analysis_df.iloc[row_idx, 0][0] == "B":
            run_balanced_contrast(
                analysis_df, row_idx, project_path, params, exp_all_df, tasks
            )
        elif analysis_df.iloc[row_idx, 0] == "Cluster":
            run_ma_clustering(
                analysis_df, row_idx, project_path, clustering_params, exp_all_df, tasks
            )

    logger.info("Analysis completed.")


def run_main_effect(analysis_df, row_idx, project_path, params, exp_all_df, tasks):
    """
    Run a main-effect analysis based on the analysis dataframe and experiment info.

    Parameters
    ----------
    analysis_df : pandas.DataFrame
        DataFrame containing the analysis information, including meta-analysis names
        and conditions for experiment selection.
    row_idx : int
        Index of the current row in the analysis dataframe.
    project_path : str or Path
        Path to the project directory where results are saved.
    params : dict
        Dictionary of parameters for analysis, including Monte Carlo iterations and
        subsample size.
    exp_all_df : pandas.DataFrame
        DataFrame containing all available experimental data.
    tasks : pandas.DataFrame
        DataFrame containing task information used for compiling experiments.

    Returns
    -------
    None
        The function performs computations and saves the results.
    """
    logger = logging.getLogger("ale_logger")
    meta_name = analysis_df.iloc[row_idx, 1]

    result_path = project_path / f"Results/MainEffect/Volumes/{meta_name}_cFWE.nii"
    if result_path.exists():
        logger.info(f"Main Effect results for {meta_name} already exist.")
        return

    logger.info("Running Main-Effect Analysis")
    conditions = analysis_df.iloc[row_idx, 2:].dropna().to_list()
    exp_idxs, masks, mask_names = compile_experiments(conditions, tasks)
    exp_df = exp_all_df.loc[exp_idxs].reset_index(drop=True)

    check_for_exp_independence(exp_df)

    main_effect(
        project_path,
        exp_df,
        meta_name,
        tfce_enabled=params["tfce_enabled"],
        cutoff_predict_enabled=params["cutoff_predict_enabled"],
        gm_masking=params["gm_masking"],
        bin_steps=params["bin_steps"],
        cluster_forming_threshold=params["cluster_forming_threshold"],
        monte_carlo_iterations=params["monte_carlo_iterations"],
        nprocesses=params["nprocesses"],
    )
    contribution(
        project_path, exp_df, exp_idxs, meta_name, tasks, params["tfce_enabled"]
    )

    if masks:
        for idx, mask in enumerate(masks):
            roi_ale(
                project_path,
                exp_df,
                meta_name,
                mask,
                mask_names[idx],
                monte_carlo_iterations=params["monte_carlo_iterations"],
            )


def run_probabilistic_ale(
    analysis_df, row_idx, project_path, params, exp_all_df, tasks
):
    """
    Run a probabilistic Activation Likelihood Estimation (ALE) analysis.

    Parameters
    ----------
    analysis_df : pandas.DataFrame
        DataFrame containing the analysis information, including meta-analysis
        names and conditions for experiment selection.
    row_idx : int
        Index of the current row in the analysis dataframe.
    project_path : str or Path
        Path to the project directory where results are saved.
    params : dict
        Dictionary of parameters for analysis, including Monte Carlo iterations
        and subsample size.
    exp_all_df : pandas.DataFrame
        DataFrame containing all available experimental data.
    tasks : pandas.DataFrame
        DataFrame containing task information used for compiling experiments.

    Returns
    -------
    None
        The function performs computations and saves the results.
    """
    logger = logging.getLogger("ale_logger")
    meta_name = analysis_df.iloc[row_idx, 1]

    target_n = (
        int(analysis_df.iloc[row_idx, 0][1:])
        if len(analysis_df.iloc[row_idx, 0]) > 1
        else None
    )

    result_path = (
        project_path
        / f"Results/Probabilistic/Volumes/{meta_name}_sub_ale_{target_n}.nii"
    )
    if result_path.exists():
        logger.info(f"Probabilistic ALE results for {meta_name} already exist.")
        return

    logger.info("Running Probabilistic ALE")
    conditions = analysis_df.iloc[row_idx, 2:].dropna().to_list()
    exp_idxs, _, _ = compile_experiments(conditions, tasks)
    exp_df = exp_all_df.loc[exp_idxs].reset_index(drop=True)

    check_for_exp_independence(exp_df)

    if target_n:
        probabilistic_ale(
            project_path,
            exp_df,
            meta_name,
            target_n=target_n,
            monte_carlo_iterations=params["monte_carlo_iterations"],
            sample_n=params["subsample_n"],
            nprocesses=params["nprocesses"],
        )
    else:
        logger.warning(f"{meta_name}: Need to specify subsampling N")


def run_contrast_analysis(
    analysis_df, row_idx, project_path, params, exp_all_df, tasks
):
    """
    Run a contrast analysis between two meta-analyses.

    Parameters
    ----------
    analysis_df : pandas.DataFrame
        DataFrame containing the analysis information.
    row_idx : int
        Index of the current row in the DataFrame.
    project_path : str or Path
        Path to the project directory.
    params : dict
        Dictionary of parameters for analysis, including significance threshold and
        number of permutations.
    exp_all_df : pandas.DataFrame
        DataFrame containing all experiment data.
    tasks : pandas.DataFrame
        DataFrame containing task information.

    Returns
    -------
    None
        The function performs computations and saves the results as NIfTI files in the
        specified `project_path` directory.
    """
    meta_names, exp_dfs, exp_idxs = setup_contrast_data(
        analysis_df, row_idx, exp_all_df, tasks
    )

    check_for_exp_independence(exp_dfs[0])
    check_for_exp_independence(exp_dfs[1])

    for idx, meta_name in enumerate(meta_names):
        result_path = project_path / f"Results/MainEffect/Volumes/{meta_name}_cFWE.nii"
        if not result_path.exists():
            logger = logging.getLogger("ale_logger")
            logger.info(
                f"Running main effect for {meta_name} as prerequisite for contrast analysis"
            )
            main_effect(
                project_path,
                exp_dfs[idx],
                meta_name,
                tfce_enabled=params["tfce_enabled"],
                cutoff_predict_enabled=params["cutoff_predict_enabled"],
                bin_steps=params["bin_steps"],
                cluster_forming_threshold=params["cluster_forming_threshold"],
                monte_carlo_iterations=params["monte_carlo_iterations"],
                nprocesses=params["nprocesses"],
            )
            contribution(
                project_path,
                exp_dfs[idx],
                exp_idxs[idx],
                meta_name,
                tasks,
                params["tfce_enabled"],
            )

    exp_overlap = set(exp_dfs[0].index) & set(exp_dfs[1].index)
    exp_dfs = [exp_dfs[0].drop(exp_overlap), exp_dfs[1].drop(exp_overlap)]

    contrast(
        project_path,
        meta_names,
        correction_method=params["contrast_correction_method"],
        significance_threshold=params["significance_threshold"],
        null_repeats=params["contrast_permutations"],
        nprocesses=params["nprocesses"],
    )


def run_balanced_contrast(
    analysis_df, row_idx, project_path, params, exp_all_df, tasks
):
    """
    Run a balanced contrast analysis using the provided experiment data.

    Parameters
    ----------
    analysis_df : pandas.DataFrame
        DataFrame containing the analysis information.
    row_idx : int
        Index of the current row in the DataFrame.
    project_path : str or Path
        Path to the project directory.
    params : dict
        Dictionary of parameters for analysis, including TFCE and cutoff prediction settings.
    exp_all_df : pandas.DataFrame
        DataFrame containing all experiment data.
    tasks : pandas.DataFrame
        DataFrame containing task information.

    Returns
    -------
    None
        The function performs computations and saves the results as NIfTI files in the
        specified `project_path` directory.
    """
    meta_names, exp_dfs, exp_idxs = setup_contrast_data(
        analysis_df, row_idx, exp_all_df, tasks
    )

    check_for_exp_independence(exp_dfs[0])
    check_for_exp_independence(exp_dfs[1])

    target_n = determine_target_n(analysis_df.iloc[row_idx, 0], exp_dfs)

    # Check if subsampling ALE were already run; if not - run them
    for idx, meta_name in enumerate(meta_names):
        result_path = (
            project_path
            / f"Results/MainEffect/Volumes/{meta_name}_sub_ale_{target_n}.nii"
        )
        if not result_path.exists():
            logger = logging.getLogger("ale_logger")
            logger.info(
                f"Running subsampling ale for {meta_name} as prerequisite for balanced contrast analysis"
            )
            probabilistic_ale(
                project_path,
                exp_dfs[idx],
                meta_name,
                target_n=target_n,
                monte_carlo_iterations=params["monte_carlo_iterations"],
                sample_n=params["subsample_n"],
                nprocesses=params["nprocesses"],
            )

    balanced_contrast(
        project_path,
        exp_dfs,
        meta_names,
        target_n,
        difference_iterations=params["difference_iterations"],
        monte_carlo_iterations=params["monte_carlo_iterations"],
        nprocesses=params["nprocesses"],
    )


def run_ma_clustering(analysis_df, row_idx, project_path, params, exp_all_df, tasks):
    """
    Run a meta-analysis clustering (MA-Clustering) analysis based on the analysis dataframe and experiment info.

    Parameters
    ----------
    analysis_df : pandas.DataFrame
        DataFrame containing the analysis information, including meta-analysis names and conditions for experiment selection.
    row_idx : int
        Index of the current row in the analysis dataframe.
    project_path : str or Path
        Path to the project directory where results are saved.
    params : dict
        Dictionary of parameters for analysis, including clustering method, correlation type, linkage method, max clusters, subsample fraction, sampling iterations and null iterations.
    exp_all_df : pandas.DataFrame
        DataFrame containing all available experimental data.
    tasks : pandas.DataFrame
        DataFrame containing task information used for compiling experiments.

    Returns
    -------
    None
        The function performs computations and saves the results.
    """
    logger = logging.getLogger("ale_logger")
    logger.info("Running MA Clustering")

    meta_name = analysis_df.iloc[row_idx, 1]
    conditions = analysis_df.iloc[row_idx, 2:].dropna().to_list()
    exp_idxs, masks, mask_names = compile_experiments(conditions, tasks)
    exp_df = exp_all_df.loc[exp_idxs].reset_index(drop=True)

    check_for_exp_independence(exp_df)

    logger.info(
        f"{meta_name} : {len(exp_idxs)} experiments; average of {exp_df.Subjects.mean():.2f} subjects per experiment"
    )

    clustering(
        project_path=project_path,
        exp_df=exp_df,
        meta_name=meta_name,
        correlation_type=params["correlation_type"],
        clustering_method=params["clustering_method"],
        linkage_method=params["linkage_method"],
        max_clusters=params["max_clusters"],
        subsample_fraction=params["subsample_fraction"],
        sampling_iterations=params["sampling_iterations"],
        null_iterations=params["null_iterations"],
        use_pooled_std=params["use_pooled_std"],
    )
