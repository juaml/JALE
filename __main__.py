import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from core.analyses.contrast import balanced_contrast, contrast
from core.analyses.main_effect import main_effect
from core.utils.compile_experiments import compile_experiments
from core.utils.contribution import contribution
from core.utils.folder_setup import folder_setup
from core.utils.input import load_excel, read_experiment_info


def load_config(yaml_path):
    """Load configuration from YAML file."""
    try:
        with open(yaml_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"YAML file not found at path: {yaml_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error loading YAML file: {e}")
        sys.exit(1)


def setup_project_folder(config):
    """Set up project paths and folders based on configuration."""
    project_path = Path(config["project"]["path"]).resolve()
    folder_setup(project_path)
    return project_path


def setup_logger(project_path):
    """Initialize logging with a file handler in the project directory."""
    logger = logging.getLogger("pyALE_logger")
    logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)

    # File handler in the project directory
    start_time = datetime.now().strftime("%Y%m%d_%H%M")
    file_handler = logging.FileHandler(project_path / f"logs/{start_time}.log")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_format)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def load_dataframes(project_path, config):
    """Load experiment info and analysis dataframes."""
    exp_all_df, tasks = read_experiment_info(
        project_path / config["project"]["experiment_info"]
    )
    analysis_df = load_excel(
        project_path / config["project"]["analysis_info"], type="analysis"
    )
    return exp_all_df, tasks, analysis_df


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
    logger = logging.getLogger("pyALE_logger")
    logger.info("Running Main-Effect Analysis")
    meta_name = analysis_df.iloc[row_idx, 1]
    conditions = analysis_df.iloc[row_idx, 2:].dropna().to_list()
    exp_idxs, _, _ = compile_experiments(conditions, tasks)
    exp_df = exp_all_df.loc[exp_idxs].reset_index(drop=True)

    if len(exp_idxs) <= 17:
        logger.warning(
            "Analysis contains less than 18 experiments; interpret results carefully."
        )

    logger.info(
        f"{meta_name} : {len(exp_idxs)} experiments; average of {exp_df.Subjects.mean():.2f} subjects per experiment"
    )
    main_effect(
        project_path,
        exp_df,
        meta_name,
        tfce_enabled=params["tfce_enabled"],
        cutoff_predict_enabled=params["cutoff_predict_enabled"],
        bin_steps=params["bin_steps"],
        cluster_forming_threshold=params["cluster_forming_threshold"],
        monte_carlo_iterations=params["monte_carlo_iterations"],
        nprocesses=params["nprocesses"],
    )
    contribution(project_path, exp_df, meta_name, tasks, params["tfce_enabled"])


def run_probabilistic_ale(
    analysis_df, row_idx, project_path, params, exp_all_df, tasks
):
    """
    Run a probabilistic Activation Likelihood Estimation (ALE) analysis.

    This function performs a probabilistic ALE analysis using the specified
    experiment data and parameters. It checks if a target number of subsamples
    is specified in the analysis dataframe. If so, it calls the `main_effect`
    function with subsampling; otherwise, it logs a warning message.

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
    logger = logging.getLogger("pyALE_logger")
    logger.info("Running Probabilistic ALE")
    meta_name = analysis_df.iloc[row_idx, 1]
    conditions = analysis_df.iloc[row_idx, 2:].dropna().to_list()
    exp_idxs, _, _ = compile_experiments(conditions, tasks)
    exp_df = exp_all_df.loc[exp_idxs].reset_index(drop=True)

    if len(analysis_df.iloc[row_idx, 0]) > 1:
        target_n = int(analysis_df.iloc[row_idx, 0][1:])
        main_effect(
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

    This function prepares and runs a contrast analysis between two meta-analyses
    by setting up experiment data, checking and computing main effects if necessary,
    removing overlapping experiments, and computing statistical contrasts.

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
    meta_names, exp_dfs = setup_contrast_data(analysis_df, row_idx, exp_all_df, tasks)
    check_and_run_main_effect(meta_names, exp_dfs, project_path, params, tasks)

    # Remove overlap in experiments for contrast analysis
    exp_overlap = set(exp_dfs[0].index) & set(exp_dfs[1].index)
    exp_dfs = [exp_dfs[0].drop(exp_overlap), exp_dfs[1].drop(exp_overlap)]

    contrast(
        project_path,
        meta_names,
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

    meta_names, exp_dfs = setup_contrast_data(analysis_df, row_idx, exp_all_df, tasks)
    target_n = determine_target_n(analysis_df.iloc[row_idx, 0], exp_dfs)

    check_and_run_main_effect(
        meta_names, exp_dfs, project_path, params, tasks, target_n
    )

    balanced_contrast(
        project_path,
        exp_dfs,
        meta_names,
        target_n,
        difference_iterations=params["difference_iterations"],
        monte_carlo_iterations=params["monte_carlo_iterations"],
        nprocesses=2,
    )


def setup_contrast_data(analysis_df, row_idx, exp_all_df, tasks):
    """
    Prepare experiment data for contrast analysis.

    This function extracts the names and conditions of meta-analyses from the
    given analysis dataframe for two consecutive rows starting at `row_idx`. It
    compiles the experiment indices based on these conditions and returns the
    meta-analysis names along with the corresponding experiment dataframes.

    Parameters
    ----------
    analysis_df : pandas.DataFrame
        DataFrame containing analysis information, including meta-analysis names
        and conditions for experiment selection.
    row_idx : int
        Index of the current row in the analysis dataframe from which to start
        extracting meta-analysis data.
    exp_all_df : pandas.DataFrame
        DataFrame containing all available experimental data.
    tasks : pandas.DataFrame
        DataFrame containing task information used for compiling experiments.

    Returns
    -------
    tuple
        A tuple containing:
        - list of str: Names of the meta-analyses for the selected rows.
        - list of pandas.DataFrame: DataFrames for the experiments corresponding
          to each meta-analysis.
    """
    meta_names = [analysis_df.iloc[row_idx, 1], analysis_df.iloc[row_idx + 1, 1]]
    conditions = [
        analysis_df.iloc[row_idx, 2:].dropna().to_list(),
        analysis_df.iloc[row_idx + 1, 2:].dropna().to_list(),
    ]
    exp_idxs1, _, _ = compile_experiments(conditions[0], tasks)
    exp_idxs2, _, _ = compile_experiments(conditions[1], tasks)

    exp_dfs = [
        exp_all_df.loc[exp_idxs1].reset_index(drop=True),
        exp_all_df.loc[exp_idxs2].reset_index(drop=True),
    ]
    return meta_names, exp_dfs


def determine_target_n(row_value, exp_dfs):
    """
    Determine the target number of subsamples for analysis.

    This function calculates the target 'n' for subsampling based on the input `row_value`.
    If `row_value` has more than one character, it extracts and returns the integer value
    from the second character onwards. Otherwise, it computes a target number by considering
    the sizes of the experimental dataframes in `exp_dfs`, taking the minimum of the average
    of the smallest size and 17, reduced by 2.

    Parameters
    ----------
    row_value : str
        A string value from the analysis dataframe indicating the target subsample size.
    exp_dfs : list of pandas.DataFrame
        List of DataFrames containing experiment data for different meta-analyses.

    Returns
    -------
    int
        The calculated target number of subsamples.
    """
    if len(row_value) > 1:
        return int(row_value[1:])
    n = [len(exp_dfs[0]), len(exp_dfs[1])]
    return int(min(np.floor(np.mean((np.min(n), 17))), np.min(n) - 2))


def check_and_run_main_effect(
    meta_names, exp_dfs, project_path, params, tasks, target_n=None
):
    """
    Check for existing main effect results and run analysis if not found.

    This function iterates over the provided meta-analysis names and checks if the
    main effect results already exist for each. If the results are missing, it computes
    the main effect using the `main_effect` function and performs a contribution analysis.

    Parameters
    ----------
    meta_names : list of str
        List of meta-analysis names for which to check and run main effect analysis.
    exp_dfs : list of pandas.DataFrame
        List of DataFrames containing experiment data for each meta-analysis.
    project_path : str or Path
        Path to the project directory.
    params : dict
        Dictionary of parameters for analysis, including TFCE and cutoff prediction settings.
    tasks : pandas.DataFrame
        DataFrame containing task information.
    target_n : int, optional
        Target number of subsamples for probabilistic ALE, by default None.
    """
    for idx, meta_name in enumerate(meta_names):
        result_path = (
            project_path / f"Results/MainEffect/Full/Volumes/{meta_name}_cFWE05.nii"
        )
        if not result_path.exists():
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
                target_n=target_n,
            )
            contribution(
                project_path, exp_dfs[idx], meta_name, tasks, params["tfce_enabled"]
            )


def main():
    # Load config and set up paths
    yaml_path = sys.argv[1]
    config = load_config(yaml_path)
    project_path = setup_project_folder(config)

    # Initialize the logger after setting up the project directory
    logger = setup_logger(project_path)
    logger.info("Logger initialized and project setup complete.")

    params = config.get("parameters", {})
    exp_all_df, tasks, analysis_df = load_dataframes(project_path, config)

    # Main loop to process each row in the analysis dataframe
    for row_idx in range(analysis_df.shape[0]):
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


if __name__ == "__main__":
    main()
