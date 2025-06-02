import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from jsonschema import validate

from jale.core.utils.compile_experiments import compile_experiments
from jale.core.utils.tal2icbm_spm import tal2icbm_spm
from jale.core.utils.template import MNI_AFFINE

logger = logging.getLogger("ale_logger")


def load_config(yaml_path):
    """Load configuration from YAML file."""
    try:
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
            config = validate_config(config)
    except FileNotFoundError:
        logger.error(f"YAML file not found at path: {yaml_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error loading YAML file: {e}")
        sys.exit(1)

    return config


def load_experiment_file(filepath):
    """
    Load an Excel or CSV file and perform basic processing based on the specified type.

    This function reads a file (either Excel or CSV), assigns headers,
    handles missing values, and sets specific column names for 'experiment' data.

    Parameters
    ----------
    filepath : str or Path
        Path to the file to be loaded.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the loaded and processed data.
    """
    # Convert filepath to Path object if it's a string
    filepath = Path(filepath)

    # Check the file extension to determine the loading method
    try:
        if filepath.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(filepath, header=0)
        elif filepath.suffix.lower() == ".csv":
            df = pd.read_csv(filepath, header=0)
        else:
            logger.error(f"Unsupported file format: {filepath.suffix}")
            sys.exit()
    except FileNotFoundError:
        logger.error(f"File '{filepath}' not found.")
        sys.exit()
    except ValueError:
        logger.error(f"Error reading file '{filepath}'. Make sure it's a valid file.")
        sys.exit()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit()

    # Drop any rows that are completely empty
    df.dropna(inplace=True, how="all")

    # Check for rows with only one or two non-NaN entries
    mistake_rows = df[(df.notna().sum(axis=1) == 1) | (df.notna().sum(axis=1) == 2)]
    if not mistake_rows.empty:
        row_indices = mistake_rows.index.tolist()
        row_indices = np.array(row_indices) + 2
        logger.error(
            f"Error: Rows with only one or two entries found at indices: {row_indices}"
        )
        sys.exit()

    # Rename the first columns to standard names
    current_column_names = df.columns.values
    current_column_names[:6] = [
        "Articles",
        "Subjects",
        "x",
        "y",
        "z",
        "CoordinateSpace",
    ]
    df.columns = current_column_names

    df[["x", "y", "z"]] = df[["x", "y", "z"]].astype(float)

    return df


def load_analysis_file(filepath):
    """
    Load an Excel or CSV file for analysis.

    This function reads a file (either Excel or CSV), handles missing values,
    and prepares the data for further analysis.

    Parameters
    ----------
    filepath : str or Path
        Path to the file to be loaded.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the loaded and processed data.
    """
    # Convert filepath to Path object if it's a string
    filepath = Path(filepath)

    # Check the file extension to determine the loading method
    try:
        if filepath.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(filepath, header=None)
        elif filepath.suffix.lower() == ".csv":
            df = pd.read_csv(filepath, header=None)
        else:
            logger.error(f"Unsupported file format: {filepath.suffix}")
            sys.exit()
    except FileNotFoundError:
        logger.error(f"File '{filepath}' not found.")
        sys.exit()
    except ValueError:
        logger.error(f"Error reading file '{filepath}'. Make sure it's a valid file.")
        sys.exit()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit()

    df.dropna(inplace=True, how="all")

    return df


def check_coordinates_are_numbers(df):
    """
    Check if coordinate columns in a DataFrame contain only numeric values.

    This function verifies that 'x', 'y', and 'z' columns contain numeric values.
    If non-numeric values are found, it prints the row numbers with errors and exits.
    If all values are valid, it resets the index and returns the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'x', 'y', and 'z' coordinate columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with reset index if all coordinates are numeric.
    """

    # Initialize flag to track if all coordinates are numeric
    all_coord_numbers_flag = 1

    # Check each coordinate column for non-numeric values
    for coord_col in ["x", "y", "z"]:
        # Check if the column contains only float values
        coord_col_all_number_bool = pd.api.types.is_float_dtype(df[coord_col])

        # If non-numeric values are found, print their row numbers and set the flag
        if not coord_col_all_number_bool:
            all_coord_numbers_flag = 0
            coerced_column = pd.to_numeric(df[coord_col], errors="coerce")
            non_integer_mask = (coerced_column.isnull()) | (coerced_column % 1 != 0)
            rows_with_errors = df.index[non_integer_mask]
            logger.error(
                f"Non-numeric Coordinates in column {coord_col}: {rows_with_errors.values + 2}"
            )

    # Exit if any non-numeric coordinates were found; otherwise, reset index and return df
    if all_coord_numbers_flag == 0:
        sys.exit()
    else:
        return df.reset_index(drop=True)


def concat_coordinates(exp_info, pool_experiments):
    """
    Concatenate coordinate columns into arrays grouped by article + tag.

    This function consolidates 'x', 'y', and 'z' coordinates into a single array
    for each article, creating a 'Coordinates_mm' column. It also counts the number
    of foci for each article.

    Parameters
    ----------
    exp_info : pandas.DataFrame
        DataFrame containing experimental data with 'Articles', 'x', 'y', and 'z' columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with concatenated coordinates for each article and a count of foci.
    """

    # logic for excel files where each line features information in every cell (old structure)
    if exp_info["Articles"].isna().sum() == 0:
        if pool_experiments:
            grouping_columns = ["Articles"]
        else:
            grouping_columns = ["Articles", "Tags"]
        # Group by 'Articles' and consolidate coordinates into lists
        exp_info_firstlines = exp_info.groupby(grouping_columns).first().reset_index()
        exp_info_firstlines["x"] = (
            exp_info.groupby(grouping_columns)["x"].apply(list).values
        )
        exp_info_firstlines["y"] = (
            exp_info.groupby(grouping_columns)["y"].apply(list).values
        )
        exp_info_firstlines["z"] = (
            exp_info.groupby(grouping_columns)["z"].apply(list).values
        )

        # Create an array of coordinates and assign it to 'Coordinates_mm'
        exp_info_firstlines["Coordinates_mm"] = exp_info_firstlines.apply(
            lambda row: np.array([row["x"], row["y"], row["z"]]).T, axis=1
        )

        # Drop original coordinate columns
        exp_info_firstlines = exp_info_firstlines.drop(["x", "y", "z"], axis=1)

        # Calculate and add the number of foci for each article
        exp_info_firstlines["NumberOfFoci"] = exp_info_firstlines.apply(
            lambda row: row["Coordinates_mm"].shape[0], axis=1
        )

    # logic for excel files where author, subject N, coordinate space and tags are only in first line
    else:
        # Get rows where 'Articles' column has data
        article_rows = exp_info.index[exp_info["Articles"].notnull()].tolist()
        # Identify the last row for each article to separate data blocks
        end_of_articles = [x - 1 for x in article_rows]
        end_of_articles.pop(0)
        end_of_articles.append(exp_info.shape[0])

        # Initialize 'Coordinates_mm' and 'NumberOfFoci' columns for the results
        exp_info_firstlines = exp_info.loc[article_rows].reset_index(drop=True)
        exp_info_firstlines = exp_info_firstlines.drop(["x", "y", "z"], axis=1)
        exp_info_firstlines["Coordinates_mm"] = np.nan
        exp_info_firstlines["Coordinates_mm"] = exp_info_firstlines[
            "Coordinates_mm"
        ].astype(object)
        exp_info_firstlines["NumberOfFoci"] = np.nan

        # Iterate over each article to concatenate coordinates into arrays
        for i in range(len(article_rows)):
            # Extract coordinates for the current article
            x = exp_info.loc[article_rows[i] : end_of_articles[i]].x.values
            y = exp_info.loc[article_rows[i] : end_of_articles[i]].y.values
            z = exp_info.loc[article_rows[i] : end_of_articles[i]].z.values

            # Create a 2D array of coordinates and assign it to 'Coordinates_mm'
            coordinate_array = np.array((x, y, z)).T
            exp_info_firstlines.at[i, "Coordinates_mm"] = coordinate_array

            # Count the number of foci for each article
            exp_info_firstlines.loc[i, "NumberOfFoci"] = len(x)

    return exp_info_firstlines


def concat_tags(exp_info):
    """
    Concatenate non-null tag columns for each row in a DataFrame into a single list.

    This function collects all non-null tags from columns after the sixth position,
    converts them to lowercase, strips whitespace, and stores them in a 'Tags' column.

    Parameters
    ----------
    exp_info : pandas.DataFrame
        DataFrame containing experiment information, with tags in columns after the sixth.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a new 'Tags' column and unnecessary tag columns removed.
    """
    # Collect all non-null tag columns for each row and format them as lowercase strings
    exp_info["Tags"] = exp_info.apply(
        lambda row: tuple(row.iloc[6:].dropna().str.lower().str.strip().values), axis=1
    )

    # Drop original tag columns, keeping only up to the 'Tags' column
    exp_info = exp_info.drop(exp_info.iloc[:, 6:-1], axis=1)

    return exp_info


def convert_tal_2_mni(exp_info):
    """
    Convert TAL coordinates to MNI space in a DataFrame.

    This function converts coordinates in 'Coordinates_mm' from TAL to MNI space
    for rows where 'CoordinateSpace' is set to 'TAL'.

    Parameters
    ----------
    exp_info : pandas.DataFrame
        DataFrame containing experiment information with 'Coordinates_mm'
        and 'CoordinateSpace' columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with TAL coordinates converted to MNI space.
    """
    # Apply TAL-to-MNI conversion to rows where 'CoordinateSpace' is 'TAL'
    exp_info.loc[exp_info["CoordinateSpace"] == "TAL", "Coordinates_mm"] = exp_info[
        exp_info["CoordinateSpace"] == "TAL"
    ].apply(lambda row: tal2icbm_spm(row["Coordinates_mm"]), axis=1)

    return exp_info


def transform_coordinates_to_voxel_space(exp_info):
    """
    Transform MNI coordinates to voxel space and constrain values by threshold.

    This function transforms coordinates in 'Coordinates_mm' from MNI to voxel space,
    padding them to homogeneous coordinates for matrix multiplication. Values are then
    constrained by predefined thresholds to avoid exceeding voxel dimensions.

    Parameters
    ----------
    exp_info : pandas.DataFrame
        DataFrame with 'Coordinates_mm' containing MNI coordinates for each experiment.

    Returns
    -------
    pandas.DataFrame
        DataFrame with transformed 'Coordinates' column in voxel space.
    """
    # Pad 'Coordinates_mm' to homogeneous coordinates and store in 'padded_xyz'
    padded_xyz = exp_info.apply(
        lambda row: np.pad(
            row["Coordinates_mm"], ((0, 0), (0, 1)), constant_values=[1]
        ),
        axis=1,
    ).values

    # Transform padded coordinates to voxel space using inverse of MNI affine matrix
    exp_info["Coordinates"] = [
        np.ceil(np.dot(np.linalg.inv(MNI_AFFINE), xyzmm.T))[:3].T.astype(int)
        for xyzmm in padded_xyz
    ]

    # Constrain voxel coordinates by maximum threshold to stay within bounds
    thresholds = [90, 108, 90]
    exp_info["Coordinates"] = exp_info.apply(
        lambda row: np.minimum(row["Coordinates"], thresholds), axis=1
    )

    return exp_info


def create_tasks_table(exp_info):
    """
    Create a tasks summary table from experiment information.

    This function generates a DataFrame summarizing tasks associated with each
    experiment, including the number of experiments, articles involved, total
    subjects, and experiment indices for each task.

    Parameters
    ----------
    exp_info : pandas.DataFrame
        DataFrame containing experiment data with 'Tags', 'Articles', and 'Subjects' columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame summarizing tasks, with columns for task name, experiment count,
        associated articles, total subjects, and experiment indices.
    """
    # Initialize the tasks DataFrame with specified columns
    tasks = pd.DataFrame(
        columns=["Name", "Num_Exp", "Who", "TotalSubjects", "ExpIndex"]
    )

    # Calculate unique task names and the number of occurrences for each task
    task_names, task_counts = np.unique(np.hstack(exp_info["Tags"]), return_counts=True)
    tasks["Name"] = task_names
    tasks["Num_Exp"] = task_counts

    # Initialize a list to store experiment indices associated with each task
    task_exp_idxs = []

    # Populate task-specific information: experiment indices, articles, and total subjects
    for count, task in enumerate(task_names):
        # Get experiment indices where the task appears
        task_exp_idxs = exp_info.index[
            exp_info.apply(lambda row: task in row.Tags, axis=1)
        ].to_list()

        # Assign details to tasks DataFrame
        tasks.at[count, "ExpIndex"] = task_exp_idxs
        tasks.at[count, "Who"] = exp_info.loc[task_exp_idxs, "Articles"].values
        tasks.at[count, "TotalSubjects"] = np.sum(
            exp_info.loc[task_exp_idxs, "Subjects"].values
        )

    # Add a row summarizing all experiments
    tasks.loc[len(tasks)] = [
        "all",
        exp_info.shape[0],
        exp_info["Articles"].values,
        np.sum(exp_info["Subjects"].values),
        list(range(exp_info.shape[0])),
    ]

    # Sort tasks by the number of associated experiments and reset the index
    tasks = tasks.sort_values(by="Num_Exp", ascending=False).reset_index(drop=True)

    return tasks


def read_experiment_info(filename, pool_experiments):
    """
    Load and process experimental data from an Excel file, creating a summary of tasks.

    This function reads an experiment file, processes the data through multiple
    transformations (e.g., coordinate validation, tag concatenation), and saves
    the processed data and tasks summary to Excel files.

    Parameters
    ----------
    filename : str or Path
        Path to the Excel file containing experiment information.

    Returns
    -------
    tuple
        - pandas.DataFrame : Processed experiment data.
        - pandas.DataFrame : Summary of tasks.
    """
    # Load the experimental data from the Excel file
    exp_info = load_experiment_file(filepath=filename)

    # Verify coordinates are numeric and concatenate tag information
    exp_info = check_coordinates_are_numbers(exp_info)
    exp_info = concat_tags(exp_info)

    # Concatenate coordinates for each article and convert coordinate spaces if needed
    exp_info = concat_coordinates(exp_info, pool_experiments)
    exp_info = convert_tal_2_mni(exp_info)

    # Transform MNI coordinates to voxel space and filter relevant columns
    exp_info = transform_coordinates_to_voxel_space(exp_info)
    exp_info = exp_info[
        [
            "Articles",
            "Subjects",
            "CoordinateSpace",
            "Tags",
            "NumberOfFoci",
            "Coordinates",
        ]
    ]

    # Save processed experiment data to an Excel file
    # exp_info.to_excel("experiment_info_concat.xlsx", index=False)

    # Create a tasks table summarizing task-related information and save it
    tasks = create_tasks_table(exp_info)
    # tasks.to_excel("tasks_info.xlsx", index=False)

    return exp_info, tasks


def load_dataframes(project_path, config):
    """Load experiment info and analysis dataframes."""
    exp_all_df, tasks = read_experiment_info(
        project_path / config["project"]["experiment_info"],
        config["parameters"]["pool_experiments"],
    )
    analysis_df = load_analysis_file(project_path / config["project"]["analysis_info"])
    return exp_all_df, tasks, analysis_df


def check_for_exp_independence(exp_df):
    """
    Check if any articles appear with multiple tags, indicating non-independent experiments.

    This function checks the experiment DataFrame for any duplicate articles and warns
    if any are found. It also provides
    optional output listing the problematic articles and their associated tags.

    Parameters
    ----------
    exp_df : pandas.DataFrame
        DataFrame containing experimental data, including 'Articles' and 'Tags' columns.
    """
    duplicate_mask = exp_df.duplicated(subset="Articles", keep=False)
    if duplicate_mask.any():
        logger.warning("Atleast one Article is associated with multiple experiments.")
        logger.warning("Please check carefully for independence of experiments.")
        logger.warning(
            "If not independent please change config and set pool_experiments to 'True'."
        )
        # Optionally, list the problematic articles and their associated tags
        duplicated_articles = exp_df[duplicate_mask]
        for article in duplicated_articles["Articles"].unique():
            logger.warning(f"{article}")


def check_params(params):
    """
    Adjust parameters based on cutoff prediction settings.

    This function checks if cutoff prediction is enabled and sets default values
    for significance threshold, cluster forming threshold, and Monte Carlo iterations
    if it is.

    Parameters
    ----------
    params : dict
        Dictionary containing analysis parameters, including a boolean for cutoff
        prediction enablement.

    Returns
    -------
    dict
        Updated dictionary with adjusted parameters if cutoff prediction is enabled.
    """

    if params["cutoff_predict_enabled"]:
        params["significance_threshold"] = 0.05
        params["cluster_forming_threshold"] = 0.001
        params["monte_carlo_iterations"] = 5000
    return params


def setup_contrast_data(analysis_df, row_idx, exp_all_df, tasks):
    """
    Prepare experiment data for contrast analysis.

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

    exp_idxs = [exp_idxs1, exp_idxs2]

    exp_dfs = [
        exp_all_df.loc[exp_idxs1].reset_index(drop=True),
        exp_all_df.loc[exp_idxs2].reset_index(drop=True),
    ]
    return meta_names, exp_dfs, exp_idxs


def determine_target_n(row_value, exp_dfs):
    """
    Determine the target number of subsamples for analysis.

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


def validate_config(config):
    """
    Validate a YAML file (structure and values).

    Parameters:
        config (dict): parsed dictionary.

    Returns:
        dict: validated config dictionary.

    Raises:
        jsonschema.ValidationError: If validation fails.
        yaml.YAMLError: If the YAML is not properly formatted.
    """

    SCHEMA = {
        "type": "object",
        "properties": {
            "project": {
                "type": "object",
                "properties": {
                    "analysis_info": {"type": "string"},
                    "experiment_info": {"type": "string"},
                },
                "required": ["analysis_info", "experiment_info"],
            },
            "parameters": {
                "type": "object",
                "properties": {
                    "pool_experiments": {"type": "boolean"},
                    "tfce_enabled": {"type": "boolean"},
                    "gm_masking": {"type": "boolean"},
                    "bin_steps": {"type": "number"},
                    "cutoff_predict_enabled": {"type": "boolean"},
                    "significance_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "cluster_forming_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "monte_carlo_iterations": {"type": "integer", "minimum": 1},
                    "subsample_n": {"type": "integer", "minimum": 1},
                    "contrast_permutations": {"type": "integer", "minimum": 1},
                    "contrast_correction_method": {
                        "type": "string",
                        "enum": ["cFWE", "vFWE", "tfce"],
                    },
                    "difference_iterations": {"type": "integer", "minimum": 1},
                    "nprocesses": {"type": "integer", "minimum": 1},
                },
                "required": [
                    "pool_experiments",
                    "tfce_enabled",
                    "gm_masking",
                    "bin_steps",
                    "cutoff_predict_enabled",
                    "significance_threshold",
                    "cluster_forming_threshold",
                    "monte_carlo_iterations",
                    "subsample_n",
                    "contrast_permutations",
                    "contrast_correction_method",
                    "difference_iterations",
                    "nprocesses",
                ],
            },
            "clustering_parameters": {
                "type": "object",
                "properties": {
                    "max_clusters": {"type": "integer", "minimum": 1},
                    "subsample_fraction": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "sampling_iterations": {"type": "integer", "minimum": 1},
                    "null_iterations": {"type": "integer", "minimum": 1},
                    "correlation_type": {
                        "type": "string",
                        "enum": ["spearman", "pearson"],
                    },
                    "clustering_method": {
                        "type": "string",
                        "enum": ["hierarchical", "kmeans"],
                    },
                    "linkage_method": {
                        "type": "string",
                        "enum": ["complete", "average", "ward"],
                    },
                    "use_pooled_std": {"type": "boolean"},
                },
                "required": [
                    "max_clusters",
                    "subsample_fraction",
                    "sampling_iterations",
                    "null_iterations",
                    "correlation_type",
                    "clustering_method",
                    "linkage_method",
                    "use_pooled_std",
                ],
            },
        },
        "required": ["project", "parameters", "clustering_parameters"],
    }

    validate(instance=config, schema=SCHEMA)
    return config
