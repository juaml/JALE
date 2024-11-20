import sys

import numpy as np
import pandas as pd

from jale.core.utils.tal2icbm_spm import tal2icbm_spm
from jale.core.utils.template import MNI_AFFINE


def load_excel(filepath, type="analysis"):
    """
    Load an Excel file and perform basic processing based on the specified type.

    Depending on the file type, this function reads an Excel file, assigns headers,
    handles missing values, and sets specific column names for 'experiment' data.

    Parameters
    ----------
    filepath : str or Path
        Path to the Excel file to be loaded.
    type : str, optional
        Type of the Excel file, either "analysis" or "experiment".
        Defaults to "analysis".

    Returns
    -------
    pandas.DataFrame
        DataFrame with the loaded and processed data.
    """

    # Set header row based on file type
    header = None
    if type == "experiment":
        header = 0

    # Attempt to load the Excel file and handle errors if they occur
    try:
        df = pd.read_excel(filepath, header=header)
    except FileNotFoundError:
        print(f"File '{filepath}' not found.")
        sys.exit()
    except ValueError:
        print(
            f"Error reading Excel file '{filepath}'. Make sure it's a valid Excel file."
        )
        sys.exit()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit()

    # Drop any rows that are completely empty
    df.dropna(inplace=True, how="all")

    # If type is 'experiment', rename the first columns to standard names
    if type == "experiment":
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
            print(
                f"Non-numeric Coordinates in column {coord_col}: {rows_with_errors.values + 2}"
            )

    # Exit if any non-numeric coordinates were found; otherwise, reset index and return df
    if all_coord_numbers_flag == 0:
        sys.exit()
    else:
        return df.reset_index(drop=True)


def concat_coordinates(exp_info):
    """
    Concatenate coordinate columns into arrays grouped by article.

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

    # Check for missing values in 'Articles' column
    if exp_info["Articles"].isna().sum() == 0:
        # Group by 'Articles' and consolidate coordinates into lists
        exp_info_firstlines = exp_info.groupby("Articles").first().reset_index()
        exp_info_firstlines["x"] = exp_info.groupby("Articles")["x"].apply(list).values
        exp_info_firstlines["y"] = exp_info.groupby("Articles")["y"].apply(list).values
        exp_info_firstlines["z"] = exp_info.groupby("Articles")["z"].apply(list).values

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
        lambda row: row.iloc[6:].dropna().str.lower().str.strip().values, axis=1
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
            exp_info.apply(lambda row: np.any(row.Tags == task), axis=1)
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


def read_experiment_info(filename):
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
    exp_info = load_excel(filepath=filename, type="experiment")

    # Verify coordinates are numeric and concatenate tag information
    exp_info = check_coordinates_are_numbers(exp_info)
    exp_info = concat_tags(exp_info)

    # Concatenate coordinates for each article and convert coordinate spaces if needed
    exp_info = concat_coordinates(exp_info)
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
