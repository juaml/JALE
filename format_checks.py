import pandas as pd

def find_row_indices_with_non_number_values(df, column_name):
    """
    Identify row indices in the DataFrame where the specified column contains non-integer values.

    Args:
    df (pd.DataFrame): The DataFrame to check.
    column_name (str): The name of the column to check.

    Returns:
    pd.Index: An Index object containing the row indices with non-integer values in the specified column.
    """
    coerced_column = pd.to_numeric(df[column_name], errors='coerce')
    non_integer_mask = (coerced_column.isnull()) | (coerced_column % 1 != 0)
    return df.index[non_integer_mask]

def is_column_only_numbers(df, column_name):
    """
    Check if all values in the specified column of the Pandas DataFrame are floats.

    Args:
    df (pd.DataFrame): The DataFrame to check.
    column_name (str): The name of the column to check.

    Returns:
    bool: True if all values are floats, False otherwise.
    """
    return pd.api.types.is_float_dtype(df[column_name])

def check_coordinates_for_numbers(df):
    all_true_flag = 1
    for coord_col in ['x', 'y', 'z']:
        coord_col_all_number_bool = is_column_only_numbers(df, coord_col)
        if coord_col_all_number_bool == False:
            all_true_flag = 0
            rows_with_errors = find_row_indices_with_non_number_values(df, coord_col)
            print(f'Non-numeric Coordinates in column {coord_col}: {rows_with_errors.values + 2}')

    if all_true_flag == False:
        exit()
    else:
        return df.reset_index(drop=True)