import os
import sys
import pandas as pd
import numpy as np
from format_checks import check_coordinates_for_numbers

def load_excel(filepath):
    """
    Load an Excel file into a Pandas DataFrame.

    This function attempts to read an Excel file from the given filepath. If successful, it returns
    the file content as a Pandas DataFrame. The function handles several types of exceptions
    that might occur during file reading, including file not found, invalid file format, and other
    general exceptions.

    Args:
    filepath (str): The path to the Excel file to be loaded.

    Returns:
    pd.DataFrame: A DataFrame containing the data from the Excel file.

    Raises:
    SystemExit: If the file is not found, is not a valid Excel file, or if any other exception occurs
                during the file reading process, an error message is printed and the program is terminated.
    """
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        print(f"File '{filepath}' not found.")
        exit()
    except ValueError:
        print(f"Error reading Excel file '{filepath}'. Make sure it's a valid Excel file.")
        exit()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        exit()
    df.dropna(inplace=True, how='all')
    return df

def concat_coordinates(experiment_info_df):
    article_rows = experiment_info_df.index[experiment_info['Articles'].notnull()].tolist()
    end_of_articles = [x - 1 for x in article_rows]
    end_of_articles.pop(0)
    end_of_articles.append(experiment_info_df.shape[0])

    experiment_info_df_first_lines = experiment_info_df.loc[article_rows].reset_index(drop=True)
    experiment_info_df_first_lines = experiment_info_df_first_lines.drop(['x','y','z'], axis=1)

    experiment_info_df_first_lines['Coordinates'] = np.nan
    experiment_info_df_first_lines['Coordinates'] = experiment_info_df_first_lines['Coordinates'].astype(object)

    for i in range(len(article_rows)):
        x = experiment_info_df.loc[article_rows[i]:end_of_articles[i]].x.values
        y = experiment_info_df.loc[article_rows[i]:end_of_articles[i]].y.values
        z = experiment_info_df.loc[article_rows[i]:end_of_articles[i]].z.values
        
        coordinate_array = np.array((x,y,z))
        experiment_info_df_first_lines.loc[i,'Coordinates'] = [coordinate_array]
    
    return experiment_info_df_first_lines

def concat_tags(experiment_info_df):
    experiment_info_df['Tags'] = experiment_info_df.apply(lambda row: row.iloc[3:-2].tolist(), axis=1)
    experiment_info_df = experiment_info_df.drop(experiment_info_df.iloc[:, 3:-2],axis = 1)
    return experiment_info_df

filepath = sys.argv[1]
experiment_info = load_excel(filepath=filepath)
experiment_info = check_coordinates_for_numbers(df=experiment_info)
experiment_info_concat_coords = concat_coordinates(experiment_info)
experiment_info_concat_tags = concat_tags(experiment_info_concat_coords)
experiment_info_concat_reorder = experiment_info_concat_tags[['Articles', 'Subjects', 'CoordinateSpace', 'Tags', 'Coordinates']]

experiment_info_concat_reorder.to_excel('experiment_info_concat.xlsx', index=False)
