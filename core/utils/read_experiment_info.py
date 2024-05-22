import os
import sys
import pandas as pd
import numpy as np
from core.utils.input_utils import *

def read_experiment_info(filename):
    exp_info = load_excel(filepath=filename)
    exp_info = check_coordinates_are_numbers(exp_info)
    exp_info = concat_tags(exp_info)
    exp_info = concat_coordinates(exp_info)
    exp_info = convert_tal_2_mni(exp_info)
    exp_info = transform_coordinates_to_voxel_space(exp_info)
    exp_info = exp_info[['Articles', 'Subjects', 'CoordinateSpace', 'Tags', 'NumberOfFoci', 'Coordinates']]
    exp_info.to_excel('experiment_info_concat.xlsx', index=False)

    tasks = create_tasks_table(exp_info)
    tasks.to_excel('tasks_info.xlsx', index=False)
    
    return exp_info, tasks