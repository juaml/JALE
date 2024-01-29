import os
import sys
import pandas as pd
import numpy as np
from format_checks import check_coordinates_for_numbers
from input_utils import *
from tal2icbm_spm import tal2icbm_spm
from template import affine


filepath = sys.argv[1]
exp_info = load_excel(filepath=filepath)
exp_info = check_coordinates_for_numbers(exp_info)
exp_info = concat_tags(exp_info)
exp_info = concat_coordinates(exp_info)
exp_info = convert_tal_2_mni(exp_info)
exp_info = transform_coordinates_to_voxel_space(exp_info)
exp_info = exp_info[['Articles', 'Subjects', 'CoordinateSpace', 'Tags', 'NumberOfFoci', 'Coordinates']]
exp_info.to_excel('experiment_info_concat.xlsx', index=False)

tasks = create_tasks_table(exp_info)
tasks.to_excel('tasks_info.xlsx', index=False)