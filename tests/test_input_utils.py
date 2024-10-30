import os.path as op

import numpy as np
import pandas as pd
import pytest

from core.utils.input import (
    check_coordinates_are_numbers,
    concat_coordinates,
    concat_tags,
    convert_tal_2_mni,
    create_tasks_table,
    load_excel,
    transform_coordinates_to_voxel_space,
)
from tests.utils import get_test_data_path

"""Load excel fixtures"""


@pytest.fixture(scope="module")
def loaded_excel_base():
    # Load the Excel file
    data_file = op.join(get_test_data_path(), "test_expinfo_correct.xlsx")
    exp_info = pd.read_excel(data_file)
    exp_info.dropna(inplace=True, how="all")
    return exp_info


@pytest.fixture(scope="module")
def loaded_excel_base_multi():
    # Load the Excel file
    data_file = op.join(get_test_data_path(), "test_expinfo_correct_multi.xlsx")
    exp_info = pd.read_excel(data_file)
    exp_info.dropna(inplace=True, how="all")
    return exp_info


@pytest.fixture(scope="module")
def loaded_excel_concat():
    # Load the Excel file
    data_file = op.join(get_test_data_path(), "test_expinfo_concat.pickle")
    exp_info = pd.read_pickle(data_file)
    return exp_info


"""Test load/concat/transform functions"""


def test_load_excel(tmp_path):
    data_file = op.join(get_test_data_path(), "test_expinfo_correct.xlsx")
    loaded_df = load_excel(data_file, type="experiment")
    assert loaded_df.shape == (25, 8)

    # Test loading a non-existent file
    with pytest.raises(SystemExit):
        load_excel("non_existent_file.xlsx")

    # Create a temporary invalid file
    test_file = tmp_path / "test.txt"
    # Test that a ValueError is raised for invalid Excel files
    with pytest.raises(SystemExit):
        load_excel(test_file)


def test_coordinate_numbers_check_true(loaded_excel_base, capfd):
    exp_info = check_coordinates_are_numbers(loaded_excel_base)
    assert exp_info.index.to_list()[-1] == 24

    data_file = op.join(get_test_data_path(), "test_expinfo_coordinate_letter.xlsx")
    exp_info = pd.read_excel(data_file)
    exp_info.dropna(inplace=True, how="all")
    with pytest.raises(SystemExit):
        exp_info = check_coordinates_are_numbers(exp_info)
    out, err = capfd.readouterr()
    assert out == "Non-numeric Coordinates in column x: [13]\n"


def test_concat_tags(loaded_excel_base):
    exp_info = concat_tags(loaded_excel_base)
    assert exp_info.shape == (25, 7)
    assert list(exp_info.Tags[0]) == ["a", "visual"]


def test_concat_coordinates(loaded_excel_base, loaded_excel_base_multi):
    exp_info_firstlines = concat_coordinates(loaded_excel_base)
    assert np.array_equal(exp_info_firstlines.Coordinates_mm[0][0], [12, 18, 48])
    assert exp_info_firstlines.NumberOfFoci[0] == 5

    exp_info_firstlines = concat_coordinates(loaded_excel_base_multi)
    assert np.array_equal(exp_info_firstlines.Coordinates_mm[0][0], [12, 18, 48])
    assert exp_info_firstlines.NumberOfFoci[0] == 5


def test_convert_tal_2_mni(loaded_excel_concat):
    exp_info = convert_tal_2_mni(loaded_excel_concat)
    assert np.array_equal(exp_info.Coordinates_mm[1][0], np.array([4.0, 21.0, 39.0]))


def test_convert_2_voxel_space(loaded_excel_concat, capfd):
    exp_info = transform_coordinates_to_voxel_space(loaded_excel_concat)
    assert np.array_equal(exp_info.Coordinates[0][0], np.array([39, 72, 60]))


def test_create_tasks_table(loaded_excel_concat):
    tasks = create_tasks_table(loaded_excel_concat)
    assert tasks.TotalSubjects[1] == 20
