import pytest
import os.path as op
from input_utils import *
from tests.utils import get_test_data_path
import pandas as pd

"""Load excel fixtures"""

@pytest.fixture(scope="module")
def loaded_excel_base():
    # Load the Excel file
    data_file = op.join(get_test_data_path(), "test_expinfo_correct.xlsx")
    exp_info = pd.read_excel(data_file)
    exp_info.dropna(inplace=True, how='all')
    return exp_info

@pytest.fixture(scope="module")
def loaded_excel_concat():
    # Load the Excel file
    data_file = op.join(get_test_data_path(), "test_expinfo_concat.xlsx")
    exp_info = pd.read_excel(data_file)
    exp_info.dropna(inplace=True, how='all')
    return exp_info

"""Test load/concat/transform functions"""

def test_load_excel(tmp_path):
    data_file = op.join(get_test_data_path(), "test_expinfo_correct.xlsx")
    loaded_df = load_excel(data_file)
    assert loaded_df.shape == (25,8)

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
    exp_info.dropna(inplace=True, how='all')
    with pytest.raises(SystemExit):
        exp_info = check_coordinates_are_numbers(exp_info)
    out, err = capfd.readouterr()
    assert out == "Non-numeric Coordinates in column x: [13]\n"

def test_concat_tags(loaded_excel_base):
    exp_info = concat_tags(loaded_excel_base)
    assert exp_info.shape == (25,7)
    assert list(exp_info.Tags[0]) == ['a', 'visual']

def test_concat_coordinates(loaded_excel_base):
    exp_info_firstlines = concat_coordinates(loaded_excel_base)
    assert exp_info_firstlines.shape == (3,8)
    assert exp_info_firstlines.NumberOfFoci[0] == 5

def test_convert_tal_2_mni(loaded_excel_concat):
    exp_info = convert_tal_2_mni(loaded_excel_concat)
    assert exp_info.Coordinates_mm[1][0] == np.array([10.,42.,27])
