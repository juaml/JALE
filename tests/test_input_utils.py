import pytest
import os.path as op
from ..input_utils import load_excel, check_coordinates_are_numbers
from tests.utils import get_test_data_path
import pandas as pd

"""  load_excel  """
def test_load_excel_success(tmp_path):
    data_file = op.join(get_test_data_path(), "test_expinfo_correct.xlsx")
    loaded_df = load_excel(data_file)
    assert loaded_df.shape == (25,6)

def test_load_excel_file_not_found():
    # Test loading a non-existent file
    with pytest.raises(SystemExit):
        load_excel("non_existent_file.xlsx")

def test_load_excel_invalid_file(tmp_path):
    # Create a temporary invalid file
    test_file = tmp_path / "test.txt"

    # Test that a ValueError is raised for invalid Excel files
    with pytest.raises(SystemExit):
        load_excel(test_file)


""" coordinate are numbers check"""

def test_coordinate_numbers_check_true():
    data_file = op.join(get_test_data_path(), "test_expinfo_correct.xlsx")
    exp_info = pd.read_excel(data_file)
    exp_info = data_file(check_coordinates_are_numbers)
    assert exp_info.index.to_list()[-1] == 24
