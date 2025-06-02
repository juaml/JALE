import pathlib
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import jale.core.utils.input as io


# --- Test load_config ---
def test_load_config_success():
    current_dir = pathlib.Path(__file__).parent
    config_path = current_dir / "config.yml"
    cfg = io.load_config(config_path)
    assert isinstance(cfg, dict)
    assert "project" in cfg and "parameters" in cfg


def test_load_config_file_not_found():
    with pytest.raises(SystemExit):
        io.load_config("nonexistent_file.yml")


def test_load_config_bad_yaml(tmp_path):
    bad_yaml = "this: [unbalanced brackets"
    file = tmp_path / "bad.yml"
    file.write_text(bad_yaml)
    with pytest.raises(SystemExit):
        io.load_config(file)


# --- Test load_experiment_file ---
@patch("pandas.read_excel")
def test_load_experiment_file_excel(mock_read_excel):
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9],
            "D": [10, 11, 12],
            "E": [13, 14, 15],
            "F": [16, 17, 18],
        }
    )
    mock_read_excel.return_value = df
    result = io.load_experiment_file("file.xlsx")
    assert "Articles" in result.columns
    assert "x" in result.columns
    assert result.shape[0] == 3


def test_load_experiment_file_unsupported_format():
    with pytest.raises(SystemExit):
        io.load_experiment_file("file.unsupported")


@patch("pandas.read_excel")
def test_load_experiment_file_with_bad_rows(mock_read_excel):
    # DataFrame with row having only 1 or 2 non-NaN entries triggers exit
    df = pd.DataFrame(
        {
            "A": [1, None, None],
            "B": [None, None, 3],
            "C": [None, None, None],
            "D": [None, None, None],
            "E": [None, None, None],
            "F": [None, None, None],
        }
    )
    mock_read_excel.return_value = df
    with pytest.raises(SystemExit):
        io.load_experiment_file("file.xlsx")


# --- Test check_coordinates_are_numbers ---
def test_check_coordinates_are_numbers_valid():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "z": [5.0, 6.0],
        }
    )
    result = io.check_coordinates_are_numbers(df)
    assert result.index.equals(pd.RangeIndex(start=0, stop=2))


def test_check_coordinates_are_numbers_invalid():
    df = pd.DataFrame(
        {
            "x": [1.0, "bad"],
            "y": [3.0, 4.0],
            "z": [5.0, 6.0],
        }
    )
    with pytest.raises(SystemExit):
        io.check_coordinates_are_numbers(df)


# --- Test concat_tags ---
def test_concat_tags_basic():
    df = pd.DataFrame(
        {
            "Articles": ["art1", "art2"],
            "Subjects": [10, 20],
            "x": [1, 2],
            "y": [3, 4],
            "z": [5, 6],
            "CoordinateSpace": ["MNI", "TAL"],
            "Tag1": ["A", None],
            "Tag2": ["B", "C"],
        }
    )
    out = io.concat_tags(df)
    assert "Tags" in out.columns
    assert out.loc[0, "Tags"] == ("a", "b")
    assert out.loc[1, "Tags"] == ("c",)


# --- Test concat_coordinates ---
def test_concat_coordinates_pool_true():
    df = pd.DataFrame(
        {
            "Articles": ["art1", "art1"],
            "Subjects": [10, 10],
            "x": [1, 2],
            "y": [3, 4],
            "z": [5, 6],
            "CoordinateSpace": ["MNI", "MNI"],
            "Tags": [("tag1",), ("tag1",)],
        }
    )
    out = io.concat_coordinates(df, pool_experiments=True)
    assert "Coordinates_mm" in out.columns
    assert out.iloc[0]["NumberOfFoci"] == 2
    assert isinstance(out.iloc[0]["Coordinates_mm"], np.ndarray)


def test_concat_coordinates_pool_false():
    df = pd.DataFrame(
        {
            "Articles": ["art1", "art1"],
            "Subjects": [10, 10],
            "x": [1, 2],
            "y": [3, 4],
            "z": [5, 6],
            "CoordinateSpace": ["MNI", "MNI"],
            "Tags": [("tag1",), ("tag2",)],
        }
    )
    out = io.concat_coordinates(df, pool_experiments=False)
    assert out.shape[0] == 2


# --- Test convert_tal_2_mni ---
def test_convert_tal_2_mni_calls_tal2icbm_spm(monkeypatch):
    df = pd.DataFrame(
        {
            "CoordinateSpace": ["TAL", "MNI"],
            "Coordinates_mm": [np.array([[1, 2, 3]]), np.array([[4, 5, 6]])],
        }
    )

    print("Before transform:", df.loc[0, "Coordinates_mm"])
    called = {}

    def fake_tal2icbm_spm(coords):
        called["called"] = True
        return coords + 1  # dummy transform

    monkeypatch.setattr(io, "tal2icbm_spm", fake_tal2icbm_spm)
    out = df.copy(deep=True)
    out = io.convert_tal_2_mni(out)
    print("After transform:", df.loc[0, "Coordinates_mm"])
    assert called.get("called") is True
    assert np.all(out.loc[0, "Coordinates_mm"] == df.loc[0, "Coordinates_mm"] + 1)
    assert np.all(out.loc[1, "Coordinates_mm"] == df.loc[1, "Coordinates_mm"])


# --- Test transform_coordinates_to_voxel_space ---
def test_transform_coordinates_to_voxel_space_basic():
    df = pd.DataFrame(
        {"Coordinates_mm": [np.array([[0, 0, 0]]), np.array([[10, 20, 30]])]}
    )
    out = io.transform_coordinates_to_voxel_space(df)
    assert "Coordinates" in out.columns
    assert isinstance(out.iloc[0]["Coordinates"], np.ndarray)


# --- Test create_tasks_table ---
def test_create_tasks_table_basic():
    df = pd.DataFrame(
        {
            "Tags": [("task1", "task2"), ("task1",)],
            "Articles": ["art1", "art2"],
            "Subjects": [10, 20],
        }
    )
    tasks = io.create_tasks_table(df)
    assert "Name" in tasks.columns
    assert "Num_Exp" in tasks.columns
    assert "all" in tasks["Name"].values


# --- Test check_params ---
def test_check_params_cutoff_enabled():
    params = {"cutoff_predict_enabled": True}
    out = io.check_params(params)
    assert out["significance_threshold"] == 0.05
    assert out["cluster_forming_threshold"] == 0.001
    assert out["monte_carlo_iterations"] == 5000


def test_check_params_cutoff_disabled():
    params = {"cutoff_predict_enabled": False}
    out = io.check_params(params)
    # Should not ioify parameters
    assert "significance_threshold" not in out


# --- Test check_for_exp_independence ---
def test_check_for_exp_independence_warns(caplog):
    df = pd.DataFrame(
        {
            "Articles": ["art1", "art1", "art2"],
            "Tags": [("a",), ("b",), ("c",)],
        }
    )
    with caplog.at_level("WARNING"):
        io.check_for_exp_independence(df)
        assert "multiple experiments" in caplog.text


def test_check_for_exp_independence_no_warn(caplog):
    df = pd.DataFrame(
        {
            "Articles": ["art1", "art2"],
            "Tags": [("a",), ("b",)],
        }
    )
    with caplog.at_level("WARNING"):
        io.check_for_exp_independence(df)
        assert "multiple experiments" not in caplog.text
