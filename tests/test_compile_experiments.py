from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from jale.core.utils.compile_experiments import (
    compile_experiments,
)


@pytest.fixture
def sample_tasks():
    return pd.DataFrame(
        {
            "Name": ["pain", "touch", "vision", "emotion"],
            "ExpIndex": [[1, 2], [3, 4], [5], [1, 3, 5, 7]],
        }
    )


def test_include_tag(sample_tasks):
    result, masks, names = compile_experiments(["+pain"], sample_tasks)
    assert result == [1, 2], "Should return the matches for 'pain'"


def test_include_tag_single_exp(sample_tasks):
    result, masks, names = compile_experiments(["+vision"], sample_tasks)
    assert result == [5], "Should return the only match for 'vision'"


def test_exclude_tag(sample_tasks):
    result, masks, names = compile_experiments(["+pain", "-pain"], sample_tasks)
    assert result == [], "Exclusion should remove all included experiments"


def test_logical_or(sample_tasks):
    result, masks, names = compile_experiments(["+pain", "+touch", "?"], sample_tasks)
    assert set(result) == {1, 2, 3, 4}, "Logical OR should return union of both"


def test_no_matching_tag(sample_tasks):
    with pytest.raises(ValueError, match="No experiments found for tag: fail"):
        compile_experiments(["+fail"], sample_tasks)


def test_bad_combination_raises(sample_tasks):
    with pytest.raises(ValueError, match="Bad tag combination"):
        compile_experiments(["+pain", "+touch"], sample_tasks)


@patch("jale.core.utils.compile_experiments.nb")
def test_mask_loading(mock_nb, sample_tasks):
    dummy_mask = np.array([[[0, 1], [1, 0]]])
    mock_img = MagicMock()
    mock_img.get_fdata.return_value = dummy_mask
    mock_nb.load.return_value = mock_img

    result, masks, names = compile_experiments(
        ["+pain", "?", "$dummy.nii.gz"], sample_tasks
    )

    assert result == [1, 2]
    assert masks[0].dtype == bool
    assert names[0] == "dummy"


@patch("jale.core.utils.compile_experiments.nb")
def test_mask_file_not_found(mock_nb, sample_tasks):
    mock_nb.load.side_effect = FileNotFoundError

    with pytest.raises(FileNotFoundError, match="Mask file not found"):
        compile_experiments(["+pain", "?", "$missing.nii.gz"], sample_tasks)
