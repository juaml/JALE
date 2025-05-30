import sys
from unittest import mock

import numpy as np
import pytest

import jale.core.utils.cutoff_prediction as cp


def test_feature_extraction_basic(monkeypatch):
    nexp = 3
    nsub = np.array([10, 20, 30])
    nfoci = np.array([5, 10, 15])

    # Patch logger.warning to track calls
    monkeypatch.setattr(cp.logger, "warning", lambda msg: None)

    features = cp.feature_extraction(nexp, nsub, nfoci)
    assert isinstance(features, np.ndarray)
    assert features.shape[0] == 1  # Because np.c_ adds a new 2D array with shape (1, N)
    # Check some expected values roughly
    assert features[0, 0] == nexp
    assert features[0, 1] == np.sum(nsub)
    assert features[0, 9] == np.sum(nfoci)


def test_feature_extraction_exit_on_large_subjects(monkeypatch):
    nexp = 2
    nsub = np.array([100, 301])  # 301 triggers exit
    nfoci = np.array([5, 5])

    # Patch logger.warning and sys.exit
    called = {}

    def fake_exit():
        called["exit"] = True
        raise SystemExit()

    monkeypatch.setattr(
        cp.logger, "warning", lambda msg: called.setdefault("warn", msg)
    )
    monkeypatch.setattr(sys, "exit", fake_exit)

    with pytest.raises(SystemExit):
        cp.feature_extraction(nexp, nsub, nfoci)
    assert "warn" in called
    assert "exit" in called


def test_feature_extraction_warn_on_large_foci(monkeypatch):
    nexp = 10
    nsub = np.array([10, 15, 5, 3, 10, 5, 8, 10, 20, 10])
    nfoci = np.array(
        [100, 200, 50, 25, 100, 50, 75, 100, 200, 100]
    )  # 200 triggers warning

    warnings = []
    monkeypatch.setattr(cp.logger, "warning", lambda msg: warnings.append(msg))

    features = cp.feature_extraction(nexp, nsub, nfoci)
    assert any("Out-Of-Distribution" in w for w in warnings)
    assert features.shape[1] == 26  # Number of features


@mock.patch("jale.core.utils.cutoff_prediction.xgb.XGBRegressor")
def test_predict_cutoff(mock_xgb, monkeypatch):
    # Setup mock model and predict behavior
    instance = mock_xgb.return_value
    instance.load_model.return_value = None
    instance.predict.return_value = np.array([1.5])

    import pandas as pd

    # Create fake DataFrame with expected columns
    exp_df = pd.DataFrame({"Subjects": [10, 20, 30], "NumberOfFoci": [5, 10, 15]})

    # Patch feature_extraction to return a fixed feature vector
    monkeypatch.setattr(
        cp, "feature_extraction", lambda nexp, nsub, nfoci: np.array([[3] * 26])
    )

    # Patch logger.warning to detect warnings
    warnings = []
    monkeypatch.setattr(cp.logger, "warning", lambda msg: warnings.append(msg))

    vfwe, cfwe, tfce = cp.predict_cutoff(exp_df)

    # Validate mocked predict calls
    instance.load_model.assert_called()
    instance.predict.assert_called()

    # Check outputs are floats or numpy floats
    assert isinstance(vfwe[0], (float, np.floating))
    assert isinstance(cfwe[0], (float, np.floating))
    assert isinstance(tfce[0], (float, np.floating))

    # Check cutoff is from mock predict return (1.5 or rounded)
    assert np.isclose(vfwe, 1.5)
    assert cfwe == 2  # Rounded from 1.5
    assert np.isclose(tfce, 1.5)

    # No warnings for nexp <= 150
    assert len(warnings) == 0


def test_predict_cutoff_warn_large_nexp(monkeypatch):
    import pandas as pd

    exp_df = pd.DataFrame({"Subjects": [10] * 200, "NumberOfFoci": [5] * 200})

    monkeypatch.setattr(
        cp.logger,
        "warning",
        lambda msg: setattr(test_predict_cutoff_warn_large_nexp, "warned", True),
    )
    monkeypatch.setattr(
        cp, "feature_extraction", lambda nexp, nsub, nfoci: np.array([[200] * 26])
    )

    # Mock xgb model
    with mock.patch("jale.core.utils.cutoff_prediction.xgb.XGBRegressor") as mock_xgb:
        instance = mock_xgb.return_value
        instance.load_model.return_value = None
        instance.predict.return_value = np.array([1.0])

        test_predict_cutoff_warn_large_nexp.warned = False
        cp.predict_cutoff(exp_df)
        assert test_predict_cutoff_warn_large_nexp.warned is True
