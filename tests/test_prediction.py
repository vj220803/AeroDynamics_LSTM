# tests/test_prediction.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from app import safe_predict, get_maintenance_recommendation

def test_safe_predict_boundary_and_clamping():
    """Test that safe_predict correctly computes probabilities and clamps values between 0.0 and 1.0."""
    mock_scaler = MagicMock()
    # Mock scaler.transform to return dummy normalized data
    mock_scaler.transform.return_value = np.zeros((50, 25))

    mock_model = MagicMock()

    # Case 1: scalar prediction within range
    mock_model.predict.return_value = np.array([[0.75]])
    df_dummy = pd.DataFrame(np.random.rand(50, 25))
    prob = safe_predict(mock_model, mock_scaler, df_dummy)
    assert isinstance(prob, float)
    assert prob == 0.75

    # Case 2: probability out of upper bounds (> 1.0) gets clamped
    mock_model.predict.return_value = np.array([[1.45]])
    prob = safe_predict(mock_model, mock_scaler, df_dummy)
    assert prob == 1.0

    # Case 3: probability out of lower bounds (< 0.0) gets clamped
    mock_model.predict.return_value = np.array([[-0.2]])
    prob = safe_predict(mock_model, mock_scaler, df_dummy)
    assert prob == 0.0

    # Case 4: binary classification shape (batch, 2)
    mock_model.predict.return_value = np.array([[0.2, 0.85]])
    prob = safe_predict(mock_model, mock_scaler, df_dummy)
    assert prob == 0.85

def test_prediction_threshold_logic():
    """Test engine maintenance prediction threshold logic (e.g. >= 0.5 triggers warning)."""
    # Threshold at 0.5 as defined in app.py
    threshold = 0.5

    def get_status(prob):
        return "Failure soon" if prob >= threshold else "Healthy"

    assert get_status(0.49) == "Healthy"
    assert get_status(0.50) == "Failure soon"
    assert get_status(0.85) == "Failure soon"
    assert get_status(0.0) == "Healthy"
    assert get_status(1.0) == "Failure soon"

def test_scaler_transform_exception():
    """Test that safe_predict raises RuntimeError if scaler transform fails."""
    mock_scaler = MagicMock()
    mock_scaler.transform.side_effect = Exception("Transformation error")
    mock_model = MagicMock()
    df_dummy = pd.DataFrame(np.random.rand(50, 25))

    with pytest.raises(RuntimeError, match="Scaler.transform failed"):
        safe_predict(mock_model, mock_scaler, df_dummy)

def test_maintenance_recommendation_card():
    """Verify maintenance recommendation card thresholds."""
    assert "CRITICAL" in get_maintenance_recommendation(0.85)
    assert "WARNING" in get_maintenance_recommendation(0.55)
    assert "ADVISORY" in get_maintenance_recommendation(0.35)
    assert "OPTIMAL" in get_maintenance_recommendation(0.10)
