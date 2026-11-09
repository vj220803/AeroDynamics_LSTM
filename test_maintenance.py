# test_maintenance.py
import pytest

def get_maintenance_recommendation(prob: float) -> dict:
    """
    Maps failure probability to a maintenance recommendation tier.
    Tiers:
    - Normal (prob < 0.3): Regular operation and scheduled checks.
    - Warning (0.3 <= prob < 0.7): Increased monitoring and inspection recommended.
    - Critical (prob >= 0.7): Immediate grounding and maintenance required.
    """
    if prob < 0.0 or prob > 1.0:
        raise ValueError("Probability must be between 0.0 and 1.0")
    
    recommendation = {}
    if prob >= 0.7:
        recommendation = {
            "tier": "Critical",
            "action": "Immediate grounding and maintenance required",
            "color": "red"
        }
    elif prob >= 0.3:
        recommendation = {
            "tier": "Warning",
            "action": "Increased monitoring and inspection recommended",
            "color": "yellow"
        }
    else:
        recommendation = {
            "tier": "Normal",
            "action": "Regular operation and scheduled checks",
            "color": "green"
        }

    if prob > 0.50:
        recommendation["turbine_blade_inspection"] = "Urgent inspection recommendation: Inspect turbine blades immediately."

    return recommendation


def test_normal_tier():
    # Test probabilities below 0.3
    for p in [0.0, 0.1, 0.2, 0.29]:
        rec = get_maintenance_recommendation(p)
        assert rec["tier"] == "Normal"
        assert rec["color"] == "green"
        assert "scheduled checks" in rec["action"]


def test_warning_tier():
    # Test probabilities between 0.3 and 0.69
    for p in [0.3, 0.45, 0.5, 0.69]:
        rec = get_maintenance_recommendation(p)
        assert rec["tier"] == "Warning"
        assert rec["color"] == "yellow"
        assert "monitoring" in rec["action"]


def test_critical_tier():
    # Test probabilities 0.7 and above
    for p in [0.7, 0.85, 1.0]:
        rec = get_maintenance_recommendation(p)
        assert rec["tier"] == "Critical"
        assert rec["color"] == "red"
        assert "Immediate grounding" in rec["action"]


def test_invalid_probabilities():
    # Test out of bounds probabilities raise ValueError
    with pytest.raises(ValueError):
        get_maintenance_recommendation(-0.1)
    with pytest.raises(ValueError):
        get_maintenance_recommendation(1.05)
