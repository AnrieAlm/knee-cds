"""
Unit tests for the Ottawa Knee Rule implementation.
Tests are based on published criteria from Stiell et al. (1995).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.rules.ottawa import apply_ottawa_knee_rule, OttawaInput

def test_no_criteria_met_no_xray():
    """Young patient, no tenderness, full ROM, can weight bear — no xray."""
    result = apply_ottawa_knee_rule(OttawaInput(
        age=30,
        isolated_patella_tenderness=False,
        fibula_head_tenderness=False,
        unable_to_flex_90=False,
        unable_to_weight_bear=False
    ))
    assert result.xray_indicated == False
    assert result.triggered_criteria == []

def test_age_55_triggers_xray():
    """Age 55 alone is sufficient to indicate xray."""
    result = apply_ottawa_knee_rule(OttawaInput(
        age=55,
        isolated_patella_tenderness=False,
        fibula_head_tenderness=False,
        unable_to_flex_90=False,
        unable_to_weight_bear=False
    ))
    assert result.xray_indicated == True
    assert "Age 55 or older" in result.triggered_criteria

def test_patella_tenderness_triggers_xray():
    result = apply_ottawa_knee_rule(OttawaInput(
        age=30,
        isolated_patella_tenderness=True,
        fibula_head_tenderness=False,
        unable_to_flex_90=False,
        unable_to_weight_bear=False
    ))
    assert result.xray_indicated == True
    assert "Isolated tenderness of the patella" in result.triggered_criteria

def test_fibula_head_tenderness_triggers_xray():
    result = apply_ottawa_knee_rule(OttawaInput(
        age=30,
        isolated_patella_tenderness=False,
        fibula_head_tenderness=True,
        unable_to_flex_90=False,
        unable_to_weight_bear=False
    ))
    assert result.xray_indicated == True
    assert "Tenderness at the head of the fibula" in result.triggered_criteria

def test_unable_to_flex_triggers_xray():
    result = apply_ottawa_knee_rule(OttawaInput(
        age=30,
        isolated_patella_tenderness=False,
        fibula_head_tenderness=False,
        unable_to_flex_90=True,
        unable_to_weight_bear=False
    ))
    assert result.xray_indicated == True
    assert "Inability to flex knee to 90 degrees" in result.triggered_criteria

def test_unable_to_weight_bear_triggers_xray():
    result = apply_ottawa_knee_rule(OttawaInput(
        age=30,
        isolated_patella_tenderness=False,
        fibula_head_tenderness=False,
        unable_to_flex_90=False,
        unable_to_weight_bear=True
    ))
    assert result.xray_indicated == True

def test_multiple_criteria_all_reported():
    """Multiple criteria should all appear in triggered list."""
    result = apply_ottawa_knee_rule(OttawaInput(
        age=60,
        isolated_patella_tenderness=True,
        fibula_head_tenderness=True,
        unable_to_flex_90=False,
        unable_to_weight_bear=False
    ))
    assert result.xray_indicated == True
    assert len(result.triggered_criteria) == 3

def test_age_54_does_not_trigger_age_criterion():
    """Age 54 is below threshold — age criterion should not trigger."""
    result = apply_ottawa_knee_rule(OttawaInput(
        age=54,
        isolated_patella_tenderness=False,
        fibula_head_tenderness=False,
        unable_to_flex_90=False,
        unable_to_weight_bear=False
    ))
    assert result.xray_indicated == False
