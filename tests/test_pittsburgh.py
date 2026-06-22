"""
Unit tests for the Pittsburgh Knee Rule implementation.
Tests are based on published criteria from Seaberg & Jackson (1994).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.rules.pittsburgh import apply_pittsburgh_knee_rule, PittsburghInput

def test_no_mechanism_no_xray():
    """Without blunt trauma or fall mechanism, rule does not apply."""
    result = apply_pittsburgh_knee_rule(PittsburghInput(
        mechanism_blunt_trauma_or_fall=False,
        age=60,
        unable_to_weight_bear=True
    ))
    assert result.xray_indicated == False

def test_mechanism_plus_age_over_50_triggers_xray():
    result = apply_pittsburgh_knee_rule(PittsburghInput(
        mechanism_blunt_trauma_or_fall=True,
        age=51,
        unable_to_weight_bear=False
    ))
    assert result.xray_indicated == True
    assert "Age over 50" in result.triggered_criteria

def test_mechanism_plus_age_under_12_triggers_xray():
    result = apply_pittsburgh_knee_rule(PittsburghInput(
        mechanism_blunt_trauma_or_fall=True,
        age=10,
        unable_to_weight_bear=False
    ))
    assert result.xray_indicated == True
    assert "Age under 12" in result.triggered_criteria

def test_mechanism_plus_weight_bear_triggers_xray():
    result = apply_pittsburgh_knee_rule(PittsburghInput(
        mechanism_blunt_trauma_or_fall=True,
        age=30,
        unable_to_weight_bear=True
    ))
    assert result.xray_indicated == True
    assert "Unable to weight bear 4 steps in ED" in result.triggered_criteria

def test_mechanism_alone_not_sufficient():
    """Mechanism present but age 12-50 and can weight bear — no xray."""
    result = apply_pittsburgh_knee_rule(PittsburghInput(
        mechanism_blunt_trauma_or_fall=True,
        age=30,
        unable_to_weight_bear=False
    ))
    assert result.xray_indicated == False

def test_age_boundary_exactly_50_no_trigger():
    """Age exactly 50 does not meet 'over 50' criterion."""
    result = apply_pittsburgh_knee_rule(PittsburghInput(
        mechanism_blunt_trauma_or_fall=True,
        age=50,
        unable_to_weight_bear=False
    ))
    assert result.xray_indicated == False

def test_age_boundary_exactly_12_no_trigger():
    """Age exactly 12 does not meet 'under 12' criterion."""
    result = apply_pittsburgh_knee_rule(PittsburghInput(
        mechanism_blunt_trauma_or_fall=True,
        age=12,
        unable_to_weight_bear=False
    ))
    assert result.xray_indicated == False
