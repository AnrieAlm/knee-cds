"""
Unit tests for the red flag safety screener.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.safety.red_flags import screen_red_flags, RedFlagInput

def test_no_flags_cleared_to_proceed():
    result = screen_red_flags(RedFlagInput())
    assert result.escalate_immediately == False
    assert result.triggered_flags == []
    assert "Proceed" in result.action

def test_suspected_dvt_triggers_escalation():
    result = screen_red_flags(RedFlagInput(suspected_dvt=True))
    assert result.escalate_immediately == True
    assert any("DVT" in f for f in result.triggered_flags)

def test_fever_with_joint_pain_triggers_escalation():
    result = screen_red_flags(RedFlagInput(fever_with_joint_pain=True))
    assert result.escalate_immediately == True

def test_hot_swollen_joint_triggers_escalation():
    result = screen_red_flags(RedFlagInput(hot_swollen_joint=True))
    assert result.escalate_immediately == True

def test_fracture_triad_triggers_escalation():
    """Significant trauma + unable to weight bear + bony tenderness."""
    result = screen_red_flags(RedFlagInput(
        significant_trauma=True,
        unable_to_weight_bear=True,
        bony_tenderness=True
    ))
    assert result.escalate_immediately == True
    assert len(result.triggered_flags) == 3

def test_night_pain_triggers_escalation():
    result = screen_red_flags(RedFlagInput(night_pain_at_rest=True))
    assert result.escalate_immediately == True

def test_cancer_history_triggers_escalation():
    result = screen_red_flags(RedFlagInput(history_of_cancer=True))
    assert result.escalate_immediately == True

def test_foot_drop_triggers_escalation():
    result = screen_red_flags(RedFlagInput(foot_drop=True))
    assert result.escalate_immediately == True

def test_saddle_anaesthesia_triggers_escalation():
    result = screen_red_flags(RedFlagInput(saddle_anaesthesia=True))
    assert result.escalate_immediately == True

def test_multiple_flags_all_reported():
    result = screen_red_flags(RedFlagInput(
        suspected_dvt=True,
        fever_with_joint_pain=True,
        history_of_cancer=True
    ))
    assert result.escalate_immediately == True
    assert len(result.triggered_flags) == 3

def test_single_flag_sufficient_for_escalation():
    """Any single flag alone should trigger escalation."""
    result = screen_red_flags(RedFlagInput(unexplained_weight_loss=True))
    assert result.escalate_immediately == True

def test_action_message_on_escalation():
    result = screen_red_flags(RedFlagInput(pulseless_limb=True))
    assert "Do not proceed" in result.action
    assert "emergency" in result.action.lower()
