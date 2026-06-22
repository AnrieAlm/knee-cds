"""
Red Flag Safety Screener - deterministic Python implementation.
Runs before any agent reasoning on every case, unconditionally.
The LLM cannot override or bypass this layer.

Red flags are clinical indicators of serious pathology requiring
immediate escalation rather than physiotherapy management.

Sources:
- Magee, D.J. (2014). Orthopedic Physical Assessment (6th ed.). Saunders/Elsevier.
- NICE Clinical Guidelines (various)
"""

from dataclasses import dataclass, field

@dataclass
class RedFlagInput:
    # Vascular
    suspected_dvt: bool = False           # Calf swelling, redness, warmth, pain
    pulseless_limb: bool = False          # Absent distal pulse

    # Infectious / Inflammatory
    fever_with_joint_pain: bool = False   # Temp >38C with acute joint pain
    hot_swollen_joint: bool = False       # Erythema, heat, effusion — septic joint
    recent_infection: bool = False        # Recent systemic infection + joint pain

    # Fracture indicators
    significant_trauma: bool = False      # High energy mechanism
    unable_to_weight_bear: bool = False   # Cannot take 4 steps
    bony_tenderness: bool = False         # Direct bony point tenderness

    # Neoplastic
    unexplained_weight_loss: bool = False # >10% body weight, unexplained
    night_pain_at_rest: bool = False      # Pain waking from sleep, not positional
    history_of_cancer: bool = False       # Prior malignancy + new joint pain

    # Neurological
    foot_drop: bool = False              # Inability to dorsiflex foot
    saddle_anaesthesia: bool = False     # Perineal numbness — cauda equina

@dataclass
class RedFlagResult:
    escalate_immediately: bool
    triggered_flags: list[str] = field(default_factory=list)
    rationale: str = ""
    action: str = ""

def screen_red_flags(data: RedFlagInput) -> RedFlagResult:
    """
    Screens for serious pathology indicators.
    Runs first on every case before any agent reasoning.
    This function does not use the LLM — output is computed, not generated.
    If ANY flag is triggered, system halts and recommends escalation.
    """
    triggered = []

    # Vascular
    if data.suspected_dvt:
        triggered.append("Suspected DVT (calf swelling, redness, warmth)")
    if data.pulseless_limb:
        triggered.append("Absent distal pulse — possible vascular emergency")

    # Infectious
    if data.fever_with_joint_pain:
        triggered.append("Fever with acute joint pain — possible septic arthritis")
    if data.hot_swollen_joint:
        triggered.append("Hot, swollen, erythematous joint — possible septic arthritis")
    if data.recent_infection:
        triggered.append("Recent systemic infection with new joint pain")

    # Fracture
    if data.significant_trauma:
        triggered.append("Significant trauma mechanism")
    if data.unable_to_weight_bear:
        triggered.append("Unable to weight bear (4 steps)")
    if data.bony_tenderness:
        triggered.append("Bony point tenderness on palpation")

    # Neoplastic
    if data.unexplained_weight_loss:
        triggered.append("Unexplained weight loss")
    if data.night_pain_at_rest:
        triggered.append("Night pain at rest — waking from sleep")
    if data.history_of_cancer:
        triggered.append("History of cancer with new joint pain")

    # Neurological
    if data.foot_drop:
        triggered.append("Foot drop — neurological emergency")
    if data.saddle_anaesthesia:
        triggered.append("Saddle anaesthesia — possible cauda equina syndrome")

    escalate = len(triggered) > 0

    if escalate:
        rationale = (
            f"ESCALATE IMMEDIATELY. {len(triggered)} red flag(s) detected: "
            + "; ".join(triggered) + "."
        )
        action = (
            "Do not proceed with physiotherapy assessment. "
            "Refer to emergency department or appropriate medical team immediately. "
            "Document findings and time of referral."
        )
    else:
        rationale = (
            "No red flags detected. "
            "Safe to proceed with physiotherapy assessment and agent reasoning."
        )
        action = "Proceed with clinical reasoning."

    return RedFlagResult(
        escalate_immediately=escalate,
        triggered_flags=triggered,
        rationale=rationale,
        action=action
    )
