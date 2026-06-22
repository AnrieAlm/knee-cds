"""
Pittsburgh Knee Rule - deterministic Python implementation.
Source: Seaberg & Jackson (1994). Clinical decision rule for knee radiographs.
American Journal of Emergency Medicine, 12(5), 541-543.

Rule: X-ray is indicated if EITHER condition is met:
  Condition A — Mechanism: blunt trauma OR fall
    AND one of:
      - Age under 12
      - Age over 50
      - Unable to weight bear (4 steps) in ED

  The Pittsburgh rule is narrower than Ottawa — mechanism is required.
"""

from dataclasses import dataclass

@dataclass
class PittsburghInput:
    mechanism_blunt_trauma_or_fall: bool
    age: int
    unable_to_weight_bear: bool  # 4 steps in ED

@dataclass
class PittsburghResult:
    xray_indicated: bool
    triggered_criteria: list[str]
    rationale: str
    source: str = "Seaberg & Jackson (1994). Am J Emerg Med, 12(5), 541-543."

def apply_pittsburgh_knee_rule(data: PittsburghInput) -> PittsburghResult:
    """
    Applies the Pittsburgh Knee Rule as deterministic boolean logic.
    Returns imaging recommendation and which criteria were triggered.
    This function does not use the LLM - output is computed, not generated.
    """
    triggered = []

    if not data.mechanism_blunt_trauma_or_fall:
        return PittsburghResult(
            xray_indicated=False,
            triggered_criteria=[],
            rationale=(
                "X-ray is NOT indicated by Pittsburgh Knee Rule. "
                "Mechanism is not blunt trauma or fall — rule does not apply."
            )
        )

    triggered.append("Mechanism: blunt trauma or fall")

    if data.age < 12:
        triggered.append("Age under 12")
    elif data.age > 50:
        triggered.append("Age over 50")

    if data.unable_to_weight_bear:
        triggered.append("Unable to weight bear 4 steps in ED")

    # Mechanism alone is not enough — need age criterion OR weight bear
    age_criterion = data.age < 12 or data.age > 50
    xray_indicated = data.mechanism_blunt_trauma_or_fall and (
        age_criterion or data.unable_to_weight_bear
    )

    if xray_indicated:
        rationale = (
            f"X-ray IS indicated. Pittsburgh criteria met: "
            + "; ".join(triggered) + "."
        )
    else:
        rationale = (
            "X-ray is NOT indicated by Pittsburgh Knee Rule. "
            "Mechanism present but no age or weight-bearing criterion met."
        )

    return PittsburghResult(
        xray_indicated=xray_indicated,
        triggered_criteria=triggered,
        rationale=rationale
    )
