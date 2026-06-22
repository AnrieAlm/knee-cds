"""
Ottawa Knee Rule - deterministic Python implementation.
Source: Stiell et al. (1995). Implementation of the Ottawa Knee Rule.
JAMA, 274(11), 827-831.

Rule: X-ray is indicated if ANY of the following are present:
  1. Age 55 or older
  2. Isolated tenderness of the patella (no other bony tenderness)
  3. Tenderness at the head of the fibula
  4. Inability to flex to 90 degrees
  5. Inability to weight bear (4 steps) immediately after injury and in ED
"""

from dataclasses import dataclass

@dataclass
class OttawaInput:
    age: int
    isolated_patella_tenderness: bool
    fibula_head_tenderness: bool
    unable_to_flex_90: bool
    unable_to_weight_bear: bool  # immediately after injury AND at assessment

@dataclass
class OttawaResult:
    xray_indicated: bool
    triggered_criteria: list[str]
    rationale: str
    source: str = "Stiell et al. (1995). JAMA, 274(11), 827-831."

def apply_ottawa_knee_rule(data: OttawaInput) -> OttawaResult:
    """
    Applies the Ottawa Knee Rule as deterministic boolean logic.
    Returns imaging recommendation and which criteria were triggered.
    This function does not use the LLM - output is computed, not generated.
    """
    triggered = []

    if data.age >= 55:
        triggered.append("Age 55 or older")

    if data.isolated_patella_tenderness:
        triggered.append("Isolated tenderness of the patella")

    if data.fibula_head_tenderness:
        triggered.append("Tenderness at the head of the fibula")

    if data.unable_to_flex_90:
        triggered.append("Inability to flex knee to 90 degrees")

    if data.unable_to_weight_bear:
        triggered.append("Inability to weight bear (4 steps) immediately after injury and at assessment")

    xray_indicated = len(triggered) > 0

    if xray_indicated:
        rationale = (
            f"X-ray IS indicated. {len(triggered)} Ottawa criterion/criteria met: "
            + "; ".join(triggered) + "."
        )
    else:
        rationale = (
            "X-ray is NOT indicated by Ottawa Knee Rule. "
            "No criteria met. Rule has high sensitivity for fracture detection."
        )

    return OttawaResult(
        xray_indicated=xray_indicated,
        triggered_criteria=triggered,
        rationale=rationale
    )
