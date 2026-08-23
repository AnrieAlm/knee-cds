"""
Ottawa Knee Rule - deterministic Python implementation.
Source: Stiell IG, Greenberg GH, Wells GA, McKnight RD, Cwinn AA,
Cacciotti T, McDowell I, Smith NA (1995). Derivation of a decision rule
for the use of radiography in acute knee injuries. Annals of Emergency
Medicine, 26(4), 405-413. doi:10.1016/S0196-0644(95)70106-0

Prospectively validated in Stiell et al. (1996), JAMA, 275(8), 611-615.

Derivation cohort: 1,047 adults; sensitivity 1.0 (95% CI 0.95-1.0),
specificity 0.54. Patients under 18 were excluded at derivation, so the
rule's operating characteristics are undefined in that population.

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
    source: str = "Stiell et al. (1995). Ann Emerg Med, 26(4), 405-413."

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
