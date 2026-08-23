"""
investigation_context.py

Builds the investigations block of the agent context.

Design note (dissertation-relevant):
This module enforces a deterministic provenance gate. Raw vision-model
extraction output is NEVER passed to the reasoning agent. Only records a
human clinician has confirmed against the source document (VERIFIED) or
typed directly (MANUAL) are visible downstream. This mirrors the
Ottawa/Pittsburgh pattern: a safety-relevant decision -- here, "is this
machine-derived text trustworthy enough to reason over?" -- is resolved in
plain Python, not by the model.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.constants import (
    INV_AGENT_VISIBLE_STATUSES,
    INV_MAX_FINDINGS_CHARS,
    INV_MAX_IMPRESSION_CHARS,
    INV_STATUS_EXTRACTED,
    INV_STATUS_FAILED,
    INV_STATUS_REJECTED,
    MODALITY_LABELS,
    SIDE_BILATERAL,
    SIDE_LABELS,
    SIDE_NOT_STATED,
    SIDES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_aware(value: Any) -> Optional[datetime]:
    """Normalise a stored value to a timezone-aware datetime, or None."""
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _format_date(value: Any) -> str:
    dt = _as_aware(value)
    return dt.strftime("%d %b %Y") if dt else "date not stated"


def _truncate(text: Optional[str], limit: int) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " [truncated]"


def _relative_age(study_date: Any, reference: Any) -> Optional[str]:
    """Describe how long before the assessment the study was performed."""
    study = _as_aware(study_date)
    ref = _as_aware(reference)
    if not study or not ref:
        return None

    days = (ref.date() - study.date()).days
    if days < 0:
        return "dated after this assessment"
    if days == 0:
        return "same day as this assessment"
    if days == 1:
        return "1 day before this assessment"
    if days < 42:
        return f"{days} days before this assessment"
    if days < 365:
        return f"approximately {days // 30} months before this assessment"
    return f"approximately {days // 365} year(s) before this assessment"


def _descriptor(inv: Dict[str, Any]) -> str:
    """e.g. 'MRI, left knee, 09 Aug 2026'"""
    modality = MODALITY_LABELS.get(inv.get("modality"), "Investigation")
    side = inv.get("side") or SIDE_NOT_STATED
    body_part = (inv.get("body_part") or "knee").lower()

    if side == SIDE_NOT_STATED:
        region = body_part
    else:
        region = f"{SIDE_LABELS[side].lower()} {body_part}"

    return f"{modality}, {region}, {_format_date(inv.get('study_date'))}"

def _normalise_side(value: Any) -> str:
    """Map either storage convention onto the SIDE_* constants.

    History stores the involved side lowercase from a form field.
    Investigations store the SIDE_* constants uppercase. Both are
    normalised here so the comparison cannot silently fail.
    """
    if not isinstance(value, str):
        return SIDE_NOT_STATED
    upper = value.strip().upper()
    return upper if upper in SIDES else SIDE_NOT_STATED


def side_conflicts(inv: Dict[str, Any], involved_side: Any) -> bool:
    """True when a study is of a limb other than the one being assessed.

    Deterministic relevance gate. The provenance gate establishes that a
    transcription is faithful to its source document. It does not establish
    that the document concerns this patient's affected limb. Those are
    different properties and are checked separately.

    Absent or bilateral values on either side are not treated as conflicts,
    because an unknown side is not evidence of a mismatch.
    """
    study = _normalise_side(inv.get("side"))
    involved = _normalise_side(involved_side)

    if study in (SIDE_NOT_STATED, SIDE_BILATERAL):
        return False
    if involved in (SIDE_NOT_STATED, SIDE_BILATERAL):
        return False
    return study != involved
# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def all_investigations(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = case.get("investigations")
    return list(value) if isinstance(value, list) else []


def visible_investigations(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Records cleared to reach the agent. This is the provenance gate."""
    return [
        inv for inv in all_investigations(case)
        if inv.get("extraction_status") in INV_AGENT_VISIBLE_STATUSES
    ]


def pending_verification(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extracted but not yet confirmed. Drives the UI banner."""
    return [
        inv for inv in all_investigations(case)
        if inv.get("extraction_status") == INV_STATUS_EXTRACTED
    ]


def problem_investigations(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Failed or rejected uploads needing manual entry."""
    return [
        inv for inv in all_investigations(case)
        if inv.get("extraction_status") in (INV_STATUS_FAILED, INV_STATUS_REJECTED)
    ]


def has_blocking_state(case: Dict[str, Any]) -> bool:
    """True if anything is attached that the agent cannot yet see."""
    return bool(pending_verification(case) or problem_investigations(case))


# ---------------------------------------------------------------------------
# Agent context
# ---------------------------------------------------------------------------

def build_investigation_context(case: Dict[str, Any]) -> str:
    """
    Render the investigations block for the agent prompt.

    Returns a plain-text block. Returns an explicit 'none available' line
    rather than an empty string, so the agent can distinguish "no prior
    imaging" from "imaging section not reached yet".

    Two gates apply. The provenance gate (visible_investigations) decides
    whether machine-derived text is trustworthy enough to reason over. The
    laterality gate (side_conflicts) decides whether a trustworthy record
    describes the limb under assessment. Both are resolved here in plain
    Python rather than left to the model.
    """
    visible = visible_investigations(case)
    withheld = len(pending_verification(case))

    involved_side = (case.get("history") or {}).get("involved_side")
    mismatched = [inv for inv in visible if side_conflicts(inv, involved_side)]
    visible = [inv for inv in visible if not side_conflicts(inv, involved_side)]

    lines: List[str] = ["PRIOR INVESTIGATIONS"]

    if not visible:
        if mismatched:
            lines.append(
                "No verified prior investigations of the limb under "
                "assessment are available for this case."
            )
        else:
            lines.append(
                "No verified prior investigations are available for this case."
            )

        for inv in mismatched:
            lines.append(
                f"({_descriptor(inv)} is attached to this case but concerns "
                "the other limb. It has been withheld as it does not describe "
                "the limb under assessment.)"
            )

        if withheld:
            lines.append(
                f"({withheld} attached document(s) are awaiting clinician "
                "verification and are deliberately withheld from you. Do not "
                "speculate about their contents.)"
            )

        lines.append(
            "Reason from the clinical history and physical examination alone."
        )
        return "\n".join(lines)

    lines.append(
        "The findings below are transcribed from radiology reports and have "
        "been confirmed by the assessing clinician. Treat them as reported "
        "findings, not as your own interpretation. You have not seen any "
        "images and must not describe or infer image appearances."
    )

    reference = case.get("created_at")

    for index, inv in enumerate(visible, start=1):
        lines.append("")
        header = f"{index}. {_descriptor(inv)}"
        age = _relative_age(inv.get("study_date"), reference)
        if age:
            header += f" ({age})"
        lines.append(header)

        findings = _truncate(inv.get("report_findings"), INV_MAX_FINDINGS_CHARS)
        impression = _truncate(inv.get("report_impression"), INV_MAX_IMPRESSION_CHARS)

        if findings:
            lines.append(f"   Reported findings: {findings}")
        if impression:
            lines.append(f"   Reported impression: {impression}")
        if not findings and not impression:
            lines.append("   No report text recorded.")

    lines.append("")
    lines.append(
        "Reconcile these reported findings with the physical examination. "
        "Where they disagree, say so plainly rather than resolving the "
        "conflict yourself. Deterministic safety rules are unaffected by "
        "prior imaging and remain in force."
    )

    for inv in mismatched:
        lines.append(
            f"({_descriptor(inv)} is attached to this case but concerns the "
            "other limb. It has been withheld as it does not describe the "
            "limb under assessment.)"
        )

    if withheld:
        lines.append(
            f"({withheld} further document(s) are awaiting verification and "
            "have been withheld.)"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary / template support
# ---------------------------------------------------------------------------

def investigation_summary_rows(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rows for the Investigations section of the case summary template.

    Carries both gates through to the template. A record the agent could not
    see is shown as withheld rather than hidden, so the clinician can tell
    that an attached document was set aside and why. The summary therefore
    reports the same set of findings the agent was given, which is the
    property that makes the page usable as a record of the assessment.
    """
    rows: List[Dict[str, Any]] = []
    involved_side = (case.get("history") or {}).get("involved_side")

    for inv in all_investigations(case):
        verified = inv.get("extraction_status") in INV_AGENT_VISIBLE_STATUSES
        other_limb = side_conflicts(inv, involved_side)
        rows.append({
            "inv_id": inv.get("inv_id"),
            "descriptor": _descriptor(inv),
            "status": inv.get("extraction_status"),
            "findings": inv.get("report_findings") or "",
            "impression": inv.get("report_impression") or "",
            "has_source_file": bool(inv.get("file_ref")),
            "original_filename": inv.get("original_filename"),
            "other_limb": other_limb,
            "agent_visible": verified and not other_limb,
        })

    return rows


def investigation_citation(inv: Dict[str, Any]) -> Dict[str, str]:
    """Citation chip payload for a reported finding."""
    return {
        "source_type": "INVESTIGATION",
        "label": _descriptor(inv),
        "detail": "Radiology report attached to this case",
    }

def build_investigation_context_from_list(
    investigations: Optional[List[Dict[str, Any]]],
    reference_date: Any = None,
    involved_side: Any = None,
) -> str:
    """Agent-facing entry point.

    Takes the investigations list directly rather than the case document,
    mirroring how physical_dict is passed to the orchestrator. Keeps the
    agent layer a pure function of what it is given, with no database read
    of its own.
    """
    shim = {
        "investigations": investigations or [],
        "created_at": reference_date,
        "history": {"involved_side": involved_side},
    }
    return build_investigation_context(shim)