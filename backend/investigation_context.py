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
    SIDE_LABELS,
    SIDE_NOT_STATED,
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
    """
    visible = visible_investigations(case)
    withheld = len(pending_verification(case)) + len(problem_investigations(case))

    lines: List[str] = ["PRIOR INVESTIGATIONS"]

    if not visible:
        lines.append(
            "No verified prior investigations are available for this case."
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
    """Rows for the Investigations section of the case summary template."""
    rows: List[Dict[str, Any]] = []

    for inv in all_investigations(case):
        rows.append({
            "inv_id": inv.get("inv_id"),
            "descriptor": _descriptor(inv),
            "status": inv.get("extraction_status"),
            "findings": inv.get("report_findings") or "",
            "impression": inv.get("report_impression") or "",
            "has_source_file": bool(inv.get("file_ref")),
            "original_filename": inv.get("original_filename"),
            "agent_visible": inv.get("extraction_status") in INV_AGENT_VISIBLE_STATUSES,
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
    }
    return build_investigation_context(shim)


def build_investigation_context_from_list(
    investigations: Optional[List[Dict[str, Any]]],
    reference_date: Any = None,
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
    }
    return build_investigation_context(shim)
