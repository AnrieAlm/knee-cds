"""
backend/investigation_routes.py

Routes for the Investigations tab.

Pipeline position: after Physical Examination, before Summary.

The ordering is deliberate and is a design decision, not a UI convenience.
Showing a junior clinician a radiology impression before they have recorded
their own examination findings turns the examination into a confirmatory
exercise. Extraction and display are therefore gated on the physical
examination having been submitted. Files may be attached at any time.

Two invariants are enforced here and nowhere else:
  1. Only an authenticated physiotherapist action can set VERIFIED.
     No model output path can reach that transition.
  2. Admin accounts are read-only at the route level, not merely in the UI.
"""

import os
from groq import Groq
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from gridfs import GridFS
from starlette.responses import Response

from backend.constants import (
    INV_MAX_FINDINGS_CHARS,
    INV_MAX_IMPRESSION_CHARS,
    INV_MAX_PER_CASE,
    INV_STATUS_EXTRACTED,
    INV_STATUS_LABELS,
    INV_STATUS_MANUAL,
    INV_STATUS_VERIFIED,
    INV_VISUAL_CLASS,
    MODALITIES,
    MODALITY_LABELS,
    SIDE_LABELS,
    SIDES,
    is_valid_modality,
    is_valid_side,
)
from backend.investigation_context import (
    all_investigations,
    pending_verification,
    problem_investigations,
)
from backend.investigation_extract import (
    UploadRejected,
    extract_investigation,
    manual_investigation_record,
)
from backend.auth import can_view_case, require_current_user, require_physio
from backend.db import get_db
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cases", tags=["investigations"])


_groq_client = None


def _cases():
    return get_db()["cases"]


def _fs():
    return GridFS(get_db(), collection="investigation_files")


def _groq():
    """Raw Groq SDK client. The agent uses ChatGroq; extraction needs the
    plain client for vision input and JSON mode."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _object_id(case_id: str) -> ObjectId:
    try:
        return ObjectId(case_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=404, detail="Case not found")


def _load_case(case_id: str, user: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch a case the current user is entitled to see.

    Case documents carry both an ObjectId _id and a separate string id,
    and templates link using the latter. Both are accepted so this tab
    resolves the same identifiers as every other tab.

    Read access is delegated to auth.can_view_case (owner, or any admin).
    Write access is enforced separately by require_physio on each mutating
    route, so an admin reaching this function can read but never modify.
    """
    case = _cases().find_one({"id": case_id})

    if case is None:
        try:
            case = _cases().find_one({"_id": ObjectId(case_id)})
        except (InvalidId, TypeError):
            case = None

    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")

    if not can_view_case(user, case):
        raise HTTPException(status_code=403, detail="Not permitted")

    return case


def _exam_submitted(case: Dict[str, Any]) -> bool:
    """Physical examination must be recorded before investigations open.

    There is no explicit completion flag on the case document; the
    'physical' subdocument is written only by the examination POST route,
    so its presence with at least one recorded value is the completion
    signal. This is checked rather than assumed because the ordering is a
    design decision, not a UI convenience: showing a radiology impression
    before the clinician has recorded their own findings turns the
    examination into a confirmatory exercise.
    """
    physical = case.get("physical") or {}
    if not isinstance(physical, dict):
        return False
    return any(
        value not in (None, "", [], {})
        for value in physical.values()
    )


def _require_exam_submitted(case: Dict[str, Any]) -> None:
    if not _exam_submitted(case):
        raise HTTPException(
            status_code=409,
            detail=(
                "Complete and submit the physical examination before "
                "recording prior investigations."
            ),
        )


def _find_investigation(case: Dict[str, Any], inv_id: str) -> Dict[str, Any]:
    for inv in all_investigations(case):
        if inv.get("inv_id") == inv_id:
            return inv
    raise HTTPException(status_code=404, detail="Investigation not found")


def _parse_study_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.strptime(value.strip()[:10], "%Y-%m-%d")
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc)


def _redirect(case_id: str, flash: Optional[str] = None) -> RedirectResponse:
    url = f"/cases/{case_id}/investigations"
    if flash:
        url += f"?flash={flash}"
    return RedirectResponse(url, status_code=303)


# ---------------------------------------------------------------------------
# Tab view
# ---------------------------------------------------------------------------

@router.get("/{case_id}/investigations")
def investigations_tab(
    request: Request,
    case_id: str,
    flash: Optional[str] = None,
    user: Dict[str, Any] = Depends(require_current_user),
):
    case = _load_case(case_id, user)
    is_admin_view = user.get("role") == "admin"

    return request.app.state.templates.TemplateResponse(
        request,
        "investigations_tab.html",
        {
            "request": request,
            "case": case,
            "case_id": case_id,
            "active_tab": "investigations",
            "investigations": all_investigations(case),
            "pending": pending_verification(case),
            "problems": problem_investigations(case),
            "exam_submitted": _exam_submitted(case),
            "at_capacity": len(all_investigations(case)) >= INV_MAX_PER_CASE,
            "max_per_case": INV_MAX_PER_CASE,
            "modalities": {m: MODALITY_LABELS[m] for m in MODALITIES},
            "sides": {s: SIDE_LABELS[s] for s in SIDES},
            "is_admin_view": is_admin_view,
            "flash": flash,
            "findings_limit": INV_MAX_FINDINGS_CHARS,
            "impression_limit": INV_MAX_IMPRESSION_CHARS,
            "status_labels": INV_STATUS_LABELS,
            "visual_classes": INV_VISUAL_CLASS,
        },
    )


# ---------------------------------------------------------------------------
# Upload and extract
# ---------------------------------------------------------------------------

@router.post("/{case_id}/investigations/upload")
async def upload_investigation(
    case_id: str,
    report_file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(require_physio),
):
    case = _load_case(case_id, user)
    _require_exam_submitted(case)

    if len(all_investigations(case)) >= INV_MAX_PER_CASE:
        return _redirect(case_id, "at_capacity")

    file_bytes = await report_file.read()

    try:
        record = extract_investigation(
            groq_client=_groq(),
            uid=user["uid"],
            file_bytes=file_bytes,
            content_type=report_file.content_type,
            filename=report_file.filename,
        )
    except UploadRejected as exc:
        logger.info("Upload rejected for case %s: %s", case_id, exc)
        return _redirect(case_id, "upload_rejected")

    # The source document is retained regardless of extraction outcome. The
    # clinician verifies the transcription against it, so it must remain
    # available even when extraction failed.
    try:
        file_ref = _fs().put(
            file_bytes,
            filename=report_file.filename,
            content_type=report_file.content_type,
            case_id=str(case["_id"]),
            inv_id=record["inv_id"],
            uploaded_by_uid=user["uid"],
            uploaded_at=datetime.now(timezone.utc),
        )
        record["file_ref"] = str(file_ref)
    except Exception:
        logger.exception("Could not store source document for case %s", case_id)

    _cases().update_one(
        {"_id": case["_id"]},
        {"$push": {"investigations": record}},
    )

    return _redirect(case_id, f"uploaded_{record['extraction_status'].lower()}")


# ---------------------------------------------------------------------------
# Verification - the only path to agent visibility
# ---------------------------------------------------------------------------

@router.post("/{case_id}/investigations/{inv_id}/verify")
def verify_investigation(
    case_id: str,
    inv_id: str,
    modality: str = Form(...),
    side: str = Form(...),
    study_date: Optional[str] = Form(None),
    report_findings: str = Form(""),
    report_impression: str = Form(""),
    user: Dict[str, Any] = Depends(require_physio),
):
    """
    Confirm a transcription against its source document.

    The clinician may correct any field before confirming. Corrections are
    written to the working fields; raw_extraction is left untouched, so the
    difference between the two is a measurable record of extraction accuracy.
    """
    case = _load_case(case_id, user)
    _require_exam_submitted(case)

    inv = _find_investigation(case, inv_id)
    if inv.get("extraction_status") != INV_STATUS_EXTRACTED:
        return _redirect(case_id, "not_pending")

    if not is_valid_modality(modality) or not is_valid_side(side):
        return _redirect(case_id, "invalid_field")

    findings = " ".join(report_findings.split())[:INV_MAX_FINDINGS_CHARS]
    impression = " ".join(report_impression.split())[:INV_MAX_IMPRESSION_CHARS]

    if not findings and not impression:
        return _redirect(case_id, "empty_report")

    _cases().update_one(
        {"_id": case["_id"], "investigations.inv_id": inv_id},
        {"$set": {
            "investigations.$.modality": modality,
            "investigations.$.side": side,
            "investigations.$.study_date": _parse_study_date(study_date),
            "investigations.$.report_findings": findings,
            "investigations.$.report_impression": impression,
            "investigations.$.extraction_status": INV_STATUS_VERIFIED,
            "investigations.$.verified_by_uid": user["uid"],
            "investigations.$.verified_at": datetime.now(timezone.utc),
        }},
    )

    return _redirect(case_id, "verified")


# ---------------------------------------------------------------------------
# Manual entry
# ---------------------------------------------------------------------------

@router.post("/{case_id}/investigations/manual")
def add_manual_investigation(
    case_id: str,
    modality: str = Form(...),
    side: str = Form(...),
    study_date: Optional[str] = Form(None),
    report_findings: str = Form(""),
    report_impression: str = Form(""),
    user: Dict[str, Any] = Depends(require_physio),
):
    case = _load_case(case_id, user)
    _require_exam_submitted(case)

    if len(all_investigations(case)) >= INV_MAX_PER_CASE:
        return _redirect(case_id, "at_capacity")

    if not is_valid_modality(modality) or not is_valid_side(side):
        return _redirect(case_id, "invalid_field")

    if not report_findings.strip() and not report_impression.strip():
        return _redirect(case_id, "empty_report")

    record = manual_investigation_record(
        uid=user["uid"],
        modality=modality,
        side=side,
        study_date=_parse_study_date(study_date),
        report_findings=report_findings,
        report_impression=report_impression,
    )

    _cases().update_one(
        {"_id": case["_id"]},
        {"$push": {"investigations": record}},
    )

    return _redirect(case_id, "manual_added")


# ---------------------------------------------------------------------------
# Source document retrieval
# ---------------------------------------------------------------------------

@router.get("/{case_id}/investigations/{inv_id}/source")
def investigation_source(
    case_id: str,
    inv_id: str,
    user: Dict[str, Any] = Depends(require_current_user),
):
    """Serve the original document so the clinician can verify against it."""
    case = _load_case(case_id, user)
    inv = _find_investigation(case, inv_id)

    file_ref = inv.get("file_ref")
    if not file_ref:
        raise HTTPException(status_code=404, detail="No source document")

    try:
        stored = _fs().get(ObjectId(file_ref))
    except Exception:
        raise HTTPException(status_code=404, detail="No source document")

    return Response(
        content=stored.read(),
        media_type=stored.content_type or "application/octet-stream",
        headers={"Content-Disposition": "inline"},
    )


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

@router.post("/{case_id}/investigations/{inv_id}/delete")
def delete_investigation(
    case_id: str,
    inv_id: str,
    user: Dict[str, Any] = Depends(require_physio),
):
    case = _load_case(case_id, user)
    inv = _find_investigation(case, inv_id)

    file_ref = inv.get("file_ref")
    if file_ref:
        try:
            _fs().delete(ObjectId(file_ref))
        except Exception:
            logger.exception("Could not delete source document %s", file_ref)

    _cases().update_one(
        {"_id": case["_id"]},
        {"$pull": {"investigations": {"inv_id": inv_id}}},
    )

    return _redirect(case_id, "deleted")