"""
backend/admin.py

Head-of-department oversight. Read-only by construction.

Every route here is guarded by require_admin, which returns 403 for any
account whose role is not "admin". There are no POST routes in this
module at all - not because writes are hidden, but because they do not
exist. The head of department can see every assessment in every
department and change none of them.

That constraint is deliberate and it is a clinical one rather than a
technical convenience. A physiotherapy record attributes findings to the
clinician who performed the examination. If a supervisor could edit the
record, or could cause the reasoning agent to regenerate a suggestion,
the stored case would no longer be evidence of what the treating
physiotherapist actually saw and decided. Oversight has to be able to
observe the assessment without altering it.
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from backend import store
from backend.auth import require_admin, get_current_user
from backend.db import get_db
from backend.constants import (
    DEPARTMENTS,
    GRADES,
    SUPERVISION_GRADES,
    department_label,
    grade_label,
)

router = APIRouter(prefix="/admin", tags=["admin"])

templates = Jinja2Templates(directory="frontend/templates")
templates.env.globals["department_label"] = department_label
templates.env.globals["grade_label"] = grade_label


def _case_flags(case: dict) -> dict:
    """
    Pull the few facts a supervisor scans a list for.

    Reads only what the deterministic layer already stored. Nothing here
    recomputes a rule or calls the agent - a supervisor's list view must
    not be able to change a clinical result.
    """
    assessment = case.get("assessment") or {}
    red_flag = assessment.get("red_flag") or {}
    ottawa = assessment.get("ottawa") or {}
    pittsburgh = assessment.get("pittsburgh") or {}

    return {
        "has_assessment": bool(assessment),
        "red_flag": bool(red_flag.get("escalate_immediately")),
        "xray_indicated": bool(
            ottawa.get("xray_indicated") or pittsburgh.get("xray_indicated")
        ),
        "has_suggestion": bool(assessment.get("agent_suggestion")),
        "suggestion_pending": assessment.get("agent_suggestion_status") == "pending",
    }


@router.get("")
async def admin_dashboard(
    request: Request,
    department: str = "",
    grade: str = "",
    supervision: str = "",
    include_legacy: str = "",
):
    """
    All cases across all departments, newest first.

    Filters are optional and combine. 'supervision' narrows to the least
    experienced grades, which is the view a head of department actually
    needs - oversight concentrates where clinical experience is lowest.
    """
    user = require_admin(request)

    cases = store.list_all_cases(
        department=department or None,
        grade=grade or None,
        include_legacy=bool(include_legacy),
    )

    if supervision:
        cases = [c for c in cases if c.get("grade") in SUPERVISION_GRADES]

    rows = [{"case": c, "flags": _case_flags(c)} for c in cases]

    # Counts for the summary strip. Computed from the filtered set so the
    # numbers always describe what is actually on screen.
    stats = {
        "total": len(rows),
        "red_flags": sum(1 for r in rows if r["flags"]["red_flag"]),
        "xray": sum(1 for r in rows if r["flags"]["xray_indicated"]),
        "no_assessment": sum(1 for r in rows if not r["flags"]["has_assessment"]),
    }

    # Department counts across everything, so the filter chips can show
    # totals even when a filter is active.
    all_cases = store.list_all_cases(include_legacy=bool(include_legacy))
    dept_counts = {}
    for c in all_cases:
        key = c.get("department", "unassigned")
        dept_counts[key] = dept_counts.get(key, 0) + 1

    return templates.TemplateResponse(
        request,
        "admin_dashboard.html",
        {
            "user": user,
            "rows": rows,
            "stats": stats,
            "departments": DEPARTMENTS,
            "grades": GRADES,
            "dept_counts": dept_counts,
            "active_department": department,
            "active_grade": grade,
            "supervision": bool(supervision),
            "include_legacy": bool(include_legacy),
            "physios": _roster(),
        },
    )


def _roster():
    """Every physiotherapist, with a case count each."""
    db = get_db()
    users = list(
        db["users"].find(
            {"role": "physio"},
            {"_id": 0, "uid": 1, "full_name": 1, "email": 1,
             "department": 1, "grade": 1},
        )
    )
    counts = {}
    for c in store.list_all_cases():
        uid = c.get("physio_uid")
        counts[uid] = counts.get(uid, 0) + 1
    for u in users:
        u["case_count"] = counts.get(u["uid"], 0)
    users.sort(key=lambda u: (u.get("department", ""), u.get("full_name", "")))
    return users


def _load_case_or_404(case_id: str) -> dict:
    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@router.get("/cases/{case_id}")
async def admin_view_case(request: Request, case_id: str):
    """Read-only history view. Entry point for a case."""
    user = require_admin(request)
    case = _load_case_or_404(case_id)
    return templates.TemplateResponse(
        request,
        "case_history.html",
        {
            "case": case,
            "active_tab": "history",
            "user": user,
            "is_admin_view": True,
        },
    )


@router.get("/cases/{case_id}/physical")
async def admin_view_physical(request: Request, case_id: str):
    """Read-only physical examination findings."""
    user = require_admin(request)
    case = _load_case_or_404(case_id)
    return templates.TemplateResponse(
        request,
        "physical_summary.html",
        {
            "case": case,
            "active_tab": "physical",
            "user": user,
            "is_admin_view": True,
        },
    )


@router.get("/cases/{case_id}/summary")
async def admin_view_summary(request: Request, case_id: str):
    """
    Read-only summary: deterministic System Check plus the STORED agent
    suggestion.

    is_admin_view suppresses the request/retry buttons, the chat panel
    and the pending-state poller in case_summary.html. The suggestion
    shown here is whatever was generated for the treating physiotherapist
    - opening this page never calls the agent, so a supervisor's visit
    cannot overwrite the record with output the junior never saw.
    """
    user = require_admin(request)
    case = _load_case_or_404(case_id)
    return templates.TemplateResponse(
        request,
        "case_summary.html",
        {
            "case": case,
            "active_tab": "summary",
            "user": user,
            "is_admin_view": True,
        },
    )


@router.get("/physios")
async def admin_physios(request: Request):
    """Roster view - who is in which department, and how active."""
    user = require_admin(request)
    return templates.TemplateResponse(
        request,
        "admin_dashboard.html",
        {
            "user": user,
            "rows": [],
            "stats": {"total": 0, "red_flags": 0, "xray": 0, "no_assessment": 0},
            "departments": DEPARTMENTS,
            "grades": GRADES,
            "dept_counts": {},
            "active_department": "",
            "active_grade": "",
            "supervision": False,
            "include_legacy": False,
            "physios": _roster(),
            "roster_only": True,
        },
    )
