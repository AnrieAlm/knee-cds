"""
Knee CDS — application entry point
===================================================================
Agentic clinical decision support for junior physiotherapists.

Route ownership
---------------
All /cases/{case_id}/physical* routes belong to backend.physical_backend
and are mounted via include_router below. They are NOT defined here.
Defining them in both places previously produced two competing
implementations, with FastAPI silently resolving in favour of whichever
registered first.

The deterministic safety layer (Ottawa, Pittsburgh, red flags, neuro
trigger) never touches the LLM. The agent runs only after that layer
has passed.

Access control
--------------
Every route touching a case goes through backend.case_access. Checking
only that a valid Firebase token exists - which is what this file did
previously - authenticates the caller but authorises nothing. See the
deviation register below, D6.

=====================================================================
DEVIATIONS FROM PaaS SCAFFOLD  (Examples 07-10, main.py)
=====================================================================
D1  Scaffold is one main.py holding every route. Cygnus mounts three
    APIRouters (physical, admin, investigations) from a backend/
    package. Retained from the scaffold: FastAPI(), app.mount for
    static, Jinja2Templates(directory=...), starlette.status for
    redirect codes, and `form = await request.form()` for POST bodies.

D2  Scaffold defines validateFirebaseToken() and getUser() inline in
    every file. Cygnus imports both from backend/auth.py.

D3  Scaffold hardcodes the MongoDB URI with a live password in source
    (Example09 line 16). Cygnus reads MONGO_URI from the environment
    in store.py. Committing a credential is not defensible for a
    system holding clinical data.

D5  Scaffold uses camelCase route handlers (uploadFile, filterByRange).
    Cygnus uses snake_case per PEP 8.

D6  Scaffold enforces ownership by scoping the query - Example07 does
    find_one({'user_id': user_token['user_id']}) and so cannot return
    another user's document. Cygnus applies the same principle through
    case_access.load_case_for_read/write, which additionally permits a
    head-of-department read that a single scoped query cannot express.

D7  Scaffold returns RedirectResponse('/') on any auth failure. Cygnus
    raises HTTPException and registers a handler (addition 2 below)
    that converts 401 into the redirect for page requests. 403 and 404
    keep their real status codes so the distinction survives into logs.

D8  Scaffold has no read/write role split. Cygnus rejects admin writes.

D10 Scaffold has no CORS middleware at all. Cygnus previously set
    allow_origins=["*"]; now restricted (addition 1). Every fetch() in
    this application is same-origin, so nothing needs the wildcard.
=====================================================================
"""

from dotenv import load_dotenv
load_dotenv()

import threading

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException
import starlette.status as status

from backend import store
from backend.auth import get_current_user

# Ownership and role gate — every case route passes through here
from backend.case_access import (
    load_case_for_read,
    load_case_for_write,
    try_load_case_for_read,
    try_load_case_for_write,
)

# Deterministic safety layer — pure Python, no LLM
from backend.rules.ottawa import apply_ottawa_knee_rule, OttawaInput
from backend.rules.pittsburgh import apply_pittsburgh_knee_rule, PittsburghInput
from backend.safety.red_flags import screen_red_flags, RedFlagInput

# Agent layer
from backend.agent.orchestrator import run_agent_only

# Physical examination router (owns all /physical routes)
from backend.physical_backend import router as physical_router

from backend.investigation_routes import router as investigation_router

# Admin router (read-only head-of-department oversight)
from backend.admin import router as admin_router


app = FastAPI(
    title="Knee CDS API",
    description="Agentic clinical decision support for junior physiotherapists",
    version="0.1.0",
)

# -------------------------------------------------------------------
# addition 1 - https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS

# CORS. Previously allow_origins=["*"], which is wider than anything
# this application needs: the UI is server-rendered Jinja2 and every
# fetch() call in case_summary.html is same-origin. Restricting costs
# nothing. If you later serve the frontend from a separate host, add
# that origin to this list rather than restoring the wildcard.
# -------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

templates = Jinja2Templates(directory="frontend/templates")

# -------------------------------------------------------------------
# Render a missing value as an empty string rather than the literal
# text "None".
#
# Jinja renders an explicit None as "None". Templates that read
# case.physical.get('test_lachman') receive None for an unrecorded
# field, which was being written into value="None" attributes, posted
# back, and stored in MongoDB. On the MMT card "None" | int evaluates
# to 0, producing a false 0/5 grade and a false neuro trigger.
#
# Dotted access (case.physical.neuro_notes) returns Undefined and was
# never affected, which is why only some fields showed the fault.
# -------------------------------------------------------------------
templates.env.finalize = lambda value: "" if value is None else value

# Shared with the physical router, which reads them off app.state.
# These must be set BEFORE include_router runs.
app.state.templates = templates
app.state.store = store

# -------------------------------------------------------------------
# Routers are mounted once, here, after app.state is populated.
# physical_router was previously included twice - once before this
# block and once after - which registered every /physical route twice
# and duplicated them in /docs.
# -------------------------------------------------------------------
app.include_router(physical_router)
app.include_router(admin_router)
app.include_router(investigation_router)


# -------------------------------------------------------------------
# addition 2
# Auth failures reach the browser as a redirect, not a JSON 401.
#
# case_access raises HTTPException so that a signed-out user (401), a
# supervisor attempting a write (403) and a case that is absent or not
# theirs (404) stay distinguishable in the server log. For a page
# request a bare 401 body is useless, so it becomes the redirect to
# /login that the scaffold performs directly. Anything that is not a
# 401 - and any request that did not ask for HTML, such as the polling
# endpoints - falls through to FastAPI's default handler and keeps its
# real status code.
# -------------------------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def auth_failure_redirect(request: Request, exc: StarletteHTTPException):
    wants_html = "text/html" in request.headers.get("accept", "")
    if exc.status_code == 401 and wants_html:
        return RedirectResponse("/login", status_code=status.HTTP_302_FOUND)
    return await http_exception_handler(request, exc)


# ===================================================================
# Helpers — agent input construction
# ===================================================================

def _case_fingerprint(case: dict) -> str:
    """
    Short signature of the inputs a suggestion was built from.

    If this changes, the stored suggestion is out of date and must be
    regenerated. Covers both history and physical, so submitting
    physical findings correctly invalidates a suggestion that was
    built from history alone.
    """
    return str(case.get("history", {})) + str(case.get("physical", {}))


def _build_patient_context(assessment: dict, history: dict | None) -> str:
    """
    Scoping instructions for the agent, derived deterministically from
    the assessment and history. No LLM involvement.

    NOTE ON A FIXED BUG: previously the return-to-sport exclusion was
    appended to patient_context *after* safety_facts had already been
    built by string concatenation, so that instruction never reached
    the agent. Context is now assembled in full before safety_facts is
    constructed.
    """
    parts = [
        "PATIENT CONTEXT (use to scope your suggestions):",
        "- Clinical phase: Acute triage only.",
        "- Do NOT suggest return-to-sport tests, hop tests, single-leg",
        "  performance tests, or high-performance athletic screening.",
        "  These are only appropriate in later rehabilitation phases.",
    ]

    # Ottawa age criterion — read from the stored triggered_criteria list
    ottawa_triggered = assessment.get("ottawa", {}).get("triggered_criteria", [])
    age_criterion_fired = any(
        "55" in c or "age" in c.lower() for c in ottawa_triggered
    )
    if age_criterion_fired:
        parts.append(
            "- Patient is aged 55 or older: adapt all test suggestions"
            " to be appropriate for this age group and acute presentation."
        )

    # Goal-based scoping
    if history and history.get("patient_goal") != "return_to_sport":
        parts.append(
            "- Patient goal is not return to sport: do NOT suggest"
            " return-to-sport outcome measures or hop tests."
        )

    # Weight-bearing status constrains which tests are performable
    if history and history.get("weight_bearing") == "none":
        parts.append(
            "- Patient is non-weight-bearing: do NOT suggest tests that"
            " require weight-bearing, such as Thessaly."
        )

    return "\n".join(parts)


def _build_safety_facts(assessment: dict, history: dict | None) -> str:
    """
    The settled, deterministic findings handed to the agent as facts it
    may not override. This is the boundary between the rule layer and
    the reasoning layer.
    """
    ottawa_text = assessment.get("ottawa", {}).get("rationale", "not assessed")
    pittsburgh_text = assessment.get("pittsburgh", {}).get("rationale", "not assessed")

    return (
        "ESTABLISHED FACTS (do not override):\n"
        f"- Ottawa: {ottawa_text}\n"
        f"- Pittsburgh: {pittsburgh_text}\n"
        "- Red-flag screen: negative.\n\n"
        + _build_patient_context(assessment, history)
    )


def _build_history_summary(history: dict) -> str:
    """
    Turn the recorded history into a clinical summary line for the agent.
    Deterministic pre-processing — the interpretive hints (haemarthrosis,
    effusion pattern) are rule-derived, not model-derived.
    """
    interpretive = []

    if history.get("swelling_present") == "yes":
        onset = history.get("swelling_onset")
        if onset == "immediate":
            interpretive.append("Immediate swelling — haemarthrosis possible.")
        elif onset == "delayed":
            interpretive.append(
                "Delayed swelling — synovial effusion pattern, "
                "consider meniscal involvement."
            )

    if history.get("weight_bearing") == "none":
        interpretive.append("Patient non-weight-bearing — defer weight-bearing tests.")

    if history.get("audible_pop") == "yes":
        interpretive.append("Audible/felt pop at injury — consider ACL involvement.")

    mech_syms = [
        sym for sym in
        ["locking", "giving_way", "catching", "clicking", "grinding"]
        if history.get("mech_sym_" + sym) == "yes"
    ]
    mech_sym_str = ", ".join(mech_syms) if mech_syms else "none reported"

    summary = (
        f"Patient: {history.get('activity_level', 'unknown activity level')} "
        f"individual, involved {history.get('involved_side', 'unknown')} knee. "
        f"Activity at injury: {history.get('activity_at_injury', 'unknown')}. "
        f"Mechanism: {history.get('mechanism_type', 'unknown')} — "
        f"{history.get('mechanism_description', 'no description')}. "
        f"Pain location: {history.get('pain_location', 'not recorded')}. "
        f"Swelling: {history.get('swelling_onset', 'none')}. "
        f"Weight-bearing: {history.get('weight_bearing', 'unknown')}. "
        f"Mechanical symptoms: {mech_sym_str}. "
        f"Goal: {history.get('patient_goal', 'not recorded')}. "
    )

    if interpretive:
        summary += " ".join(interpretive) + " "

    return summary


def _build_agent_query(case: dict) -> str:
    """
    Assemble the clinician question: history summary first, then the
    question itself. The question goes last so the agent reads the
    presentation before being asked to act on it.

    Physical findings are deliberately NOT appended here. They are passed
    separately to run_agent_only, which places them under their own
    labelled header — "recorded by the clinician, do not contradict" —
    rather than burying them inside the question text. The distinction
    matters: history is what the patient reported, physical findings are
    what the clinician measured, and the agent should be able to tell
    which is which.

    Physical formatting is delegated to the single canonical formatter in
    physical_context so that the /suggest path and the /agent-context
    endpoint cannot drift apart.
    """
    history = case.get("history")
    physical = case.get("physical")

    if history:
        query = _build_history_summary(history)
    else:
        query = "Given an acute knee injury with no red flags. "

    if physical:
        from backend.agent.physical_context import formatPhysicalForAgent
        query += formatPhysicalForAgent(
            physical, deferrals=case.get("deferrals")
        ) + " "

    query += "Which special tests should be prioritised and in what order?"
    return query


# ===================================================================
# Page routes
# ===================================================================

@app.get("/login")
async def login(request: Request):
    return templates.TemplateResponse(request, "login.html")


@app.get("/")
async def index(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login")
    if user.get("role") == "admin":
        return RedirectResponse("/admin")
    return templates.TemplateResponse(
        request, "index.html",
        {"cases": store.list_cases(user["uid"]), "user": user},
    )


@app.post("/cases/new")
async def new_case_submit(request: Request):
    """
    Creation is the one write with no existing case to check ownership
    against, so the role check is inline rather than via case_access.
    """
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login")
    if user.get("role") == "admin":
        raise HTTPException(
            status_code=403,
            detail="Head of department accounts have read-only access",
        )

    form = await request.form()

    patient_label = (form.get("patient_label") or "").strip()
    if not patient_label:
        return templates.TemplateResponse(
            request, "new_case.html",
            {"error_message": "A patient label is required."},
        )

    store.create_case(patient_label, user)
    return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/cases/new")
async def new_case_form(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login")
    return templates.TemplateResponse(
        request, "new_case.html", {"error_message": None},
    )


@app.get("/cases/{case_id}")
async def view_case(request: Request, case_id: str):
    # Loaded rather than merely existence-checked: entering the case at
    # all is what is being authorised here.
    load_case_for_read(request, case_id)
    return RedirectResponse(f"/cases/{case_id}/history")


# -------------------------------------------------------------------
# History tab
# -------------------------------------------------------------------

@app.get("/cases/{case_id}/history")
async def case_history(request: Request, case_id: str):
    user, case = load_case_for_read(request, case_id)

    return templates.TemplateResponse(
        request, "case_history.html",
        {
            "case": case,
            "active_tab": "history",
            "user": user,
            "is_admin_view": False,
        },
    )


@app.post("/cases/{case_id}/history", response_class=RedirectResponse)
async def case_history_submit(request: Request, case_id: str):
    user, case = load_case_for_write(request, case_id)

    form = await request.form()

    # Every field has a safe default. Checkboxes return their value only
    # when checked, so an absent key means unchecked, which is 'no'.
    history_dict = {
        # presenting complaint
        "involved_side": form.get("involved_side", "unknown"),
        "chief_complaint": form.get("chief_complaint", ""),
        "activity_at_injury": form.get("activity_at_injury", "other"),

        # mechanism
        "mechanism_type": form.get("mechanism_type", "unknown"),
        "mechanism_description": form.get("mechanism_description", ""),
        "audible_pop": form.get("audible_pop", "unsure"),

        # symptom profile
        "pain_location": form.get("pain_location", ""),
        "swelling_present": form.get("swelling_present", "no"),
        "swelling_onset": form.get("swelling_onset", "none"),
        "weight_bearing": form.get("weight_bearing", "full"),

        # mechanical symptoms
        "mech_sym_locking": form.get("mech_sym_locking", "no"),
        "mech_sym_giving_way": form.get("mech_sym_giving_way", "no"),
        "mech_sym_catching": form.get("mech_sym_catching", "no"),
        "mech_sym_clicking": form.get("mech_sym_clicking", "no"),
        "mech_sym_grinding": form.get("mech_sym_grinding", "no"),

        # patient context
        "occupation": form.get("occupation", ""),
        "activity_level": form.get("activity_level", "sedentary"),
        "previous_knee_injury": form.get("previous_knee_injury", "no"),
        "previous_surgery": form.get("previous_surgery", "no"),
        "surgery_detail": form.get("surgery_detail", ""),
        "patient_goal": form.get("patient_goal", "daily_function"),
    }

    store.save_history(case_id, history_dict)

    return RedirectResponse(
        f"/cases/{case_id}/exam",
        status_code=status.HTTP_302_FOUND,
    )


# -------------------------------------------------------------------
# Exam tab (GET — the POST is in the second half of this file)
# -------------------------------------------------------------------

@app.get("/cases/{case_id}/exam")
async def case_exam(request: Request, case_id: str):
    user, case = load_case_for_read(request, case_id)

    return templates.TemplateResponse(
        request, "case_exam.html",
        {
            "case": case,
            "active_tab": "exam",
            "user": user,
            "is_admin_view": False,
            "error_message": None,
        },
    )


# -------------------------------------------------------------------
# Summary tab
# -------------------------------------------------------------------

@app.get("/cases/{case_id}/summary")
async def case_summary(request: Request, case_id: str):
    """
    is_admin_view is passed explicitly as False here so the template has
    a defined value on both paths. admin.py passes True for the same
    template; without this the flag was Undefined on the physiotherapist
    route, and `{% if not is_admin_view %}` would have been silently
    true-by-accident rather than true-by-decision.
    """
    user, case = load_case_for_read(request, case_id)

    return templates.TemplateResponse(
        request, "case_summary.html",
        {
            "case": case,
            "active_tab": "summary",
            "user": user,
            "is_admin_view": False,
        },
    )
# ===================================================================
# Chat — synchronous RAG, single retrieval + single LLM call
# ===================================================================

@app.post("/cases/{case_id}/chat")
async def case_chat(request: Request, case_id: str):
    """
    Answers JSON, so this uses the non-raising loader and shapes its own
    error body rather than letting the 401 handler redirect it.

    Gated as a WRITE even though it stores nothing. Chat sends the case
    to the model, and a head-of-department account questioning the agent
    about a junior's case would generate model output attached to that
    case which the treating clinician never saw. Read-only oversight has
    to mean the supervisor observes the record rather than extending it.
    """
    user, case = try_load_case_for_write(request, case_id)
    if not case:
        return {"error": "not authorised"}

    form = await request.form()
    message = (form.get("message") or "").strip()

    if not message:
        return {"error": "empty message"}

    print(f"[chat] message: {message}")

    from backend.agent.chat import run_chat
    response = run_chat(message, case)

    print(f"[chat] response: {response}")

    return {"response": response}


# ===================================================================
# Health / status
# ===================================================================

@app.get("/api/status")
async def api_status():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ===================================================================
# POST /cases/{case_id}/exam
# -------------------------------------------------------------------
# The deterministic safety layer. Ottawa, Pittsburgh and the red-flag
# screen run here as pure Python. No model is called on this path.
# ===================================================================

@app.post("/cases/{case_id}/exam", response_class=RedirectResponse)
async def case_exam_submit(request: Request, case_id: str):
    user, case = load_case_for_write(request, case_id)

    form = await request.form()

    def checked(field_name: str) -> bool:
        """An unchecked checkbox is absent from the form entirely."""
        return field_name in form

    # ---------------------------------------------------------------
    # addition 3
    # Age is REQUIRED. It is not defaulted.
    #
    # This previously read:
    #
    #     try:
    #         age = int(form.get("age", "0"))
    #     except ValueError:
    #         age = 0
    #
    # with a comment claiming that fell closed rather than open. It did
    # the opposite, and in two different directions at once:
    #
    #   Ottawa's age criterion is >= 55. An age of 0 does not meet it,
    #   so a blank field silently DROPS the criterion. A missing age on
    #   a 78-year-old produced "Ottawa negative" and no X-ray referral.
    #   That is failing open on a fracture rule.
    #
    #   Pittsburgh's age criterion is < 12 or > 50. An age of 0 DOES
    #   meet it, so the same blank field fires Pittsburgh as a false
    #   positive.
    #
    # A safety layer that substitutes a value for one a clinician did
    # not enter is asserting a clinical fact it does not have. The only
    # defensible behaviour is to refuse the submission. The `required`
    # attribute on the input is browser-side convenience, not a
    # guarantee - it is absent on any request that does not come from
    # that form.
    #
    # NOTE: this re-render does not repopulate the checkboxes already
    # ticked. Documented limitation; the path is reachable only when the
    # browser control is bypassed.
    # ---------------------------------------------------------------
    raw_age = (form.get("age") or "").strip()

    if not raw_age.isdigit():
        return templates.TemplateResponse(
            request, "case_exam.html",
            {
                "case": case,
                "active_tab": "exam",
                "user": user,
                "is_admin_view": False,
                "error_message": (
                    "Patient age is required and must be a whole number. "
                    "The fracture rules cannot be applied without it."
                ),
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    age = int(raw_age)

    if age > 120:
        return templates.TemplateResponse(
            request, "case_exam.html",
            {
                "case": case,
                "active_tab": "exam",
                "user": user,
                "is_admin_view": False,
                "error_message": (
                    f"Patient age of {age} is outside the plausible range. "
                    "Please check and re-enter."
                ),
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    # --- red-flag screen input ---
    red_flag_input = RedFlagInput(
        suspected_dvt=checked("suspected_dvt"),
        pulseless_limb=checked("pulseless_limb"),
        fever_with_joint_pain=checked("fever_with_joint_pain"),
        hot_swollen_joint=checked("hot_swollen_joint"),
        recent_infection=checked("recent_infection"),
        significant_trauma=checked("significant_trauma"),
        unable_to_weight_bear=checked("unable_to_weight_bear"),
        bony_tenderness=checked("bony_tenderness"),
        unexplained_weight_loss=checked("unexplained_weight_loss"),
        night_pain_at_rest=checked("night_pain_at_rest"),
        history_of_cancer=checked("history_of_cancer"),
        foot_drop=checked("foot_drop"),
        saddle_anaesthesia=checked("saddle_anaesthesia"),
        compartment_syndrome_signs=checked("compartment_syndrome_signs"),
    )

    # --- Ottawa Knee Rule input ---
    ottawa_input = OttawaInput(
        age=age,
        isolated_patella_tenderness=checked("isolated_patella_tenderness"),
        fibula_head_tenderness=checked("fibula_head_tenderness"),
        unable_to_flex_90=checked("unable_to_flex_90"),
        unable_to_weight_bear=checked("unable_to_weight_bear"),
    )

    # --- Pittsburgh Knee Rule input ---
    pittsburgh_input = PittsburghInput(
        mechanism_blunt_trauma_or_fall=checked("mechanism_blunt_trauma_or_fall"),
        age=age,
        unable_to_weight_bear=checked("unable_to_weight_bear"),
    )

    # --- run the rules ---
    red_flag_result = screen_red_flags(red_flag_input)
    ottawa_result = apply_ottawa_knee_rule(ottawa_input)
    pittsburgh_result = apply_pittsburgh_knee_rule(pittsburgh_input)

    # --- flatten to plain dicts for MongoDB ---
    assessment = {
        "red_flag": {
            "escalate_immediately": red_flag_result.escalate_immediately,
            "triggered_flags": red_flag_result.triggered_flags,
            "rationale": red_flag_result.rationale,
            "action": red_flag_result.action,
        },
        "ottawa": {
            "xray_indicated": ottawa_result.xray_indicated,
            "triggered_criteria": ottawa_result.triggered_criteria,
            "rationale": ottawa_result.rationale,
        },
        "pittsburgh": {
            "xray_indicated": pittsburgh_result.xray_indicated,
            "triggered_criteria": pittsburgh_result.triggered_criteria,
            "rationale": pittsburgh_result.rationale,
        },
        # Convenience flag the case list and templates read directly
        "red_flag_positive": red_flag_result.escalate_immediately,
        # Records the age the rules were evaluated against, so a later
        # re-screen can be compared against the original inputs
        "screened_age": age,
    }

    store.save_assessment(case_id, assessment)

    # A positive red flag halts the pathway. Go straight to summary,
    # where the escalation notice is shown, rather than continuing to
    # hands-on examination of a knee that needs escalation.
    if red_flag_result.escalate_immediately:
        return RedirectResponse(
            f"/cases/{case_id}/summary",
            status_code=status.HTTP_302_FOUND,
        )

    return RedirectResponse(
        f"/cases/{case_id}/physical",
        status_code=status.HTTP_302_FOUND,
    )


# ===================================================================
# POST /cases/{case_id}/suggest
# -------------------------------------------------------------------
# Synchronous agent run. Blocks until the model returns. Kept for
# debugging and for demo fallback if the polling path misbehaves;
# the UI uses /suggest-async.
# ===================================================================

@app.post("/cases/{case_id}/suggest", response_class=RedirectResponse)
async def case_suggest(request: Request, case_id: str):
    """
    Gated as a WRITE. Running the agent appends to agentLog and replaces
    assessment.agent_suggestion, so a supervisor triggering it would
    overwrite the record with output the treating clinician never saw -
    the audit trail would then no longer represent the assessment as
    performed.
    """
    user, case = load_case_for_write(request, case_id)

    assessment = case.get("assessment")

    # SAFETY GUARD — the agent does not run unless the deterministic
    # layer has run and passed. No assessment, or a positive red flag,
    # means no model call at all.
    if not assessment or assessment.get("red_flag_positive"):
        return RedirectResponse(
            f"/cases/{case_id}/summary",
            status_code=status.HTTP_302_FOUND,
        )

    safety_facts = _build_safety_facts(assessment, case.get("history"))
    query = _build_agent_query(case)

    print(f"[suggest] query:\n{query}")

    result = run_agent_only(
        query, safety_facts, case.get("physical"), case.get("investigations"),
        deferrals=case.get("deferrals"),
    )

    # Append-only audit log: query, retrieved chunks, and output.
    # This is the chain that makes the reasoning traceable.
    store.append_agent_log(case_id, {
        "query": query,
        "retrieved": result["retrieved"],
        "suggestion": result["suggestion"],
        "generated_by_uid": user.get("uid"),
    })

    assessment["agent_suggestion"] = result["suggestion"]
    assessment["agent_suggestion_fingerprint"] = _case_fingerprint(case)
    assessment["agent_suggestion_status"] = "done"
    store.save_assessment(case_id, assessment)

    return RedirectResponse(
        f"/cases/{case_id}/summary",
        status_code=status.HTTP_302_FOUND,
    )


# ===================================================================
# POST /cases/{case_id}/suggest-async
# -------------------------------------------------------------------
# Starts the agent in a background thread and returns immediately.
# The browser polls /suggest-status for the result.
# ===================================================================

@app.post("/cases/{case_id}/suggest-async")
async def case_suggest_async(request: Request, case_id: str):
    # JSON endpoint, so use the non-raising loader. Write-gated for the
    # same audit-trail reason as /suggest above.
    user, case = try_load_case_for_write(request, case_id)
    if not case:
        return {"status": "error", "message": "not authorised"}

    assessment = case.get("assessment")

    # Same safety guard as the synchronous path
    if not assessment:
        return {"status": "no_assessment"}
    if assessment.get("red_flag_positive"):
        return {"status": "red_flag_halted"}

    # ---------------------------------------------------------------
    # addition 4
    # Refuse to start a second run while one is in flight. Two threads
    # racing on the same case both write to agentLog and both write
    # agent_suggestion, so the stored suggestion could end up being the
    # older of the two runs. The UI can double-fire this by reloading
    # the summary page while the spinner is showing.
    # ---------------------------------------------------------------
    if assessment.get("agent_suggestion_status") == "pending":
        return {"status": "pending"}

    # Only skip regeneration if the stored suggestion was built from
    # exactly the data currently in the case. Saving any physical
    # section changes the fingerprint and forces a fresh run, so a
    # suggestion built from history alone is never served after the
    # physical examination has been recorded.
    current_fingerprint = _case_fingerprint(case)

    if (assessment.get("agent_suggestion")
            and assessment.get("agent_suggestion_fingerprint") == current_fingerprint):
        return {"status": "already_done"}

    safety_facts = _build_safety_facts(assessment, case.get("history"))
    query = _build_agent_query(case)

    # Snapshots taken deliberately at request time, alongside the query
    # that was built from them. The agent must reason over one coherent
    # picture of the case; re-reading these inside the thread would let
    # the model see physical findings the query never described.
    physical_snapshot = case.get("physical")
    investigations_snapshot = case.get("investigations")
    deferrals_snapshot = case.get("deferrals")
    author_uid = user.get("uid")

    print(f"[suggest-async] query:\n{query}")

    # Mark pending so the UI shows the spinner immediately
    assessment["agent_suggestion_status"] = "pending"
    store.save_assessment(case_id, assessment)

    def run_in_background():
        # The RESULT is written against a freshly read assessment rather
        # than the snapshot above, so anything saved to the case while
        # the model was working is not silently discarded. Only the
        # agent's own three fields are overwritten.
        try:
            result = run_agent_only(
                query, safety_facts, physical_snapshot, investigations_snapshot,
                deferrals=deferrals_snapshot,
            )

            store.append_agent_log(case_id, {
                "query": query,
                "retrieved": result["retrieved"],
                "suggestion": result["suggestion"],
                "generated_by_uid": author_uid,
            })

            fresh = store.get_case(case_id)
            current = fresh.get("assessment", {}) if fresh else {}

            current["agent_suggestion"] = result["suggestion"]
            current["agent_suggestion_fingerprint"] = current_fingerprint
            current["agent_suggestion_status"] = "done"
            store.save_assessment(case_id, current)

        except Exception as exc:
            print(f"[suggest-async] agent error: {exc}")
            fresh = store.get_case(case_id)
            current = fresh.get("assessment", {}) if fresh else {}
            current["agent_suggestion_status"] = "error"
            store.save_assessment(case_id, current)

    threading.Thread(target=run_in_background, daemon=True).start()

    return {"status": "pending"}


# ===================================================================
# GET /cases/{case_id}/suggest-status
# -------------------------------------------------------------------
# Polled by the browser every few seconds until the agent finishes.
# ===================================================================

@app.get("/cases/{case_id}/suggest-status")
async def case_suggest_status(request: Request, case_id: str):
    # Read-gated: a supervisor may observe a run in progress, they just
    # may not have started it.
    user, case = try_load_case_for_read(request, case_id)
    if not case:
        return {"status": "error", "message": "not authorised"}

    assessment = case.get("assessment", {})

    return {
        "status": assessment.get("agent_suggestion_status", "idle"),
        "suggestion": assessment.get("agent_suggestion", ""),
    }