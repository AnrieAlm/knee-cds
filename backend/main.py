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
"""

from dotenv import load_dotenv
load_dotenv()

import threading

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import starlette.status as status

from backend import store
from backend.auth import get_user, validate_firebase_token

# Deterministic safety layer — pure Python, no LLM
from backend.rules.ottawa import apply_ottawa_knee_rule, OttawaInput
from backend.rules.pittsburgh import apply_pittsburgh_knee_rule, PittsburghInput
from backend.safety.red_flags import screen_red_flags, RedFlagInput

# Agent layer
from backend.agent.orchestrator import run_agent_only

# Physical examination router (owns all /physical routes)
from backend.physical_backend import router as physical_router


app = FastAPI(
    title="Knee CDS API",
    description="Agentic clinical decision support for junior physiotherapists",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# Shared with the physical router, which reads them off app.state
app.state.templates = templates
app.state.store = store

app.include_router(physical_router)

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
        query += formatPhysicalForAgent(physical) + " "

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
    if not get_user(request):
        return RedirectResponse("/login")

    return templates.TemplateResponse(
        request, "index.html", {"cases": store.list_cases()}
    )


@app.get("/cases/new")
async def new_case_form(request: Request):
    if not get_user(request):
        return RedirectResponse("/login")

    return templates.TemplateResponse(request, "new_case.html")


@app.post("/cases/new")
async def new_case_submit(request: Request):
    if not get_user(request):
        return RedirectResponse("/login")

    form = await request.form()
    store.create_case(form["patient_label"])
    return RedirectResponse("/", status_code=303)


@app.get("/cases/{case_id}")
async def view_case(request: Request, case_id: str):
    if not get_user(request):
        return RedirectResponse("/login")

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return RedirectResponse(f"/cases/{case_id}/history")


# -------------------------------------------------------------------
# History tab
# -------------------------------------------------------------------

@app.get("/cases/{case_id}/history")
async def case_history(request: Request, case_id: str):
    if not get_user(request):
        return RedirectResponse("/login")

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return templates.TemplateResponse(
        request, "case_history.html",
        {"case": case, "active_tab": "history"},
    )


@app.post("/cases/{case_id}/history", response_class=RedirectResponse)
async def case_history_submit(request: Request, case_id: str):
    if not get_user(request):
        return RedirectResponse("/login")

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

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
# Exam tab (GET — the POST is in Part 4)
# -------------------------------------------------------------------

@app.get("/cases/{case_id}/exam")
async def case_exam(request: Request, case_id: str):
    if not get_user(request):
        return RedirectResponse("/login")

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return templates.TemplateResponse(
        request, "case_exam.html",
        {"case": case, "active_tab": "exam"},
    )


# -------------------------------------------------------------------
# Summary tab
# -------------------------------------------------------------------

@app.get("/cases/{case_id}/summary")
async def case_summary(request: Request, case_id: str):
    if not get_user(request):
        return RedirectResponse("/login")

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return templates.TemplateResponse(
        request, "case_summary.html",
        {"case": case, "active_tab": "summary"},
    )


# -------------------------------------------------------------------
# Chat — synchronous RAG, single retrieval + single LLM call
# -------------------------------------------------------------------

@app.post("/cases/{case_id}/chat")
async def case_chat(request: Request, case_id: str):
    if not get_user(request):
        return {"error": "not authenticated"}

    case = store.get_case(case_id)
    if not case:
        return {"error": "case not found"}

    form = await request.form()
    message = (form.get("message") or "").strip()

    if not message:
        return {"error": "empty message"}

    print(f"[chat] message: {message}")

    from backend.agent.chat import run_chat
    response = run_chat(message, case)

    print(f"[chat] response: {response}")

    return {"response": response}


# -------------------------------------------------------------------
# Health / status
# -------------------------------------------------------------------

@app.get("/api/status")
async def api_status():
    return {"status": "ok", "message": "Knee CDS API running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}

# ===================================================================
# POST /cases/{case_id}/exam
# -------------------------------------------------------------------
# The deterministic safety layer. Ottawa, Pittsburgh and the red-flag
# screen run here as pure Python with no LLM involvement whatsoever.
# A positive red flag halts the pathway before any hands-on assessment
# and before the agent is ever invoked.
# ===================================================================

@app.post("/cases/{case_id}/exam", response_class=RedirectResponse)
async def case_exam_submit(request: Request, case_id: str):
    if not get_user(request):
        return RedirectResponse("/login")

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    form = await request.form()

    def checked(field_name: str) -> bool:
        """An unchecked checkbox is absent from the form entirely."""
        return field_name in form

    # Age arrives as a string. A blank or malformed value becomes 0,
    # which fails the Ottawa age criterion closed rather than open.
    try:
        age = int(form.get("age", "0"))
    except ValueError:
        age = 0

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
    if not get_user(request):
        return RedirectResponse("/login")

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

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

    result = run_agent_only(query, safety_facts, case.get("physical"))

    # Append-only audit log: query, retrieved chunks, and output.
    # This is the chain that makes the reasoning traceable.
    store.append_agent_log(case_id, {
        "query": query,
        "retrieved": result["retrieved"],
        "suggestion": result["suggestion"],
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
    if not get_user(request):
        return {"status": "error", "message": "not authenticated"}

    case = store.get_case(case_id)
    if not case:
        return {"status": "error", "message": "case not found"}

    assessment = case.get("assessment")

    # Same safety guard as the synchronous path
    if not assessment:
        return {"status": "no_assessment"}
    if assessment.get("red_flag_positive"):
        return {"status": "red_flag_halted"}

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

    print(f"[suggest-async] query:\n{query}")

    # Mark pending so the UI shows the spinner immediately
    assessment["agent_suggestion_status"] = "pending"
    store.save_assessment(case_id, assessment)

    def run_in_background():
        # Re-read the assessment inside the thread rather than closing
        # over the outer dict. The outer copy is a snapshot taken before
        # the model ran; writing it back would silently discard anything
        # saved to the case while the agent was working.
        try:
            result = run_agent_only(query, safety_facts, case.get("physical"))

            fresh = store.get_case(case_id)
            current = fresh.get("assessment", {}) if fresh else {}

            store.append_agent_log(case_id, {
                "query": query,
                "retrieved": result["retrieved"],
                "suggestion": result["suggestion"],
            })

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
    if not get_user(request):
        return {"status": "error", "message": "not authenticated"}

    case = store.get_case(case_id)
    if not case:
        return {"status": "error", "message": "case not found"}

    assessment = case.get("assessment", {})

    return {
        "status": assessment.get("agent_suggestion_status", "idle"),
        "suggestion": assessment.get("agent_suggestion", ""),
    }    