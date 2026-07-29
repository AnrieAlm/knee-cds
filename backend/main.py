from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import google.oauth2.id_token
from google.auth.transport import requests as google_requests
from backend import store
from fastapi import HTTPException
import starlette.status as status
from backend.rules.ottawa import apply_ottawa_knee_rule, OttawaInput
from backend.rules.pittsburgh import apply_pittsburgh_knee_rule, PittsburghInput
from backend.safety.red_flags import screen_red_flags, RedFlagInput
from backend.agent.orchestrator import run_agent_only, run_assessment
app = FastAPI(
    title="Knee CDS API",
    description="Agentic clinical decision support for junior physiotherapists",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

firebase_request_adapter = google_requests.Request()


def validate_firebase_token(id_token):
    if not id_token:
        return None
    try:
        return google.oauth2.id_token.verify_firebase_token(id_token, firebase_request_adapter)
    except ValueError as err:
        print(str(err))
        return None


def require_user(request: Request):
    id_token = request.cookies.get("token")
    return validate_firebase_token(id_token)


@app.get("/login")
async def login(request: Request):
    return templates.TemplateResponse(request, "login.html")


@app.get("/")
async def index(request: Request):
    if not require_user(request):
        return RedirectResponse("/login")

    return templates.TemplateResponse(request, "index.html", {"cases": store.list_cases()})



@app.get("/cases/new")
async def new_case_form(request: Request):
    if not require_user(request):
        return RedirectResponse("/login")

    return templates.TemplateResponse(request, "new_case.html")


@app.post("/cases/new")
async def new_case_submit(request: Request):
    if not require_user(request):
        return RedirectResponse("/login")

    form = await request.form()
    store.create_case(form["patient_label"])
    return RedirectResponse("/", status_code=303)

@app.get("/cases/{case_id}")
async def view_case(request: Request, case_id: str):
    if not require_user(request):
        return RedirectResponse("/login")

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return RedirectResponse(f"/cases/{case_id}/history")


@app.get("/cases/{case_id}/history")
async def case_history(request: Request, case_id: str):
    if not require_user(request):
        return RedirectResponse("/login")

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return templates.TemplateResponse(request, "case_history.html", {"case": case, "active_tab": "history"})


@app.get("/cases/{case_id}/exam")
async def case_exam(request: Request, case_id: str):
    if not require_user(request):
        return RedirectResponse("/login")

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return templates.TemplateResponse(request, "case_exam.html", {"case": case, "active_tab": "exam"})


@app.get("/cases/{case_id}/summary")
async def case_summary(request: Request, case_id: str):
    if not require_user(request):
        return RedirectResponse("/login")

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return templates.TemplateResponse(request, "case_summary.html", {"case": case, "active_tab": "summary"})

@app.get("/api/status")
async def api_status():
    return {"status": "ok", "message": "Knee CDS API running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}

# the POST route — deterministic layer only (Option B), no LLM here
@app.post("/cases/{case_id}/exam", response_class=RedirectResponse)
async def case_exam_submit(request: Request, case_id: str):
    id_token = request.cookies.get("token")
    user_token = validate_firebase_token(id_token)
    if not user_token:
        return RedirectResponse("/")

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    form = await request.form()

    # helper: an unchecked checkbox is absent from the form, so
    # "present" means True and "absent" means False.
    def checked(field_name):
        return field_name in form

    # age comes in as a string; convert to int (default 0 if blank)
    age_str = form.get("age", "0")
    try:
        age = int(age_str)
    except ValueError:
        age = 0

    # --- build the red-flag input from the form ---
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

    # --- build the Ottawa input ---
    ottawa_input = OttawaInput(
        age=age,
        isolated_patella_tenderness=checked("isolated_patella_tenderness"),
        fibula_head_tenderness=checked("fibula_head_tenderness"),
        unable_to_flex_90=checked("unable_to_flex_90"),
        unable_to_weight_bear=checked("unable_to_weight_bear"),
    )

    # --- build the Pittsburgh input ---
    pittsburgh_input = PittsburghInput(
        mechanism_blunt_trauma_or_fall=checked("mechanism_blunt_trauma_or_fall"),
        age=age,
        unable_to_weight_bear=checked("unable_to_weight_bear"),
    )

    # --- run the deterministic layer (no LLM) ---
    red_flag_result = screen_red_flags(red_flag_input)
    ottawa_result = apply_ottawa_knee_rule(ottawa_input)
    pittsburgh_result = apply_pittsburgh_knee_rule(pittsburgh_input)

    # --- build a plain-dict assessment to store (dataclasses aren't
    #     directly JSON/Mongo friendly, so pull out the fields we need) ---
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
        # convenience flag the templates/case list can read quickly
        "red_flag_positive": red_flag_result.escalate_immediately,
    }

    store.save_assessment(case_id, assessment)

    # go to the summary tab, where the result (or the halt) is shown
    return RedirectResponse(f"/cases/{case_id}/summary", status_code=302)

# -----------------------------------------------------------
# POST /cases/{case_id}/suggest
# Runs the LLM agent to suggest special tests.
# This is the SLOW path (the language model runs here).
# It only runs if the safety screen already passed.
# -----------------------------------------------------------
@app.post("/cases/{case_id}/suggest", response_class=RedirectResponse)
async def case_suggest(request: Request, case_id: str):

    # check the user is logged in
    id_token = request.cookies.get("token")
    user_token = validate_firebase_token(id_token)
    if not user_token:
        return RedirectResponse("/")

    # load the case
    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # get the saved assessment (from when the exam was submitted)
    assessment = case.get("assessment")

    # SAFETY GUARD:
    # if there is no assessment yet, or a red flag fired,
    # do NOT run the agent. Just go back to the summary.
    if not assessment:
        return RedirectResponse(
            f"/cases/{case_id}/summary",
            status_code=status.HTTP_302_FOUND,
        )
    if assessment.get("red_flag_positive"):
        return RedirectResponse(
            f"/cases/{case_id}/summary",
            status_code=status.HTTP_302_FOUND,
        )

    # build the safety-facts text to hand the agent as settled facts
    ottawa_text = assessment["ottawa"]["rationale"]
    pittsburgh_text = assessment["pittsburgh"]["rationale"]

    safety_facts = (
        "ESTABLISHED FACTS (do not override):\n"
        "- Ottawa: " + ottawa_text + "\n"
        "- Pittsburgh: " + pittsburgh_text + "\n"
        "- Red-flag screen: negative."
    )

    # a simple fixed question for now
    query = (
        "Given an acute knee injury with no red flags, which physical "
        "special tests should be prioritised, and what does each assess?"
    )

    # run the agent (this is the slow part - can take 1 to 3 minutes)
    result = run_agent_only(query, safety_facts)

    # write the retrieval to the append-only audit log
    log_entry = {
        "query": query,
        "retrieved": result["retrieved"],
        "suggestion": result["suggestion"],
    }
    store.append_agent_log(case_id, log_entry)

    # save the suggestion onto the assessment so the summary can show it
    assessment["agent_suggestion"] = result["suggestion"]
    store.save_assessment(case_id, assessment)

    # go back to the summary, which will now show the suggestion
    return RedirectResponse(
        f"/cases/{case_id}/summary",
        status_code=status.HTTP_302_FOUND,
    )