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
import threading
from backend.rules.ottawa import apply_ottawa_knee_rule, OttawaInput
from backend.rules.pittsburgh import apply_pittsburgh_knee_rule, PittsburghInput
from backend.safety.red_flags import screen_red_flags, RedFlagInput
from backend.agent.orchestrator import run_agent_only, run_assessment
# add this import at the top of main.py with your other imports
from backend.safety.neuro_trigger import check_neuro_trigger
def _readable(value):
    if value in ("", "not_assessed", "not_done", None):
        return "not recorded"
    return value

def _case_fingerprint(case):
    # a short signature of the inputs a suggestion was built from
    # if this changes, the suggestion is out of date and must regenerate
    # assumes pymongo returns field order consistently, which it does
    return str(case.get('history', {})) + str(case.get('physical', {}))

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


# GET /cases/{case_id}/history
@app.get('/cases/{case_id}/history')
async def case_history(request: Request, case_id: str):
    id_token = request.cookies.get('token')
    user_token = validate_firebase_token(id_token)
    if not user_token:
        return RedirectResponse('/')

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail='Case not found')

    return templates.TemplateResponse(
        request, 'case_history.html',
        {'case': case, 'active_tab': 'history'},
    )


# POST /cases/{case_id}/history
@app.post('/cases/{case_id}/history', response_class=RedirectResponse)
async def case_history_submit(request: Request, case_id: str):
    id_token = request.cookies.get('token')
    user_token = validate_firebase_token(id_token)
    if not user_token:
        return RedirectResponse('/')

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail='Case not found')

    form = await request.form()

    # build history dict from form — every field has a safe default
    # checkboxes return the value only if checked; absent = 'no'
    history_dict = {
        # presenting complaint
        'involved_side': form.get('involved_side', 'unknown'),
        'chief_complaint': form.get('chief_complaint', ''),
        'activity_at_injury': form.get('activity_at_injury', 'other'),

        # mechanism
        'mechanism_type': form.get('mechanism_type', 'unknown'),
        'mechanism_description': form.get('mechanism_description', ''),
        'audible_pop': form.get('audible_pop', 'unsure'),

        # symptom profile
        'pain_location': form.get('pain_location', ''),
        'swelling_present': form.get('swelling_present', 'no'),
        'swelling_onset': form.get('swelling_onset', 'none'),
        'weight_bearing': form.get('weight_bearing', 'full'),

        # mechanical symptoms — checkboxes; absent means unchecked = 'no'
        'mech_sym_locking': form.get('mech_sym_locking', 'no'),
        'mech_sym_giving_way': form.get('mech_sym_giving_way', 'no'),
        'mech_sym_catching': form.get('mech_sym_catching', 'no'),
        'mech_sym_clicking': form.get('mech_sym_clicking', 'no'),
        'mech_sym_grinding': form.get('mech_sym_grinding', 'no'),

        # patient context
        'occupation': form.get('occupation', ''),
        'activity_level': form.get('activity_level', 'sedentary'),
        'previous_knee_injury': form.get('previous_knee_injury', 'no'),
        'previous_surgery': form.get('previous_surgery', 'no'),
        'surgery_detail': form.get('surgery_detail', ''),
        'patient_goal': form.get('patient_goal', 'daily_function'),
    }

    store.save_history(case_id, history_dict)

    # redirect to exam tab after saving
    return RedirectResponse(
        f'/cases/{case_id}/exam',
        status_code=status.HTTP_302_FOUND,
    )

    print(query)

@app.get("/cases/{case_id}/exam")
async def case_exam(request: Request, case_id: str):
    if not require_user(request):
        return RedirectResponse("/login")

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return templates.TemplateResponse(request, "case_exam.html", {"case": case, "active_tab": "exam"})

# POST /cases/{case_id}/chat
# synchronous RAG chat endpoint — single retrieval + single LLM call
@app.post('/cases/{case_id}/chat')
async def case_chat(request: Request, case_id: str):
    id_token = request.cookies.get('token')
    user_token = validate_firebase_token(id_token)
    if not user_token:
        return {'error': 'not authenticated'}

    case = store.get_case(case_id)
    if not case:
        return {'error': 'case not found'}

    form = await request.form()
    message = form.get('message', '').strip()

    if not message:
        return {'error': 'empty message'}

    print(f'chat message: {message}')

    from backend.agent.chat import run_chat
    response = run_chat(message, case)

    print(f'chat response: {response}')

    return {'response': response}

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
    # a positive red flag halts the pathway, so go straight to summary
    # where the escalation notice is displayed rather than continuing
    # to hands-on examination of a knee that needs escalation
    if red_flag_result.escalate_immediately:
        return RedirectResponse(f"/cases/{case_id}/summary", status_code=302)

    return RedirectResponse(f"/cases/{case_id}/physical", status_code=302)

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
    # build the safety-facts text to hand the agent as settled facts
    ottawa_text = assessment["ottawa"]["rationale"]
    pittsburgh_text = assessment["pittsburgh"]["rationale"]

    # check whether the Ottawa age criterion fired
    # triggered_criteria is a list of strings stored at assessment time
    # e.g. ["Age 55 or older", "Unable to flex to 90 degrees"]
    ottawa_triggered = assessment["ottawa"].get("triggered_criteria", [])
    
    age_criterion_fired = any("55" in c or "age" in c.lower() for c in ottawa_triggered)

    # build a patient context block to scope the agent's suggestions
    # this tells the agent what phase we are in and what NOT to suggest
    patient_context = (
        "PATIENT CONTEXT (use to scope your suggestions):\n"
        "- Clinical phase: Acute triage only.\n"
        "- Do NOT suggest return-to-sport tests, hop tests, single-leg\n"
        "  performance tests, or high-performance athletic screening.\n"
        "  These are only appropriate in later rehabilitation phases."
    )

    # if the Ottawa age criterion fired, add an explicit age instruction
    if age_criterion_fired:
        patient_context = patient_context + (
            "\n- Patient is aged 55 or older: adapt all test suggestions"
            " to be appropriate for this age group and acute presentation."
        )

    safety_facts = (
        "ESTABLISHED FACTS (do not override):\n"
        "- Ottawa: " + ottawa_text + "\n"
        "- Pittsburgh: " + pittsburgh_text + "\n"
        "- Red-flag screen: negative.\n\n"
        + patient_context
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
    
   # -----------------------------------------------------------
# POST /cases/{case_id}/suggest-async
# Kicks off the agent in a background thread and returns
# immediately. The browser polls /cases/{case_id}/suggest-status
# to find out when the result is ready.
# -----------------------------------------------------------
@app.post("/cases/{case_id}/suggest-async")
async def case_suggest_async(request: Request, case_id: str):

    # check login
    id_token = request.cookies.get("token")
    user_token = validate_firebase_token(id_token)
    if not user_token:
        return {"error": "not authenticated"}

    # load the case
    case = store.get_case(case_id)
    if not case:
        return {"error": "case not found"}

    assessment = case.get("assessment")

    # safety guard: no assessment or red flag fired - do nothing
    if not assessment:
        return {"status": "no_assessment"}
    if assessment.get("red_flag_positive"):
        return {"status": "red_flag_halted"}

    # only skip if the suggestion matches the data currently in the case
    # submitting physical findings changes the fingerprint, which forces a
    # regeneration rather than serving a suggestion built from history alone
    current_fingerprint = _case_fingerprint(case)

    if assessment.get("agent_suggestion") and \
       assessment.get("agent_suggestion_fingerprint") == current_fingerprint:
        return {"status": "already_done"}

    # mark the suggestion as pending so the UI shows the spinner
    assessment["agent_suggestion_status"] = "pending"
    store.save_assessment(case_id, assessment)

    # build the safety facts (same logic as case_suggest)
    ottawa_text = assessment["ottawa"]["rationale"]
    pittsburgh_text = assessment["pittsburgh"]["rationale"]

    ottawa_triggered = assessment["ottawa"].get("triggered_criteria", [])
    age_criterion_fired = any("55" in c or "age" in c.lower() for c in ottawa_triggered)

    patient_context = (
        "PATIENT CONTEXT (use to scope your suggestions):\n"
        "- Clinical phase: Acute triage only.\n"
        "- Do NOT suggest return-to-sport tests, hop tests, single-leg\n"
        "  performance tests, or high-performance athletic screening.\n"
        "  These are only appropriate in later rehabilitation phases."
    )

    if age_criterion_fired:
        patient_context = patient_context + (
            "\n- Patient is aged 55 or older: adapt all test suggestions"
            " to be appropriate for this age group and acute presentation."
        )

    safety_facts = (
        "ESTABLISHED FACTS (do not override):\n"
        "- Ottawa: " + ottawa_text + "\n"
        "- Pittsburgh: " + pittsburgh_text + "\n"
        "- Red-flag screen: negative.\n\n"
        + patient_context
    )

    # build a dynamic query from history if it exists,
    # otherwise fall back to the generic fixed query
    history = case.get('history')

    if history:
        # deterministic pre-processing — no LLM involved
        swelling_fact = ''
        if history.get('swelling_present') == 'yes':
            if history.get('swelling_onset') == 'immediate':
                swelling_fact = 'Immediate swelling — haemarthrosis possible.'
            elif history.get('swelling_onset') == 'delayed':
                swelling_fact = 'Delayed swelling — synovial effusion pattern, consider meniscal involvement.'

        weight_bearing_fact = ''
        if history.get('weight_bearing') == 'none':
            weight_bearing_fact = 'Patient non-weight-bearing — defer weight-bearing tests.'

        mech_syms_reported = [
            sym for sym in ['locking', 'giving_way', 'catching', 'clicking', 'grinding']
            if history.get('mech_sym_' + sym) == 'yes'
        ]
        mech_sym_str = ', '.join(mech_syms_reported) if mech_syms_reported else 'none reported'

        if history.get('patient_goal') != 'return_to_sport':
            patient_context = patient_context + (
                '\n- Patient goal is not return to sport: do NOT suggest'
                ' return-to-sport outcome measures or hop tests.'
            )

        pop_str = ''
        if history.get('audible_pop') == 'yes':
            pop_str = 'Audible/felt pop at injury — consider ACL involvement.'

        query = (
            f"Patient: {history.get('activity_level', 'unknown activity level')} individual, "
            f"involved {history.get('involved_side', 'unknown')} knee. "
            f"Activity at injury: {history.get('activity_at_injury', 'unknown')}. "
            f"Mechanism: {history.get('mechanism_type', 'unknown')} — "
            f"{history.get('mechanism_description', 'no description')}. "
            f"Pain location: {history.get('pain_location', 'not recorded')}. "
            f"Swelling: {history.get('swelling_onset', 'none')}. "
            f"Weight-bearing: {history.get('weight_bearing', 'unknown')}. "
            f"Mechanical symptoms: {mech_sym_str}. "
            f"Goal: {history.get('patient_goal', 'not recorded')}. "
            + (pop_str + ' ' if pop_str else '')
            + (swelling_fact + ' ' if swelling_fact else '')
            + (weight_bearing_fact + ' ' if weight_bearing_fact else '')
        )

    else:
        # no history recorded yet — use the generic fallback query
        query = (
            'Given an acute knee injury with no red flags. '

        )
     # addition — extend the query with physical findings if recorded
    # this block previously sat after a return statement in the history
    # route and never executed, so the agent never saw physical findings
    physical = case.get('physical')

    if physical:
        flex_inv = _readable(physical.get('rom_flexion_involved'))
        flex_uninv = _readable(physical.get('rom_flexion_uninvolved'))
        ext_inv = _readable(physical.get('rom_extension_involved'))

        # MMT — only include recorded grades
        mmt_findings = []
        mmt_map = {
            'mmt_quadriceps': 'quadriceps',
            'mmt_hamstrings': 'hamstrings',
            'mmt_hip_flexors': 'hip flexors',
            'mmt_hip_abductors': 'hip abductors',
            'mmt_hip_external_rotators': 'hip external rotators',
            'mmt_gastroc_soleus': 'gastroc/soleus',
        }
        for field, label in mmt_map.items():
            val = physical.get(field, '')
            if val:
                mmt_findings.append(f'{label} {val}/5')
        mmt_str = ', '.join(mmt_findings) if mmt_findings else 'not recorded'

        # special tests already done — the agent should not re-suggest these
        tests_done = []
        test_map = {
            'test_lachman': 'Lachman',
            'test_anterior_drawer': 'anterior drawer',
            'test_pivot_shift': 'pivot shift',
            'test_posterior_drawer': 'posterior drawer',
            'test_sag_sign': 'sag sign',
            'test_valgus_0': 'valgus stress 0°',
            'test_valgus_30': 'valgus stress 30°',
            'test_varus_0': 'varus stress 0°',
            'test_varus_30': 'varus stress 30°',
            'test_mcmurray': 'McMurray',
            'test_thessaly': 'Thessaly',
            'test_apley_compression': 'Apley compression',
            'test_apley_distraction': 'Apley distraction',
            'test_patellar_apprehension': 'patellar apprehension',
        }
        for field, label in test_map.items():
            val = physical.get(field, '')
            if val and val != 'not_done':
                tests_done.append(f'{label}: {val}')
        tests_str = ', '.join(tests_done) if tests_done else 'none recorded'

        # bony palpation points
        palp_findings = []
        palp_map = {
            'patellar_tenderness': 'patellar tenderness',
            'fibular_head_tenderness': 'fibular head tenderness',
            'tibial_tubercle_tenderness': 'tibial tubercle tenderness',
        }
        for field, label in palp_map.items():
            if physical.get(field) == 'yes':
                palp_findings.append(label)
        palp_str = ', '.join(palp_findings) if palp_findings else 'none recorded'

        physical_context = (
            f"Physical findings: gait {_readable(physical.get('gait'))}, "
            f"effusion {_readable(physical.get('effusion'))} "
            f"({_readable(physical.get('effusion_type'))}), "
            f"joint line tenderness {_readable(physical.get('joint_line_tenderness'))}, "
            f"collateral ligament line tenderness "
            f"{_readable(physical.get('collateral_tenderness'))}, "
            f"bony tenderness: {palp_str}, "
            f"temperature {_readable(physical.get('temperature'))}. "
            f"ROM: flexion {flex_inv} degrees (uninvolved {flex_uninv} degrees), "
            f"extension {ext_inv} degrees, "
            f"able to flex to 90°: {_readable(physical.get('able_to_flex_90'))}, "
            f"end feel flexion {_readable(physical.get('end_feel_flexion'))}, "
            f"pain-resistance sequence "
            f"{_readable(physical.get('pain_resistance_sequence'))}. "
            f"Strength: {mmt_str}. "
            f"Special tests already performed: {tests_str}. "
        )

        if physical.get('neuro_triggered') == 'yes':
            physical_context += (
                f"Neuro screen triggered "
                f"({physical.get('neuro_trigger_reason', '')}): "
                f"L3 sensation {_readable(physical.get('sensation_l3'))}, "
                f"L4 {_readable(physical.get('sensation_l4'))}, "
                f"L5 {_readable(physical.get('sensation_l5'))}, "
                f"S1 {_readable(physical.get('sensation_s1'))}, "
                f"patella reflex {_readable(physical.get('reflex_patella'))}, "
                f"achilles reflex {_readable(physical.get('reflex_achilles'))}, "
                f"peroneal dorsiflexion "
                f"{_readable(physical.get('peroneal_dorsiflexion'))}. "
            )

        query = query + physical_context

    # the question always goes last so the agent reads all findings first
    query = query + 'Which special tests should be prioritised and in what order?'

    print(query)

  
  

    # run the slow agent call in a background thread so we return immediately
    def run_in_background():
        try:
            result = run_agent_only(query, safety_facts)

            # write to audit log
            log_entry = {
                "query": query,
                "retrieved": result["retrieved"],
                "suggestion": result["suggestion"],
            }
            store.append_agent_log(case_id, log_entry)

            # save the suggestion and mark status as done
            assessment["agent_suggestion"] = result["suggestion"]
            assessment["agent_suggestion_fingerprint"] = current_fingerprint
            assessment["agent_suggestion_status"] = "done"
            store.save_assessment(case_id, assessment)
            
        except Exception as e:
            print(f"agent error: {e}")
            # mark as failed so the UI can show an error message
            assessment["agent_suggestion_status"] = "error"
            store.save_assessment(case_id, assessment)

    thread = threading.Thread(target=run_in_background, daemon=True)
    thread.start()

    # return immediately — the browser will poll for the result
    return {"status": "pending"}


# -----------------------------------------------------------
# GET /cases/{case_id}/suggest-status
# Polled by the browser every few seconds to check whether
# the agent has finished. Returns the suggestion once ready.
# -----------------------------------------------------------
@app.get("/cases/{case_id}/suggest-status")
async def case_suggest_status(request: Request, case_id: str):

    # check login
    id_token = request.cookies.get("token")
    user_token = validate_firebase_token(id_token)
    if not user_token:
        return {"status": "error", "message": "not authenticated"}

    case = store.get_case(case_id)
    if not case:
        return {"status": "error", "message": "case not found"}

    assessment = case.get("assessment", {})
    suggestion_status = assessment.get("agent_suggestion_status", "idle")
    suggestion = assessment.get("agent_suggestion", "")

    return {
        "status": suggestion_status,   # "pending" | "done" | "error" | "idle"
        "suggestion": suggestion,
    }
    
    # GET /cases/{case_id}/physical
@app.get('/cases/{case_id}/physical')
async def case_physical(request: Request, case_id: str):
    id_token = request.cookies.get('token')
    user_token = validate_firebase_token(id_token)
    if not user_token:
        return RedirectResponse('/')

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail='Case not found')

    return templates.TemplateResponse(
        request, 'case_physical.html',
        {'case': case, 'active_tab': 'physical'},
    )


# POST /cases/{case_id}/physical
@app.post('/cases/{case_id}/physical', response_class=RedirectResponse)
async def case_physical_submit(request: Request, case_id: str):
    id_token = request.cookies.get('token')
    user_token = validate_firebase_token(id_token)
    if not user_token:
        return RedirectResponse('/')

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail='Case not found')

    form = await request.form()

    physical_dict = {
        # observation
        'gait': form.get('gait', ''),
        'visible_swelling': form.get('visible_swelling', ''),
        'bruising': form.get('bruising', 'no'),
        'muscle_wasting': form.get('muscle_wasting', 'no'),
        'altered_sensation_reported': form.get('altered_sensation_reported', 'no'),
        'alignment_notes': form.get('alignment_notes', ''),

        # palpation
        'effusion': form.get('effusion', ''),
        'effusion_type': form.get('effusion_type', ''),
        'joint_line_tenderness': form.get('joint_line_tenderness', ''),
        'temperature': form.get('temperature', ''),
        'popliteal_tenderness': form.get('popliteal_tenderness', 'no'),

        # addition 1 — bony palpation points, documentation only
        # ottawa reads its own fields from the exam form, these do not feed the gate
        'patellar_tenderness': form.get('patellar_tenderness', 'not_assessed'),
        'fibular_head_tenderness': form.get('fibular_head_tenderness', 'not_assessed'),
        'tibial_tubercle_tenderness': form.get('tibial_tubercle_tenderness', 'not_assessed'),
        'collateral_tenderness': form.get('collateral_tenderness', 'not_assessed'),

        # ROM — involved
        'rom_flexion_involved': form.get('rom_flexion_involved', ''),
        'rom_extension_involved': form.get('rom_extension_involved', ''),
        'extension_lag': form.get('extension_lag', 'no'),
        'end_feel_flexion': form.get('end_feel_flexion', ''),
        'end_feel_extension': form.get('end_feel_extension', ''),
        'pain_on_movement': form.get('pain_on_movement', ''),
        'pain_resistance_sequence': form.get('pain_resistance_sequence', ''),

        # addition 2 — recorded explicitly, a blank rom value is not the same as unable
        'able_to_flex_90': form.get('able_to_flex_90', 'not_assessed'),

        # ROM — uninvolved
        'rom_flexion_uninvolved': form.get('rom_flexion_uninvolved', ''),
        'rom_extension_uninvolved': form.get('rom_extension_uninvolved', ''),

        # flexibility
        'flexibility_hamstrings': form.get('flexibility_hamstrings', ''),
        'flexibility_gastroc_soleus': form.get('flexibility_gastroc_soleus', ''),

        # MMT
        'mmt_quadriceps': form.get('mmt_quadriceps', ''),
        'mmt_hamstrings': form.get('mmt_hamstrings', ''),
        'mmt_hip_flexors': form.get('mmt_hip_flexors', ''),
        'mmt_hip_abductors': form.get('mmt_hip_abductors', ''),
        'mmt_hip_external_rotators': form.get('mmt_hip_external_rotators', ''),
        'mmt_gastroc_soleus': form.get('mmt_gastroc_soleus', ''),

        # special tests
        'test_lachman': form.get('test_lachman', 'not_done'),
        'test_anterior_drawer': form.get('test_anterior_drawer', 'not_done'),
        'test_pivot_shift': form.get('test_pivot_shift', 'not_done'),
        'test_posterior_drawer': form.get('test_posterior_drawer', 'not_done'),
        'test_sag_sign': form.get('test_sag_sign', 'not_done'),
        'test_valgus_0': form.get('test_valgus_0', 'not_done'),
        'test_valgus_30': form.get('test_valgus_30', 'not_done'),
        'test_varus_0': form.get('test_varus_0', 'not_done'),
        'test_varus_30': form.get('test_varus_30', 'not_done'),
        'test_mcmurray': form.get('test_mcmurray', 'not_done'),
        'test_thessaly': form.get('test_thessaly', 'not_done'),
        'test_apley_compression': form.get('test_apley_compression', 'not_done'),
        'test_apley_distraction': form.get('test_apley_distraction', 'not_done'),
        'test_patellar_apprehension': form.get('test_patellar_apprehension', 'not_done'),
        'special_tests_notes': form.get('special_tests_notes', ''),

        # neuro — manual trigger from form
        'neuro_screen_requested': form.get('neuro_screen_requested', 'no'),
        'neuro_triggered': 'no',
        'neuro_trigger_reason': '',

        # neuro findings — only populated if triggered
        'sensation_l3': form.get('sensation_l3', ''),
        'sensation_l4': form.get('sensation_l4', ''),
        'sensation_l5': form.get('sensation_l5', ''),
        'sensation_s1': form.get('sensation_s1', ''),
        'reflex_patella': form.get('reflex_patella', ''),
        'reflex_achilles': form.get('reflex_achilles', ''),
        'peroneal_dorsiflexion': form.get('peroneal_dorsiflexion', ''),
        'balance_single_leg': form.get('balance_single_leg', ''),
        'mechanoreceptor_involvement': form.get('mechanoreceptor_involvement', ''),
        'neuro_notes': form.get('neuro_notes', ''),
    }

    # run deterministic neuro trigger check
    triggered, reason = check_neuro_trigger(physical_dict)
    physical_dict['neuro_triggered'] = triggered
    physical_dict['neuro_trigger_reason'] = reason

    print(f'neuro trigger: {triggered} — {reason}')
    print('trigger type:', type(triggered))          #
    # addition 3 — clear neuro findings if the screen was not indicated
    # a hidden select still posts its first option, so without this a case
    # would record a full normal neuro exam that nobody actually performed
    # the client side javascript disables these fields but can be bypassed
    neuro_finding_fields = [
        'sensation_l3', 'sensation_l4', 'sensation_l5', 'sensation_s1',
        'reflex_patella', 'reflex_achilles', 'peroneal_dorsiflexion',
        'balance_single_leg', 'mechanoreceptor_involvement', 'neuro_notes'
    ]

    if triggered != 'yes':
        for field in neuro_finding_fields:
            physical_dict[field] = ''
        print('neuro findings cleared — screen not indicated')

    store.save_physical(case_id, physical_dict)

    return RedirectResponse(
        f'/cases/{case_id}/summary',
        status_code=status.HTTP_302_FOUND,
    )