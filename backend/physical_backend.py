"""
Physical Examination Router — Knee CDS
===================================================================
Owns every /cases/{case_id}/physical* route. These routes are NOT
defined in main.py; defining them in both places previously produced
two competing implementations resolved silently by registration order.

Mounted in main.py as:
    from backend.physical_backend import router as physical_router
    app.state.store = store
    app.state.templates = templates
    app.include_router(physical_router)

The neuro trigger is imported, never redefined. It is part of the
deterministic safety layer and has its own unit test suite; a second
copy living in this file meant the tested implementation was not the
one running.

Access control
--------------
The router previously carried Depends(require_user), which established
that the caller was signed in and nothing further. Any physiotherapist
could read or overwrite any other physiotherapist's examination findings
by knowing the case id, and a head-of-department account could write to
records it is only meant to observe. Each route now loads the case
through backend.case_access instead, which checks the case belongs to
the caller and, on the five POSTs, that the caller is not an admin.

=====================================================================
DEVIATIONS FROM PaaS SCAFFOLD  (Examples 07-10, main.py)
=====================================================================
D1  Scaffold has no routers - every route hangs off app in main.py.
    This file is an APIRouter mounted by main.py so the five
    examination sections stay together with the field contracts they
    depend on. Retained from the scaffold: `form = await
    request.form()`, RedirectResponse with starlette.status codes, and
    templates.TemplateResponse for rendering.

D2  Scaffold repeats validateFirebaseToken() inline per file. This
    router does no token handling of its own; it delegates entirely to
    backend/case_access.py.

D5  Scaffold uses camelCase handlers. snake_case here per PEP 8, with
    the single exception of formatPhysicalForAgent, which is imported
    from a module that predates that convention and is left alone
    rather than renamed mid-project.

D6  Scaffold enforces ownership inside the query itself. Here the case
    is loaded and then judged, because read access extends to admins
    and write access does not - see case_access.py.

D11 Scaffold reads its collections from module-level globals. This
    router reads store and templates off request.app.state, set by
    main.py before include_router. That indirection exists so the
    router can be mounted without importing main and creating a cycle.
    Deliberate, but it is why the two must be assigned before mounting.
=====================================================================
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse
import starlette.status as status

from backend.case_access import load_case_for_read, load_case_for_write
from backend.safety.neuro_trigger import check_neuro_trigger, check_inhibition
from backend.agent.physical_context import formatPhysicalForAgent

# No router-level Depends. Authorisation is per-route because read and
# write differ: a head of department may view an examination but must
# not alter one, and a single router-wide dependency cannot express
# that distinction.
router = APIRouter()


# ===================================================================
# Field contracts
# ===================================================================

# Findings that are only meaningful when the neuro screen is indicated.
# Cleared when the trigger evaluates false.
NEURO_FINDING_FIELDS = [
    "sensation_l3",
    "sensation_l4",
    "sensation_l5",
    "sensation_s1",
    "reflex_patella",
    "reflex_achilles",
    "peroneal_dorsiflexion",
    "balance_single_leg",
    "mechanoreceptor_involvement",
    "neuro_notes",
]

# -------------------------------------------------------------------
# BOOLEAN FIELDS — two-state only.
#
# These templates do not use real checkboxes. They use JS-driven hidden
# inputs that are ALWAYS present in the POST, carrying "yes" or "no".
# The previous coercion was:
#
#     "yes" if val and val not in ("", "off", "false") else ""
#
# which turned "no" into "yes", because "no" is truthy and absent from
# that tuple. Every toggle read as ON regardless of its actual state.
#
# Fields removed from these lists because they are NOT two-state:
#   able_to_flex_90        — not_assessed / yes / no  (Ottawa criterion)
#   popliteal_tenderness   — not_assessed / no / yes  (select)
#   mechanoreceptor_involvement — not_suspected / suspected
#   pain_on_flexion, pain_on_extension — never posted; both switches
#                            write into the single pain_on_movement field
#   neuro_screen_requested (from observation) — no control on that page,
#                            so every observation save wiped a manual
#                            request made on the neuro page
# -------------------------------------------------------------------
BOOLEAN_FIELDS = {
    "observation": [
        "bruising",
        "muscle_wasting",
        "altered_sensation_reported",
    ],
    "rom": [
        "extension_lag",
    ],
    "mmt": [],
    "special": [],
    "neuro": [
        "neuro_screen_requested",
    ],
}

# Values accepted as true. Anything else, including "no", "not_assessed"
# and "not_suspected", is false.
TRUTHY = ("yes", "on", "true", "1", True)

# -------------------------------------------------------------------
# INT FIELDS — stored as numbers, not strings.
#
# Form posts are always strings, so "135" - "90" raised a TypeError in
# the summary template's flexion-deficit calculation and took the whole
# page down as soon as ROM was saved. Coercing at the boundary means the
# summary arithmetic, the agent formatter and the gauge init all read
# one type.
#
# Extension may legitimately be negative (hyperextension), so the
# coercion must accept a leading minus.
# -------------------------------------------------------------------
INT_FIELDS = {
    "rom": [
        "rom_flexion_involved",
        "rom_extension_involved",
        "rom_flexion_uninvolved",
        "rom_extension_uninvolved",
    ],
}

# MMT grades stay as strings. The template renders its active state with
# {% if case.physical.get(field) == g %} where g is a string literal
# '0'–'5'; storing ints would break that comparison silently. Worth
# migrating later, template and store together, not one without the other.

# ===================================================================
# Value coercion
# ===================================================================

def _coerce_bool(val) -> str:
    """
    Two-state field to 'yes' or ''.

    Explicit allow-list. The hidden inputs these templates use are
    always present in the POST carrying 'yes' or 'no', so any coercion
    based on truthiness alone turns 'no' into 'yes'.
    """
    return "yes" if val in TRUTHY else ""


def _coerce_int(val):
    """
    Numeric field to int, preserving None for 'not recorded'.

    Accepts a leading minus: knee extension may legitimately be
    negative where there is hyperextension.
    """
    if val is None or val == "":
        return None
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        print(f"[physical] could not coerce to int: {val!r}")
        return None


# ===================================================================
# Section merge
# ===================================================================

def _merge_physical_section(
    case_id: str,
    form: dict,
    section_fields: list,
    store,
    case: dict,
    boolean_fields: list | None = None,
    int_fields: list | None = None,
    is_neuro_section: bool = False,
) -> dict:
    """
    Merge one section's posted fields into the existing physical dict,
    re-run the deterministic neuro trigger, and persist.

    Only fields belonging to this section are touched. Everything
    recorded on the other four pages is left alone.

    The case is passed in rather than re-fetched. It has already been
    loaded and authorised by the calling route, and loading it a second
    time here would mean the document written to is not necessarily the
    document that was checked.
    """
    boolean_fields = boolean_fields or []
    int_fields = int_fields or []

    physical = case.get("physical") or {}

    for field in section_fields:
        val = form.get(field)

        if field in boolean_fields:
            # Always written, because an absent two-state field means
            # 'off', not 'unchanged'.
            physical[field] = _coerce_bool(val)

        elif field in int_fields:
            if val is not None:
                physical[field] = _coerce_int(val)

        else:
            # Guard against the literal string "None" reaching the
            # store. Templates using case.physical.get(...) rendered
            # value="None" for unrecorded fields before the Jinja
            # finalize hook was added in main.py; this stops any
            # already-stored instances propagating on re-save.
            if val is not None:
                physical[field] = "" if val == "None" else val

    # ---------------------------------------------------------------
    # Recording the neuro screen IS requesting it.
    #
    # Without this, a clinician fills in dermatomes and reflexes, saves,
    # and the trigger evaluates false — so every finding they just
    # entered is cleared below with no explanation. Setting the request
    # flag when findings are present makes the save honest and keeps the
    # clinician's judgement in the audit trail alongside the automatic
    # triggers.
    # ---------------------------------------------------------------
    if is_neuro_section:
        has_findings = any(
            physical.get(f) not in (None, "", "not_assessed", "not_suspected")
            for f in NEURO_FINDING_FIELDS
        )
        if has_findings:
            physical["neuro_screen_requested"] = "yes"

    # ---------------------------------------------------------------
    # Deterministic trigger — pure Python, no LLM.
    # Runs against the FULL merged dict, so a low MMT grade saved on the
    # strength page or a positive posterior drawer saved on the special
    # tests page both reach it.
    #
    # Contract: returns ('yes'|'no', reason).
    # ---------------------------------------------------------------
    triggered, reason = check_neuro_trigger(physical)
    physical["neuro_triggered"] = triggered
    physical["neuro_trigger_reason"] = reason

    print(f"[neuro trigger] {triggered} — {reason or 'no indication'}")

    if triggered != "yes":
        cleared = [f for f in NEURO_FINDING_FIELDS if physical.get(f)]
        for f in NEURO_FINDING_FIELDS:
            physical[f] = ""
        if cleared:
            print(f"[neuro trigger] cleared (not indicated): {', '.join(cleared)}")

    # Deterministic derivation, computed at save time. A low grade
    # suppressed by pain or effusion is not a neurological finding, but
    # it is still a finding — physical_context reads inhibition_noted to
    # tell the agent to re-test when the joint settles. The field has
    # been read by that module all along with nothing ever writing it.
    physical["inhibition_noted"] = check_inhibition(physical)

    store.save_physical(case_id, physical)
    return physical


# ===================================================================
# Routes
# ===================================================================

def _render(request: Request, template: str, case_id: str):
    """
    Shared page render — every GET below is the same three steps.

    load_case_for_read raises 401/404 rather than returning None, so
    the previous explicit `if case is None` check is now redundant:
    an unauthorised or absent case never reaches the render.
    """
    user, case = load_case_for_read(request, case_id)
    return request.app.state.templates.TemplateResponse(
        request, template,
        {
            "case": case,
            "active_tab": "physical",
            "user": user,
            # False rather than absent: admin.py renders these same
            # templates with True, and an Undefined flag makes
            # `{% if not is_admin_view %}` true by accident.
            "is_admin_view": False,
        },
    )


def _back_to_summary(case_id: str):
    return RedirectResponse(
        f"/cases/{case_id}/physical",
        status_code=status.HTTP_302_FOUND,
    )


# ------------------------------------------------------------------
# Summary / landing page
# ------------------------------------------------------------------
@router.get("/cases/{case_id}/physical")
async def physical_summary(request: Request, case_id: str):
    return _render(request, "physical_summary.html", case_id)


# ------------------------------------------------------------------
# SECTION 1: OBSERVATION & PALPATION
# ------------------------------------------------------------------
@router.get("/cases/{case_id}/physical/edit/observation")
async def physical_edit_observation(request: Request, case_id: str):
    return _render(request, "physical_edit_observation.html", case_id)


@router.post("/cases/{case_id}/physical/edit/observation",
             response_class=RedirectResponse)
async def physical_save_observation(request: Request, case_id: str):
    user, case = load_case_for_write(request, case_id)
    form = dict(await request.form())
    _merge_physical_section(
        case_id, form,
        section_fields=[
            "gait", "visible_swelling", "bruising", "muscle_wasting",
            "altered_sensation_reported", "alignment_notes",
            # joint_temperature, NOT temperature. The template posts
            # joint_temperature and the summary reads it, but this list
            # said 'temperature', so the field was never saved.
            "joint_temperature", "effusion", "effusion_type",
            "joint_line_tenderness", "popliteal_tenderness",
            "patellar_tenderness", "fibular_head_tenderness",
            "tibial_tubercle_tenderness", "collateral_tenderness",
        ],
        boolean_fields=BOOLEAN_FIELDS["observation"],
        store=request.app.state.store,
        case=case,
    )
    return _back_to_summary(case_id)


# ------------------------------------------------------------------
# SECTION 2: RANGE OF MOTION & FLEXIBILITY
# ------------------------------------------------------------------
@router.get("/cases/{case_id}/physical/edit/rom")
async def physical_edit_rom(request: Request, case_id: str):
    return _render(request, "physical_edit_rom.html", case_id)


@router.post("/cases/{case_id}/physical/edit/rom",
             response_class=RedirectResponse)
async def physical_save_rom(request: Request, case_id: str):
    user, case = load_case_for_write(request, case_id)
    form = dict(await request.form())
    _merge_physical_section(
        case_id, form,
        section_fields=[
            "rom_flexion_involved", "rom_extension_involved",
            "rom_flexion_uninvolved", "rom_extension_uninvolved",
            "rotation_medial", "rotation_lateral",
            "extension_lag", "end_feel_flexion", "end_feel_extension",
            "pain_on_movement", "pain_resistance_sequence",
            "able_to_flex_90",
            "flexibility_hamstrings", "flexibility_gastroc_soleus",
        ],
        boolean_fields=BOOLEAN_FIELDS["rom"],
        int_fields=INT_FIELDS["rom"],
        store=request.app.state.store,
        case=case,
    )
    return _back_to_summary(case_id)


# ------------------------------------------------------------------
# SECTION 3: MUSCLE STRENGTH (MMT)
# ------------------------------------------------------------------
@router.get("/cases/{case_id}/physical/edit/mmt")
async def physical_edit_mmt(request: Request, case_id: str):
    return _render(request, "physical_edit_mmt.html", case_id)


@router.post("/cases/{case_id}/physical/edit/mmt",
             response_class=RedirectResponse)
async def physical_save_mmt(request: Request, case_id: str):
    user, case = load_case_for_write(request, case_id)
    form = dict(await request.form())
    _merge_physical_section(
        case_id, form,
        section_fields=[
            "mmt_quadriceps", "mmt_hamstrings", "mmt_hip_flexors",
            "mmt_hip_abductors", "mmt_hip_external_rotators",
            "mmt_gastroc_soleus",
            # Limiters. A grade of 2/5 from pain guarding is not the
            # same finding as a grade of 2/5 from nerve involvement;
            # check_neuro_trigger reads these before firing on a grade.
            "mmt_quadriceps_limiter", "mmt_hamstrings_limiter",
            "mmt_hip_flexors_limiter", "mmt_hip_abductors_limiter",
            "mmt_hip_external_rotators_limiter",
            "mmt_gastroc_soleus_limiter",
            "mmt_notes",
        ],
        store=request.app.state.store,
        case=case,
    )
    return _back_to_summary(case_id)


# ------------------------------------------------------------------
# SECTION 4: SPECIAL TESTS
# ------------------------------------------------------------------
@router.get("/cases/{case_id}/physical/edit/special")
async def physical_edit_special(request: Request, case_id: str):
    return _render(request, "physical_edit_special.html", case_id)


@router.post("/cases/{case_id}/physical/edit/special",
             response_class=RedirectResponse)
async def physical_save_special(request: Request, case_id: str):
    user, case = load_case_for_write(request, case_id)
    form = dict(await request.form())
    _merge_physical_section(
        case_id, form,
        section_fields=[
            "test_lachman", "test_anterior_drawer", "test_pivot_shift",
            "test_posterior_drawer", "test_sag_sign",
            "test_valgus_0", "test_valgus_30",
            "test_varus_0", "test_varus_30",
            "test_mcmurray", "test_thessaly",
            "test_apley_compression", "test_apley_distraction",
            "test_patellar_apprehension", "special_tests_notes",
        ],
        store=request.app.state.store,
        case=case,
    )
    return _back_to_summary(case_id)


# ------------------------------------------------------------------
# SECTION 5: NEUROLOGICAL SCREEN
# ------------------------------------------------------------------
@router.get("/cases/{case_id}/physical/edit/neuro")
async def physical_edit_neuro(request: Request, case_id: str):
    return _render(request, "physical_edit_neuro.html", case_id)


@router.post("/cases/{case_id}/physical/edit/neuro",
             response_class=RedirectResponse)
async def physical_save_neuro(request: Request, case_id: str):
    user, case = load_case_for_write(request, case_id)
    form = dict(await request.form())
    _merge_physical_section(
        case_id, form,
        section_fields=[
            "neuro_screen_requested",
            "sensation_l3", "sensation_l4", "sensation_l5", "sensation_s1",
            "reflex_patella", "reflex_achilles",
            "peroneal_dorsiflexion", "balance_single_leg",
            "mechanoreceptor_involvement", "neuro_notes",
        ],
        boolean_fields=BOOLEAN_FIELDS["neuro"],
        # Recording the screen is requesting it — see the merge helper.
        is_neuro_section=True,
        store=request.app.state.store,
        case=case,
    )
    return _back_to_summary(case_id)


# ------------------------------------------------------------------
# Agent context endpoint (read-only, JSON)
# Debug aid: shows exactly what the agent will be handed for this case.
# ------------------------------------------------------------------
@router.get("/api/cases/{case_id}/physical/agent-context")
async def physical_agent_context(request: Request, case_id: str):
    user, case = load_case_for_read(request, case_id)
    return {"agent_context": formatPhysicalForAgent(case.get("physical") or {})}

