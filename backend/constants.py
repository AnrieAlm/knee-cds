"""
backend/constants.py

Single source of truth for role, department and grade vocabularies.

Imported by:
  - signup route (to render and validate the dropdowns)
  - backend/auth.py (require_admin)
  - backend/admin.py (filter options)
  - templates (via the label maps, so display strings never live in Jinja)

Machine keys are stored in MongoDB. Labels are for display only.
Never store a label. Never render a raw key.
"""

# ---------------------------------------------------------------------------
# Roles - access control ONLY. Deliberately binary.
# ---------------------------------------------------------------------------
# This is intentionally NOT graded. Seniority does not grant permissions;
# it is descriptive metadata (see GRADES below). Keeping the auth surface
# to two values means every access decision in the system is answerable by
# a single equality check, which is a defensible security property.

ROLE_PHYSIO = "physio"
ROLE_ADMIN = "admin"

ROLES = {
    ROLE_PHYSIO: "Physiotherapist",
    ROLE_ADMIN: "Head of Department",
}

# NOTE: ROLE_ADMIN must never appear in the signup dropdown.
# Admin accounts are provisioned by running promote_admin() in the
# migration script. Self-selectable admin is an obvious audit finding.
SIGNUP_ROLES = {ROLE_PHYSIO: ROLES[ROLE_PHYSIO]}


# ---------------------------------------------------------------------------
# Departments - clinical service the assessment took place in
# ---------------------------------------------------------------------------

DEPT_UNASSIGNED = "unassigned"

DEPARTMENTS = {
    "musculoskeletal": "Musculoskeletal / Outpatients",
    "orthopaedics": "Orthopaedics",
    "neurology": "Neurology",
    "paediatrics": "Paediatrics",
    "sports": "Sports Medicine",
    "emergency": "Emergency Department",
    DEPT_UNASSIGNED: "Unassigned",
}

# Shown at signup - "unassigned" is a backfill value, not a choice a
# new user should be able to make.
SIGNUP_DEPARTMENTS = {
    k: v for k, v in DEPARTMENTS.items() if k != DEPT_UNASSIGNED
}


# ---------------------------------------------------------------------------
# Grades - seniority. Descriptive only, grants nothing.
# ---------------------------------------------------------------------------
# Terminology follows HSE / Irish public health service titles rather than
# the informal "junior physio". Check this against how Alan and your
# interview participants phrase it before you commit to the enum - an
# examiner with clinical background will read these labels closely.

GRADE_UNKNOWN = "unknown"

GRADES = {
    "student": "Student Physiotherapist",
    "intern": "Intern",
    "staff_grade": "Staff Grade Physiotherapist",
    "senior": "Senior Physiotherapist",
    "clinical_specialist": "Clinical Specialist",
    "manager": "Physiotherapy Manager",
    GRADE_UNKNOWN: "Not recorded",
}

SIGNUP_GRADES = {k: v for k, v in GRADES.items() if k != GRADE_UNKNOWN}

# The cohort the system is designed for. Used by the admin dashboard's
# "supervision view" filter - the single most clinically useful thing the
# head of department can do is narrow to the least experienced staff.
SUPERVISION_GRADES = ("student", "intern", "staff_grade")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def is_valid_department(value: str) -> bool:
    return value in DEPARTMENTS


def is_valid_grade(value: str) -> bool:
    return value in GRADES


def is_valid_role(value: str) -> bool:
    return value in ROLES


def department_label(value: str) -> str:
    """Safe display lookup - unknown keys render readably rather than blank."""
    return DEPARTMENTS.get(value, DEPARTMENTS[DEPT_UNASSIGNED])


def grade_label(value: str) -> str:
    return GRADES.get(value, GRADES[GRADE_UNKNOWN])


def role_label(value: str) -> str:
    return ROLES.get(value, ROLES[ROLE_PHYSIO])

def is_valid_modality(value: str) -> bool:
    return value in MODALITIES


def is_valid_side(value: str) -> bool:
    return value in SIDES


def modality_label(value: str) -> str:
    return MODALITY_LABELS.get(value, MODALITY_LABELS[MODALITY_OTHER])


def side_label(value: str) -> str:
    return SIDE_LABELS.get(value, SIDE_LABELS[SIDE_NOT_STATED])


def inv_status_label(value: str) -> str:
    return INV_STATUS_LABELS.get(value, INV_STATUS_LABELS[INV_STATUS_FAILED])


def inv_visual_class(value: str) -> str:
    return INV_VISUAL_CLASS.get(value, "inv-problem")

# ---------------------------------------------------------------------------
# Investigations
# ---------------------------------------------------------------------------

# Modality
MODALITY_XRAY = "XRAY"
MODALITY_MRI = "MRI"
MODALITY_CT = "CT"
MODALITY_ULTRASOUND = "ULTRASOUND"
MODALITY_OTHER = "OTHER"

MODALITIES = (
    MODALITY_XRAY,
    MODALITY_MRI,
    MODALITY_CT,
    MODALITY_ULTRASOUND,
    MODALITY_OTHER,
)

MODALITY_LABELS = {
    MODALITY_XRAY: "Radiograph (X-ray)",
    MODALITY_MRI: "MRI",
    MODALITY_CT: "CT",
    MODALITY_ULTRASOUND: "Ultrasound",
    MODALITY_OTHER: "Other",
}

# Side
SIDE_LEFT = "LEFT"
SIDE_RIGHT = "RIGHT"
SIDE_BILATERAL = "BILATERAL"
SIDE_NOT_STATED = "NOT_STATED"

SIDES = (SIDE_LEFT, SIDE_RIGHT, SIDE_BILATERAL, SIDE_NOT_STATED)

SIDE_LABELS = {
    SIDE_LEFT: "Left",
    SIDE_RIGHT: "Right",
    SIDE_BILATERAL: "Bilateral",
    SIDE_NOT_STATED: "Side not stated",
}

# Provenance of the record
INV_SOURCE_UPLOADED_REPORT = "UPLOADED_REPORT"
INV_SOURCE_MANUAL_ENTRY = "MANUAL_ENTRY"

INV_SOURCES = (INV_SOURCE_UPLOADED_REPORT, INV_SOURCE_MANUAL_ENTRY)

# Extraction / verification lifecycle
INV_STATUS_EXTRACTED = "EXTRACTED"   # model output, NOT yet human-verified
INV_STATUS_VERIFIED = "VERIFIED"     # human confirmed against source document
INV_STATUS_MANUAL = "MANUAL"         # typed directly by the physio
INV_STATUS_FAILED = "FAILED"         # extraction errored
INV_STATUS_REJECTED = "REJECTED"     # not a report (e.g. a raw image study)

INV_STATUSES = (
    INV_STATUS_EXTRACTED,
    INV_STATUS_VERIFIED,
    INV_STATUS_MANUAL,
    INV_STATUS_FAILED,
    INV_STATUS_REJECTED,
)

INV_STATUS_LABELS = {
    INV_STATUS_EXTRACTED: "Extracted - verify against source",
    INV_STATUS_VERIFIED: "Reported finding - verified",
    INV_STATUS_MANUAL: "Reported finding - entered manually",
    INV_STATUS_FAILED: "Extraction failed - enter manually",
    INV_STATUS_REJECTED: "Not a report - enter findings manually",
}

# Only these statuses may reach the agent. This is the provenance gate.
INV_AGENT_VISIBLE_STATUSES = (INV_STATUS_VERIFIED, INV_STATUS_MANUAL)

# Visual class driving the badge in templates.
# Deliberately distinct from SYSTEM_CHECK (navy) and SUGGESTED (violet).
INV_VISUAL_CLASS = {
    INV_STATUS_EXTRACTED: "inv-unverified",   # amber outline, document icon
    INV_STATUS_VERIFIED: "inv-verified",      # slate solid
    INV_STATUS_MANUAL: "inv-verified",
    INV_STATUS_FAILED: "inv-problem",
    INV_STATUS_REJECTED: "inv-problem",
}

# Scope caps
INV_MAX_PER_CASE = 3
INV_DEFAULT_BODY_PART = "KNEE"
INV_ALLOWED_UPLOAD_TYPES = ("application/pdf", "image/png", "image/jpeg")
INV_MAX_UPLOAD_BYTES = 15 * 1024 * 1024   # under Groq's 20MB request ceiling

# Vision model used for extraction only. Never used for image interpretation.
INV_EXTRACTION_MODEL = "qwen/qwen3.6-27b"

# Context-window guards on free text pulled from reports
INV_MAX_FINDINGS_CHARS = 1500
INV_MAX_IMPRESSION_CHARS = 600

# Citation source types (summary already cites the corpus; reports are a
# second source type)
CITATION_SOURCE_CORPUS = "CORPUS"
CITATION_SOURCE_INVESTIGATION = "INVESTIGATION"
# ---------------------------------------------------------------------------
# Deferrals - why a recordable item was not recorded
# ---------------------------------------------------------------------------
#
# Cygnus previously stored two states per special test: a result, or
# 'not_done'. The hidden input defaulted to 'not_done' on every one of
# the fourteen tests, so that value carried no information - it was set
# whether or not the clinician had considered the test at all.
#
# Three states are needed, because the clinical meanings are different
# and the system must respond to them differently:
#
#   UNSET       nothing stored. The clinician never reached this item.
#               Surfaced in the summary as outstanding.
#
#   NOT_INDICATED  a positive clinical judgement that the test does not
#               apply to this presentation. Closed. Never resurfaced,
#               and never sent to the agent as a gap - re-prompting a
#               clinician about a decision they already made correctly
#               is how a decision aid trains people to ignore it.
#
#   DEFERRED    the test is indicated but was not performed. Carries a
#               reason. Surfaced as pending, with the clinical condition
#               under which it should be revisited.
#
# The reason vocabulary is deliberately an enum rather than free text so
# that deferral patterns are countable across evaluation participants
# without transcript coding. The optional note captures nuance alongside
# it; it does not replace it.
#
# NOTE ON A REJECTED OPTION: an earlier draft of this enum included a
# patient-habitus reason. It was removed. A deferral reason is written
# into a clinical record that a head of department can read without
# having been present, so it must describe what obstructed the test, not
# an attribute of the person. DEFER_POSITIONING covers the functional
# limitation - leverage, stance, guarding, habitus - in the terms a
# handover note would use.

# The three states a recordable item can be in. Derived at read time
# from the stored value plus the deferrals dict; never stored as a
# field of its own.
ITEM_UNSET = "unset"
ITEM_NOT_INDICATED = "not_indicated"
ITEM_DEFERRED = "deferred"
ITEM_RECORDED = "recorded"

ITEM_STATES = (
    ITEM_UNSET,
    ITEM_NOT_INDICATED,
    ITEM_DEFERRED,
    ITEM_RECORDED,
)

ITEM_STATE_LABELS = {
    ITEM_UNSET: "Not yet addressed",
    ITEM_NOT_INDICATED: "Not indicated",
    ITEM_DEFERRED: "Deferred",
    ITEM_RECORDED: "Recorded",
}

# Visual class for the summary. Deliberately distinct from the navy
# System Check cards and the violet Suggested cards - an outstanding
# item is neither a deterministic finding nor a model suggestion.
ITEM_STATE_VISUAL_CLASS = {
    ITEM_UNSET: "item-outstanding",     # amber outline
    ITEM_DEFERRED: "item-pending",      # slate outline
    ITEM_NOT_INDICATED: "item-closed",  # muted, usually not rendered
    ITEM_RECORDED: "item-recorded",
}

# ---------------------------------------------------------------------------
# Deferral reasons
# ---------------------------------------------------------------------------
# Sourced from the scoping interviews. Four options, all of which
# describe an obstacle to performing the test rather than a property of
# the patient.
#
# Each reason carries the clinical condition under which the item should
# be revisited. That string lives here rather than in the template for
# the same reason the department and grade labels do: display strings
# stay out of Jinja, and the summary and the agent formatter then read
# one source rather than two that can drift.
#
# DEFER_DECLINED is deliberately absent. Patient refusal is the
# patient's decision, not a pending clinical task, and resurfacing it as
# an outstanding item would be inappropriate. Should it need recording,
# it belongs in the free-text note, not in a retry queue.

DEFER_PAIN = "pain_limited"
DEFER_EFFUSION = "effusion_limited"
DEFER_POSITIONING = "positioning_limited"
DEFER_TIME = "time_constraint"
DEFER_DECLINED = "patient_declined"

DEFERRAL_REASONS = (
    DEFER_PAIN,
    DEFER_EFFUSION,
    DEFER_POSITIONING,
    DEFER_TIME,
    DEFER_DECLINED,
)

DEFERRAL_REASON_LABELS = {
    DEFER_PAIN: "Pain limiting",
    DEFER_EFFUSION: "Effusion / swelling limiting",
    DEFER_POSITIONING: "Unable to position or achieve leverage",
    DEFER_TIME: "Insufficient time this session",
    DEFER_DECLINED: "Patient declined",
}

# The condition for revisiting. Phrased as a clinical state, not a
# deadline: the system reports what remains outstanding and under what
# circumstances it should be reassessed. It does not assert that an item
# is late.
#
# Asserting lateness would require a defensible standard for timely
# reassessment of an acute knee, which neither this system nor the
# source guidelines provide. It would also create pressure to perform a
# test in order to clear a flag, which is the automation-bias failure
# mode this project is measuring from the opposite direction.
DEFERRAL_RETRY_CONDITION = {
    DEFER_PAIN: "Reassess when pain allows.",
    DEFER_EFFUSION: "Reassess when the effusion settles.",
    DEFER_POSITIONING: "Reassess when positioning permits.",
    DEFER_TIME: "Carry forward to the next session.",
    DEFER_DECLINED: "",
}

# Cap on the optional note. Long enough for a clinical sentence, short
# enough that it does not become a substitute for the enum.
DEFERRAL_NOTE_MAX_CHARS = 300


# ---------------------------------------------------------------------------
# Special test result vocabulary
# ---------------------------------------------------------------------------
# Previously implicit in the template's chip values. Named here so the
# router, the agent formatter and the coverage module agree on what
# counts as a recorded result.
#
# 'not_done' is retained ONLY to classify documents written before the
# three-state model existed. Nothing writes it now. Legacy values cannot
# be distinguished retrospectively between "considered and declined" and
# "never reached", so coverage treats them as UNSET - the conservative
# reading, since it surfaces the item rather than silently closing it.

TEST_POSITIVE = "positive"
TEST_NEGATIVE = "negative"
TEST_LEGACY_NOT_DONE = "not_done"

TEST_RECORDED_VALUES = (TEST_POSITIVE, TEST_NEGATIVE)

TEST_RESULT_LABELS = {
    TEST_POSITIVE: "Positive",
    TEST_NEGATIVE: "Negative",
}


# ---------------------------------------------------------------------------
# Validation and display helpers
# ---------------------------------------------------------------------------

def is_valid_deferral_reason(value: str) -> bool:
    return value in DEFERRAL_REASONS


def deferral_reason_label(value: str) -> str:
    """Unknown keys render readably rather than blank."""
    return DEFERRAL_REASON_LABELS.get(value, "Reason not recorded")


def deferral_retry_condition(value: str) -> str:
    """Empty string for unknown reasons, so templates render nothing."""
    return DEFERRAL_RETRY_CONDITION.get(value, "")


def item_state_label(value: str) -> str:
    return ITEM_STATE_LABELS.get(value, ITEM_STATE_LABELS[ITEM_UNSET])


def item_state_visual_class(value: str) -> str:
    return ITEM_STATE_VISUAL_CLASS.get(value, "item-outstanding")
