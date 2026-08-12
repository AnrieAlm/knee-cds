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
