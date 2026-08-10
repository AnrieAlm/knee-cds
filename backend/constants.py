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