"""
Shared Firebase authentication helpers.
-----------------------------------------------------------
Lives in its own module so that both main.py and the physical
examination router use the same token validation. Previously the
router had no auth at all, which left physical examination data
readable and writable by anyone holding a case id.

Firebase remains the authentication authority: it decides who someone
is. MongoDB holds the clinical metadata Firebase has no concept of -
which department a physiotherapist works in, what grade they are, and
whether they are the head of department. The two are joined on uid.

Everything above the "role-aware additions" divider is unchanged from
the original module. Existing routes calling get_user() or
require_user() keep the exact same return value.
"""

from fastapi import Request, HTTPException
import google.oauth2.id_token
from google.auth.transport import requests as google_requests

from datetime import datetime, timezone

from backend.db import get_db
from backend.constants import (
    ROLE_PHYSIO,
    ROLE_ADMIN,
    DEPT_UNASSIGNED,
    GRADE_UNKNOWN,
)

firebase_request_adapter = google_requests.Request()


def validate_firebase_token(id_token):
    """Return the decoded token, or None if absent/invalid."""
    if not id_token:
        return None
    try:
        return google.oauth2.id_token.verify_firebase_token(
            id_token, firebase_request_adapter
        )
    except ValueError as err:
        print(f"[auth] token rejected: {err}")
        return None


def get_user(request: Request):
    """Non-raising lookup - use in page routes that redirect to /login."""
    return validate_firebase_token(request.cookies.get("token"))


def require_user(request: Request):
    """
    Raising dependency - use with Depends() on routers.
    Returns the decoded token so routes can read the uid if needed.
    """
    user_token = validate_firebase_token(request.cookies.get("token"))
    if not user_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user_token


# ---------------------------------------------------------------------------
# Role-aware additions
# ---------------------------------------------------------------------------

def extract_uid(token: dict) -> str | None:
    """
    Pull the Firebase uid out of a decoded token.

    verify_firebase_token returns raw JWT claims, so the uid arrives as
    'user_id' or 'sub' - NOT as 'uid'. The friendly 'uid' key only exists
    in the Firebase Admin SDK. Checking all three means this keeps working
    if the verification method is ever swapped.
    """
    if not token:
        return None
    return token.get("user_id") or token.get("sub") or token.get("uid")


def _default_profile(uid: str, token: dict) -> dict:
    return {
        "uid": uid,
        "email": token.get("email", ""),
        "full_name": token.get("name") or token.get("email", ""),
        "role": ROLE_PHYSIO,
        "department": DEPT_UNASSIGNED,
        "grade": GRADE_UNKNOWN,
        "active": True,
        "created_at": datetime.now(timezone.utc),
    }


def sync_user(token: dict) -> dict | None:
    """
    Resolve a decoded token into a Mongo user profile, creating one on
    first sight.

    IMPORTANT: role, department and grade are only ever written on
    INSERT. If this method overwrote them on every login, promoting the
    head of department to admin would silently revert the next time they
    signed in - a bug that would look like the admin page randomly
    breaking. Only identity fields Firebase owns get refreshed.
    """
    uid = extract_uid(token)
    if not uid:
        return None

    users = get_db()["users"]
    profile = users.find_one({"uid": uid})

    if profile is None:
        profile = _default_profile(uid, token)
        users.insert_one(dict(profile))
        print(f"[auth] created profile for {profile['email']} ({uid})")
        return profile

    # Refresh only what Firebase is authoritative for.
    refresh = {"last_seen": datetime.now(timezone.utc)}
    if token.get("email") and token["email"] != profile.get("email"):
        refresh["email"] = token["email"]
    if token.get("name") and not profile.get("full_name"):
        refresh["full_name"] = token["name"]

    users.update_one({"uid": uid}, {"$set": refresh})
    profile.update(refresh)
    return profile


def get_current_user(request: Request) -> dict | None:
    """
    Non-raising. Returns the decoded token merged with the Mongo profile,
    so templates can read user.role / user.department / user.full_name.
    Returns None when not signed in.
    """
    token = validate_firebase_token(request.cookies.get("token"))
    if not token:
        return None
    profile = sync_user(token)
    if not profile:
        return None
    merged = dict(token)
    merged.update(profile)
    merged.pop("_id", None)
    return merged


def require_current_user(request: Request) -> dict:
    """Raising version of get_current_user, for Depends()."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_admin(request: Request) -> dict:
    """
    Head-of-department only. 403 rather than 401: the caller is
    authenticated, just not permitted, and conflating the two makes
    debugging the admin page unnecessarily confusing.
    """
    user = require_current_user(request)
    if user.get("role") != ROLE_ADMIN:
        raise HTTPException(status_code=403, detail="Head of department only")
    return user


def require_physio(request: Request) -> dict:
    """
    Guards WRITE routes - creating cases, saving examination findings.

    Admin accounts are deliberately rejected here. The head of department
    has read-only oversight; if they could edit a record, the audit trail
    would no longer show who actually performed the assessment. Hiding
    the edit buttons in a template is presentation, not access control -
    this is the part that actually enforces it.
    """
    user = require_current_user(request)
    if user.get("role") == ROLE_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Head of department accounts have read-only access",
        )
    return user


def owns_case(user: dict, case: dict) -> bool:
    """True if this user performed the assessment."""
    if not case:
        return False
    return case.get("physio_uid") == extract_uid(user) or \
        case.get("physio_uid") == user.get("uid")


def can_view_case(user: dict, case: dict) -> bool:
    """Owner, or any admin."""
    if not case:
        return False
    return user.get("role") == ROLE_ADMIN or owns_case(user, case)