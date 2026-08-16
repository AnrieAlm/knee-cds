"""
backend/case_access.py

Ownership and role gate for case documents.
--------------------------------------------------------------------
Every route that touches a case must pass through here. Before this
module existed, main.py checked only that a valid Firebase token was
present - it never compared the case's physio_uid against the caller.
Any authenticated physiotherapist could read or overwrite any case in
any department by knowing its id.

This module exists rather than living in auth.py because judging access
needs both the identity (auth) and the document (store). auth.py is
imported by store-free modules and must stay free of that dependency,
so the join happens here instead.


DEVIATIONS   (Examples 07-10, main.py)

D1  Scaffold puts everything in one main.py. Cygnus splits into a
    backend/ package with an APIRouter per domain. main.py is 730
    lines. https://fastapi.tiangolo.com/tutorial/bigger-applications/#include-an-apirouter-with-a-custom-prefix-tags-responses-and-dependencies
    A single file stops being readable and the deterministic
    safety layer needs to be legible as a unit for the dissertation.

D2  Scaffold repeats validateFirebaseToken() inside every file that
    needs it. Cygnus has one backend/auth.py. Duplicated validation
    drifts. The physical examination router originally shipped with
    no token check at all, which is exactly that failure mode.

D6  Scaffold enforces ownership by scoping the query itself
        user_collection.find_one({'user_id': user_token['user_id']})
    (Example07, addition 3). Cygnus keeps that principle - the caller's
    uid is never trusted as an authorisation claim - but loads the case
    first and then judges it, because a head-of-department account is
    permitted to READ a case it does not own. A single scoped query
    cannot express "owner, or admin", so the check is two-stage.

D7  Scaffold returns RedirectResponse('/') for any auth failure
    (Example09, addition 6). Cygnus raises HTTPException with a real
    status code. A supervisor blocked from writing (403) and a signed
    out user (401) are different events and the audit trail has to be
    able to tell them apart. main.py registers a handler that turns 401
    back into the scaffold's redirect for page routes, so the user
    experience is unchanged.

"""

from fastapi import Request, HTTPException

from backend import store
from backend.auth import (
    get_current_user,
    can_view_case,
    owns_case,
)
from backend.constants import ROLE_ADMIN


# --------------------------------------------------------------------
# addition 1
# Resolve the caller. Raising rather than returning None so that no
# route can accidentally continue past a failed check by forgetting an
# `if not user` line - which is how the original main.py routes ended
# up enforcing nothing beyond token presence.
# --------------------------------------------------------------------

def _require_user(request: Request) -> dict:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# --------------------------------------------------------------------
# addition 2
# Read access: the treating physiotherapist, or any admin.
#
# A physiotherapist requesting someone else's case gets 404, not 403.
# 403 would confirm that a case with that id exists, which turns the
# URL into an enumeration oracle over patient records. The distinction
# costs nothing and the scaffold has no equivalent because its data is
# never shared.
# --------------------------------------------------------------------

def load_case_for_read(request: Request, case_id: str):
    """
    Return (user, case) for a case this caller may view.

    Raises 401 if not signed in, 404 if the case does not exist or does
    not belong to this caller.
    """
    user = _require_user(request)

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if not can_view_case(user, case):
        raise HTTPException(status_code=404, detail="Case not found")

    return user, case


# --------------------------------------------------------------------
# addition 3
# Write access: the treating physiotherapist only.
#
# Admin is rejected with 403 rather than 404 deliberately changed. The admin
# CAN see this case - hiding it now would be confusing - they simply
# may not modify it. Head-of-department accounts are read-only so that
# the record continues to show who actually performed the assessment.
#
# auth.require_physio() already encodes the role half of this rule, but
# it does not know about the case, so it cannot stop physiotherapist A
# writing to physiotherapist B's record. Both halves are needed.
# --------------------------------------------------------------------

def load_case_for_write(request: Request, case_id: str):
    """
    Return (user, case) for a case this caller may modify.

    Raises 401 if not signed in, 403 if the caller is an admin, and 404
    if the case does not exist or belongs to another physiotherapist.
    """
    user = _require_user(request)

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if user.get("role") == ROLE_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Head of department accounts have read-only access",
        )

    if not owns_case(user, case):
        raise HTTPException(status_code=404, detail="Case not found")

    return user, case


# --------------------------------------------------------------------
# addition 4
# Convenience for routes that answer with JSON rather than a page
# (the chat endpoint and the suggestion poller). Same rules, but the
# caller wants to shape its own error body instead of raising.
# --------------------------------------------------------------------

def try_load_case_for_read(request: Request, case_id: str):
    """Non-raising variant. Returns (user, case) or (None, None)."""
    try:
        return load_case_for_read(request, case_id)
    except HTTPException:
        return None, None


def try_load_case_for_write(request: Request, case_id: str):
    """Non-raising variant. Returns (user, case) or (None, None)."""
    try:
        return load_case_for_write(request, case_id)
    except HTTPException:
        return None, None