# store.py
# MongoDB persistence layer for Cygnus.
# Follows the pymongo patterns from the PaaS scaffold (MongoClient,
# insert_one, find, find_one, find_one_and_update).
#
# The connection string is read from the MONGO_URI environment variable
# (loaded from a git-ignored .env file) so no credential is ever hardcoded
# or committed to version control.

import os
from datetime import date, datetime, timezone

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


# --- connection ---
# Read the URI from the environment. Never hardcode the password here.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError(
        "MONGO_URI environment variable not set. "
        "Add it to your .env file (which must be git-ignored)."
    )

# Create a client and connect to the server (booklet pattern).
client = MongoClient(MONGO_URI, server_api=ServerApi("1"))

# Send a ping to confirm the connection works.
try:
    client.admin.command("ping")
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Open the database and the cases collection.
db = client["knee_cds"]
_collection = db["cases"]


# -----------------------------------------------------------
# Case CRUD
# -----------------------------------------------------------
def create_case(patient_label: str, user: dict):
    """
    Create a new case document and return it.

    Ownership is stamped at creation from the logged-in profile. The
    department and grade are denormalised onto the case deliberately:
    if a physiotherapist later transfers department, this record stays
    attributed to where the assessment actually happened.
    """
    case = {
        "patient_label": patient_label,
        "created_at": datetime.now(timezone.utc),
        "physio_uid": user["uid"],
        "physio_name": user.get("full_name") or user.get("email", ""),
        "department": user.get("department", "unassigned"),
        "grade": user.get("grade", "unknown"),
        # agentLog starts empty; append-only via append_agent_log below.
        "agentLog": [],
    }
    result = _collection.insert_one(case)
    # store the generated _id as a string 'id' field for easy URL use
    case_id = str(result.inserted_id)
    _collection.update_one(
        {"_id": result.inserted_id},
        {"$set": {"id": case_id}},
    )
    case["id"] = case_id
    return case


def list_cases(physio_uid: str):
    """
    Return cases belonging to one physiotherapist.

    physio_uid is a REQUIRED positional argument. A default of None
    returning everything would mean any caller that forgot to pass it
    silently leaked the whole collection - this way a mistake is an
    immediate TypeError instead.
    """
    cursor = _collection.find({"physio_uid": physio_uid}).sort("created_at", -1)
    return [_clean(doc) for doc in cursor]


def list_all_cases(department=None, grade=None, include_legacy=True):
    """Head-of-department view. Deliberately separate from list_cases()."""
    query = {}
    if department:
        query["department"] = department
    if grade:
        query["grade"] = grade
    if not include_legacy:
        query["legacy"] = {"$ne": True}
    cursor = _collection.find(query).sort("created_at", -1)
    return [_clean(doc) for doc in cursor]

# -----------------------------------------------------------
# Audit trail (append-only)
# -----------------------------------------------------------

def append_agent_log(case_id: str, entry: dict):
    """
    Append one entry to the case's agentLog array. Append-only: uses
    $push, which adds to the array without touching existing entries.
    This is the tamper-evident audit trail — it records what the agent
    retrieved and which deterministic rules fired.
    """
    entry["logged_at"] = datetime.now().isoformat()
    _collection.find_one_and_update(
        {"id": case_id},
        {"$push": {"agentLog": entry}},
    )
    return entry


def get_agent_log(case_id: str):
    """Return the agentLog array for a case (oldest first)."""
    doc = _collection.find_one({"id": case_id})
    if not doc:
        return []
    return doc.get("agentLog", [])


# -----------------------------------------------------------
# Helper
# -----------------------------------------------------------

def _clean(doc):
    """
    Remove MongoDB's internal _id (an ObjectId, not JSON/template friendly)
    before handing the document to templates. Our own 'id' field remains.
    """
    if doc and "_id" in doc:
        doc = dict(doc)
        del doc["_id"]
    return doc

def save_assessment(case_id: str, assessment: dict):
    """
    Save the deterministic assessment result onto the case document.
    Overwrites any previous assessment for this case (re-running the exam
    replaces the result). This is separate from agentLog, which is append-only.
    """
    _collection.find_one_and_update(
        {"id": case_id},
        {"$set": {"assessment": assessment}},
    )
    return assessment


def save_history(case_id: str, history: dict):
    # save the history form data onto the case document
    # same pattern as save_assessment — overwrites on re-submission
    _collection.find_one_and_update(
        {'id': case_id},
        {'$set': {'history': history}},
    )
    return history

def save_physical(case_id: str, physical: dict):
    # save physical examination data onto the case document
    # same pattern as save_history — overwrites on re-submission
    _collection.find_one_and_update(
        {'id': case_id},
        {'$set': {'physical': physical}},
    )
    return physical

def get_case(case_id: str):
    """Return one case by its string id, or None."""
    doc = _collection.find_one({"id": case_id})
    return _clean(doc) if doc else None

def save_deferrals(case_id: str, deferrals: dict):
    """
    Save the deferral map onto the case document.

    Same overwrite pattern as save_physical. The map is rebuilt in full
    on each save rather than patched, so an item that stops being
    deferred disappears from it.
    """
    _collection.find_one_and_update(
        {'id': case_id},
        {'$set': {'deferrals': deferrals}},
    )
    return deferrals