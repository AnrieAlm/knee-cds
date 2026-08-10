#!/usr/bin/env python3
"""
migrate_admin_schema.py  (v2)

Rewritten after inspecting the live database, which showed:
  - no users collection at all (users existed only in Firebase)
  - no owner field on cases (so every case was visible to everyone)
  - created_at stored as a date STRING with no time component

This script therefore does more than a backfill - it introduces ownership
for the first time.

  1. Creates the users collection and seeds your account.
  2. Stamps all existing cases as owned by you, flagged legacy: true.
  3. Converts created_at from "YYYY-MM-DD" strings to real datetimes.
  4. Creates the indexes the admin queries need.
  5. Optionally seeds a SEPARATE head-of-department admin account.

DRY RUN BY DEFAULT. Nothing is written unless you pass --apply.

Typical run:

    # 1. look before you leap
    python migrate_admin_schema.py --inspect

    # 2. dry run - get your uid from Firebase console > Authentication
    python migrate_admin_schema.py \
        --owner-uid IoWwlpOoX8Q9vfLNu97AO5pW3x32 \
        --owner-email lua.faimari@gmail.com \
        --owner-name "Anriel Almeida" \
        --owner-department musculoskeletal \
        --owner-grade staff_grade

    # 3. same command again with --apply

    # 4. later, once the head physio account exists in Firebase
    python migrate_admin_schema.py --apply \
        --admin-uid DeF456...uvw \
        --admin-email head.physio@example.com \
        --admin-name "Head of Department"

Run from the repo root. Run once.
"""

import argparse
import os
import sys
from collections import Counter
from datetime import datetime, timezone
try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
except ImportError:
    sys.exit("pymongo not found. Activate your venv first.")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # env vars may already be set another way

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
try:
    from constants import (
        ROLE_PHYSIO,
        ROLE_ADMIN,
        DEPT_UNASSIGNED,
        GRADE_UNKNOWN,
        is_valid_department,
        is_valid_grade,
    )
except ImportError:
    sys.exit(
        "Could not import backend/constants.py.\n"
        "Save constants.py to backend/constants.py first, then re-run."
    )


DB_NAME = os.getenv("MONGO_DB_NAME", "knee_cds")
USERS_COLLECTION = "users"
CASES_COLLECTION = "cases"

UID_FIELDS = ["physio_uid", "user_uid", "uid", "created_by_uid", "firebase_uid"]
EMAIL_FIELDS = ["physio_email", "user_email", "email", "created_by"]


def connect():
    uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI")
    if not uri:
        sys.exit("Set MONGO_URI in your environment (or .env) and re-run.")
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    return client[DB_NAME]


def detect_link_field(cases):
    sample = list(cases.find({}, limit=200))
    if not sample:
        return None, None
    present = Counter()
    for doc in sample:
        for key in doc:
            present[key] += 1
    for field in UID_FIELDS:
        if present.get(field, 0):
            return field, "uid"
    for field in EMAIL_FIELDS:
        if present.get(field, 0):
            return field, "email"
    return None, None


def inspect(db):
    users = db[USERS_COLLECTION]
    cases = db[CASES_COLLECTION]

    print(f"\ndatabase: {DB_NAME}")
    print(f"collections present: {sorted(db.list_collection_names())}")
    print(f"\nusers: {users.count_documents({})}")
    print(f"cases: {cases.count_documents({})}")

    doc = cases.find_one()
    if not doc:
        print("\nNo case documents.")
        return

    print("\ntop-level fields on a sample case document:")
    for key in sorted(doc.keys()):
        value = doc[key]
        kind = type(value).__name__
        preview = "" if kind in ("dict", "list") else f"  = {str(value)[:60]}"
        print(f"    {key:<28} {kind}{preview}")

    field, kind = detect_link_field(cases)
    print()
    if field:
        print(f"--> cases already link to users via '{field}' ({kind})")
    else:
        print("--> cases have NO owner field. This script will add one.")

    unowned = cases.count_documents({"physio_uid": {"$exists": False}})
    str_dates = sum(
        1 for c in cases.find({}, {"created_at": 1})
        if isinstance(c.get("created_at"), str)
    )
    print(f"    cases without physio_uid: {unowned}")
    print(f"    cases with string created_at: {str_dates}")


def parse_created_at(value):
    """'2026-07-27' -> datetime. Returns None if already fine or unparseable."""
    if isinstance(value, datetime):
        return None
    if not isinstance(value, str):
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip()[:len("2026-07-27T00:00:00")], fmt)
        except ValueError:
            continue
    return None


def seed_user(db, apply, uid, email, name, department, grade, role):
    """Upsert a user document. Idempotent - safe to re-run."""
    users = db[USERS_COLLECTION]

    if not is_valid_department(department):
        sys.exit(f"'{department}' is not in DEPARTMENTS. See backend/constants.py")
    if not is_valid_grade(grade):
        sys.exit(f"'{grade}' is not in GRADES. See backend/constants.py")

    existing = users.find_one({"uid": uid})
    doc = {
        "uid": uid,
        "email": email,
        "full_name": name,
        "role": role,
        "department": department,
        "grade": grade,
        "active": True,
    }

    label = f"{name} <{email}> as {role}/{department}/{grade}"
    if not apply:
        verb = "would update" if existing else "would create"
        print(f"  [dry run] {verb} user {label}")
        return

    if existing:
        users.update_one({"uid": uid}, {"$set": doc})
        print(f"  updated user {label}")
    else:
        doc["created_at"] = datetime.now(timezone.utc)
        users.insert_one(doc)
        print(f"  created user {label}")


def claim_legacy_cases(db, apply, uid, name, department, grade):
    """Assign every ownerless case to one account, flagged legacy."""
    cases = db[CASES_COLLECTION]
    unowned = list(cases.find({"physio_uid": {"$exists": False}}))

    print(f"\nownerless cases: {len(unowned)}")
    if not unowned:
        return

    for case in unowned:
        patch = {
            "physio_uid": uid,
            "physio_name": name,
            "department": department,
            "grade": grade,
            "legacy": True,
        }
        converted = parse_created_at(case.get("created_at"))
        if converted:
            patch["created_at"] = converted
        if apply:
            cases.update_one({"_id": case["_id"]}, {"$set": patch})

    verb = "claimed" if apply else "[dry run] would claim"
    print(f"  {verb} {len(unowned)} cases for {name}, flagged legacy: true")
    print("  Filter these out of the demo with {'legacy': {'$ne': True}}")


def fix_remaining_dates(db, apply):
    """Catch any owned cases whose created_at is still a string."""
    cases = db[CASES_COLLECTION]
    fixed = 0
    for case in cases.find({}, {"created_at": 1}):
        converted = parse_created_at(case.get("created_at"))
        if converted:
            if apply:
                cases.update_one(
                    {"_id": case["_id"]}, {"$set": {"created_at": converted}}
                )
            fixed += 1
    if fixed:
        verb = "converted" if apply else "[dry run] would convert"
        print(f"\n{verb} {fixed} created_at values from string to datetime")
        print("  Write datetime.utcnow() on new cases so this stays fixed.")


def create_indexes(db, apply):
    print("\nindexes:")
    specs = [
        (CASES_COLLECTION, [("physio_uid", ASCENDING)]),
        (CASES_COLLECTION, [("department", ASCENDING), ("created_at", DESCENDING)]),
        (CASES_COLLECTION, [("grade", ASCENDING), ("created_at", DESCENDING)]),
        (USERS_COLLECTION, [("uid", ASCENDING)]),
        (USERS_COLLECTION, [("role", ASCENDING)]),
        (USERS_COLLECTION, [("department", ASCENDING)]),
    ]
    for coll, keys in specs:
        desc = ", ".join(k for k, _ in keys)
        if apply:
            db[coll].create_index(keys)
            print(f"  created {coll}({desc})")
        else:
            print(f"  [dry run] would create {coll}({desc})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="actually write")
    p.add_argument("--inspect", action="store_true", help="report and exit")

    p.add_argument("--owner-uid", help="your Firebase uid - claims legacy cases")
    p.add_argument("--owner-email")
    p.add_argument("--owner-name")
    p.add_argument("--owner-department", default="musculoskeletal")
    p.add_argument("--owner-grade", default="staff_grade")

    p.add_argument("--admin-uid", help="head of department Firebase uid")
    p.add_argument("--admin-email")
    p.add_argument("--admin-name", default="Head of Department")
    p.add_argument("--admin-department", default=DEPT_UNASSIGNED)

    args = p.parse_args()

    db = connect()
    inspect(db)
    if args.inspect:
        return

    if not args.apply:
        print("\n" + "=" * 60)
        print("DRY RUN - nothing written. Re-run with --apply when happy.")
        print("=" * 60)

    if args.owner_uid:
        if not (args.owner_email and args.owner_name):
            sys.exit("--owner-uid needs --owner-email and --owner-name too")
        print("\nowner account:")
        seed_user(
            db, args.apply,
            uid=args.owner_uid,
            email=args.owner_email,
            name=args.owner_name,
            department=args.owner_department,
            grade=args.owner_grade,
            role=ROLE_PHYSIO,
        )
        claim_legacy_cases(
            db, args.apply,
            uid=args.owner_uid,
            name=args.owner_name,
            department=args.owner_department,
            grade=args.owner_grade,
        )

    if args.admin_uid:
        if not args.admin_email:
            sys.exit("--admin-uid needs --admin-email too")
        print("\nadmin account:")
        seed_user(
            db, args.apply,
            uid=args.admin_uid,
            email=args.admin_email,
            name=args.admin_name,
            department=args.admin_department,
            grade=GRADE_UNKNOWN,
            role=ROLE_ADMIN,
        )

    fix_remaining_dates(db, args.apply)
    create_indexes(db, args.apply)

    if not (args.owner_uid or args.admin_uid):
        print("\nNo accounts given - nothing seeded.")
        print("Pass --owner-uid / --owner-email / --owner-name to claim cases.")

    print("\ndone.")


if __name__ == "__main__":
    main()