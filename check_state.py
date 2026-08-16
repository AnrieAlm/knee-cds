from bson import ObjectId
from backend.db import get_db

c = get_db()["cases"].find_one({"_id": ObjectId("6a79b3ecf7b99ba2f2b6b15d")})
invs = c.get("investigations") or []
print("investigations:", len(invs))
for i in invs:
    print("  ", i["extraction_status"], i["modality"], i["side"])
print("physical keys filled:",
      len([k for k, v in (c.get("physical") or {}).items() if v not in (None, "", [], {})]))
