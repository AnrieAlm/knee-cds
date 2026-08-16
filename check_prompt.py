from bson import ObjectId
from backend.db import get_db
from backend.investigation_context import build_investigation_context_from_list

c = get_db()["cases"].find_one({"_id": ObjectId("6a79b3ecf7b99ba2f2b6b15d")})
print(build_investigation_context_from_list(
    c.get("investigations"), c.get("created_at")))
