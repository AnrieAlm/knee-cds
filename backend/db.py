"""
backend/db.py

Single shared MongoDB handle.

auth.py needs database access to resolve a Firebase uid into a clinical
profile (role, department, grade). Importing that handle from main.py
would be circular, since main.py imports auth. So the connection lives
here and both import from it.

main.py can keep its existing connection for now - two clients is
wasteful but harmless. When you get a quiet moment, point main.py at
get_db() too and delete its own client.
"""

import os
from pymongo import MongoClient

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DB_NAME = os.getenv("MONGO_DB_NAME", "knee_cds")

_client = None


def get_db():
    """Lazy singleton. MongoClient is thread-safe and pools internally."""
    global _client
    if _client is None:
        uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI")
        if not uri:
            raise RuntimeError("MONGO_URI is not set")
        _client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    return _client[DB_NAME]