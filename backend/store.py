from datetime import date
from backend.firebase_init import db

_collection = db.collection("cases")


def create_case(patient_label: str):
    doc_ref = _collection.document()
    case = {
        "id": doc_ref.id,
        "patient_label": patient_label,
        "created_at": date.today().isoformat(),
    }
    doc_ref.set(case)
    return case


def list_cases():
    return [doc.to_dict() for doc in _collection.stream()]


def get_case(case_id: str):
    doc = _collection.document(case_id).get()
    return doc.to_dict() if doc.exists else None
