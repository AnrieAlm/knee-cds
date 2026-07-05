from itertools import count
from datetime import date

_id_counter = count(1)
_cases = {}


def create_case(patient_label: str):
    case_id = next(_id_counter)
    case = {
        "id": case_id,
        "patient_label": patient_label,
        "created_at": date.today().isoformat(),
    }
    _cases[case_id] = case
    return case


def list_cases():
    return list(_cases.values())


def get_case(case_id: int):
    return _cases.get(case_id)