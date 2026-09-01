import os, time, statistics, sys

backend = os.getenv("LLM_BACKEND", "groq")
from backend import store
from backend.db import get_db
from backend.main import _build_safety_facts, _build_agent_query
from backend.agent.orchestrator import run_agent_only

db = get_db()
case = None
for c in db.cases.find():
    a = c.get("assessment")
    if a and not a.get("red_flag_positive"):
        case = c
        break

if not case:
    print("No case with a clean assessment found.")
    sys.exit(1)

print(f"backend={backend}  case={case['_id']}  label={case.get('patient_label')}")

safety_facts = _build_safety_facts(case["assessment"], case.get("history"))
query = _build_agent_query(case)

times = []
for i in range(3):
    start = time.time()
    run_agent_only(
        query, safety_facts, case.get("physical"), case.get("investigations"),
        deferrals=case.get("deferrals"),
        involved_side=(case.get("history") or {}).get("involved_side"),
    )
    d = time.time() - start
    times.append(d)
    print(f"  run {i+1}: {d:.2f}s")

print(f"mean: {statistics.mean(times):.2f}s")
