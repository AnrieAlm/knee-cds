import os, sys, time, statistics

backend = os.getenv("LLM_BACKEND", "groq")
from backend import store
from backend.db import get_db
from backend.main import _build_safety_facts, _build_agent_query
from backend.agent.orchestrator import run_agent_only

db = get_db()

# Optional: pass a specific case id as the first argument to pin the
# benchmark to a known case instead of depending on Mongo's insertion
# order. This matters because "first clean case" can silently pick up
# deliberately adversarial test data (e.g. physiologically inconsistent
# ROM values built to test the data-quality guard) and cause the agent
# to loop without converging, which looks like a hang or a crash rather
# than a benchmark result.
requested_id = sys.argv[1] if len(sys.argv) > 1 else None

case = None
if requested_id:
    from bson import ObjectId
    case = db.cases.find_one({"_id": ObjectId(requested_id)})
    if not case:
        print(f"No case found with id {requested_id}.")
        sys.exit(1)
else:
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
    if i < 2:
        time.sleep(20) 
    d = time.time() - start
    times.append(d)
    print(f"  run {i+1}: {d:.2f}s")

print(f"mean: {statistics.mean(times):.2f}s")