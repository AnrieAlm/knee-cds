# test_orchestrator_manual.py
#
# A manual smoke test for the Cygnus orchestrator. This is NOT a pytest
# file - it's a script you run by hand to watch the whole flow end to end:
#   red-flag gate -> Ottawa/Pittsburgh facts -> agent retrieval + reasoning.
#
# Run from the project root:
#   cd /media/anriel/LENOVO_USB_HDD/knee_cds
#   source venv/bin/activate
#   python -m backend.agent.test_orchestrator_manual
#
# Requires Ollama running with llama3.1:8b and nomic-embed-text pulled,
# and the ChromaDB collection already ingested.

from backend.rules.ottawa import OttawaInput
from backend.rules.pittsburgh import PittsburghInput
from backend.safety.red_flags import RedFlagInput
from backend.agent.orchestrator import run_assessment


def print_result(label, result):
    print("=" * 65)
    print("SCENARIO:", label)
    print("=" * 65)

    print("\n[ SYSTEM CHECK - deterministic ]")
    if result.stopped_at_red_flag:
        print("  RED FLAG POSITIVE - assessment halted, LLM did not run.")
        print("  " + result.red_flag_message.replace("\n", "\n  "))
        print()
        return

    print("  Red flags:", result.red_flag_message)
    if result.ottawa_result is not None:
        print("  Ottawa:", result.ottawa_result.rationale)
    if result.pittsburgh_result is not None:
        print("  Pittsburgh:", result.pittsburgh_result.rationale)

    print("\n[ SUGGESTED - review before use (LLM) ]")
    print("  " + result.agent_suggestion.replace("\n", "\n  "))
    print()


# -----------------------------------------------------------
# Scenario 1: a red flag fires -> must STOP, LLM must not run
# -----------------------------------------------------------
# Hot swollen joint + fever = possible septic arthritis. The orchestrator
# should halt at the gate and never reach the agent.

red_flag_positive_case = RedFlagInput(
    hot_swollen_joint=True,
    fever_with_joint_pain=True,
)

# The Ottawa/Pittsburgh inputs are still required by the function signature,
# but they should never be reached in this scenario.
ottawa_dummy = OttawaInput(
    age=30,
    isolated_patella_tenderness=False,
    fibula_head_tenderness=False,
    unable_to_flex_90=False,
    unable_to_weight_bear=False,
)
pittsburgh_dummy = PittsburghInput(
    mechanism_blunt_trauma_or_fall=False,
    age=30,
    unable_to_weight_bear=False,
)

result1 = run_assessment(
    red_flag_input=red_flag_positive_case,
    ottawa_input=ottawa_dummy,
    pittsburgh_input=pittsburgh_dummy,
    clinician_query="Which special tests should I prioritise?",
)
print_result("Red flag positive (septic arthritis pattern)", result1)


# -----------------------------------------------------------
# Scenario 2: clean case -> full flow runs
# -----------------------------------------------------------
# No red flags. A 28-year-old, non-contact pivoting injury with a reported
# pop and giving-way. Ottawa: unable to flex 90 -> x-ray indicated. The
# agent should then suggest ACL-relevant tests (Lachman, pivot shift,
# anterior drawer) grounded in retrieved evidence.

red_flag_clear = RedFlagInput()  # all False by default

ottawa_case = OttawaInput(
    age=28,
    isolated_patella_tenderness=False,
    fibula_head_tenderness=False,
    unable_to_flex_90=True,        # triggers x-ray
    unable_to_weight_bear=False,
)
pittsburgh_case = PittsburghInput(
    mechanism_blunt_trauma_or_fall=False,  # non-contact -> Pittsburgh won't apply
    age=28,
    unable_to_weight_bear=False,
)

result2 = run_assessment(
    red_flag_input=red_flag_clear,
    ottawa_input=ottawa_case,
    pittsburgh_input=pittsburgh_case,
    clinician_query=(
        "28-year-old, non-contact pivoting injury, felt a pop, knee gives way. "
        "Which special tests should I prioritise and why?"
    ),
)
print_result("Clean case - suspected ACL", result2)
