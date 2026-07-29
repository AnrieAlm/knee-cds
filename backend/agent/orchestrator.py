# orchestrator.py
#
# The Cygnus agent orchestrator. This is the spine of the system: it runs
# a single acute-knee assessment turn, combining the DETERMINISTIC safety
# layer with the LLM agent, and keeps them architecturally separate.
#
# Design (Option A - safety as a hard gate, NOT as agent tools):
#
#   1. RED-FLAG SCREEN  - deterministic. Runs FIRST, before the LLM sees
#                         anything. If a red flag fires, we STOP and return
#                         the deterministic warning. The agent never runs.
#
#   2. OTTAWA / PITTSBURGH - deterministic. Imaging decision rules are
#                         computed as FACTS. The LLM does not decide these;
#                         it only receives their results.
#
#   3. AGENT LOOP (ReAct) - the LLM. It reasons about which special tests
#                         to prioritise and retrieves supporting evidence
#                         from the corpus. Safety facts are injected into
#                         its context so it can reference them, but it can
#                         never override them.
#
#   4. RESPONSE ASSEMBLY - combines the deterministic "System Check" block
#                         with the LLM's "Suggested" block, kept visually
#                         and structurally separate so a clinician never
#                         mistakes an LLM suggestion for a verified rule.
#
# Why this structure matters (dissertation argument): the safety-critical
# outputs are produced by deterministic Python, outside the LLM's control.
# The LLM cannot forget to run a check, cannot run it wrong, and cannot
# talk itself out of a red flag. That is the core safety claim of Cygnus.

from dataclasses import dataclass, field

# --- Deterministic layer imports (no LLM) ---
from backend.rules.ottawa import apply_ottawa_knee_rule, OttawaInput
from backend.rules.pittsburgh import apply_pittsburgh_knee_rule, PittsburghInput
from backend.safety.red_flags import screen_red_flags, RedFlagInput

# --- Retrieval (used by the agent as a tool) ---
from backend.rag.retriever import retrieve

# --- LangChain v1 agent construction ---
# NOTE: create_agent is the LangChain v1 API (langchain 1.3.10). It replaces
# the older create_react_agent / initialize_agent seen in most tutorials.
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama


# ===========================================================
# Settings
# ===========================================================

LLM_MODEL = "llama3.2:3b"     # was "llama3.1:8b"

# System prompt for the agent. It is told, explicitly, that the safety
# results are already decided and are not its job. Its job is test
# prioritisation and evidence retrieval only.

AGENT_SYSTEM_PROMPT = """You are a clinical reasoning assistant for a junior \
physiotherapist assessing an acute knee injury.

IMPORTANT CONSTRAINTS:
- Imaging decisions (Ottawa/Pittsburgh) and red-flag screening have ALREADY \
been decided deterministically. They are provided to you as established facts. \
You must NOT contradict, re-derive, or override them.
- Your job is to (a) suggest which physical special tests to prioritise given \
the findings so far, and (b) retrieve supporting evidence from the corpus using \
the search_corpus tool.
- Every clinical claim you make must be grounded in retrieved evidence. If the \
corpus does not support a claim, say so rather than inventing it.
- You are assisting, not deciding. Frame outputs as suggestions for the \
clinician to review.
- SCOPE: This is an ACUTE TRIAGE assessment only. Do NOT suggest \
return-to-sport tests, hop tests, single-leg performance tests, or \
high-performance athletic screening. These are only appropriate in later \
rehabilitation phases, after acute management is established.
- POPULATION: Adapt all test suggestions to the patient's age and \
presentation as provided in the established facts. Do not suggest tests \
that are inappropriate for the patient's age group or clinical phase.
"""


# ===========================================================
# The agent's one tool: corpus retrieval
# ===========================================================

@tool
def search_corpus(query: str) -> str:
    """Search the clinical knowledge corpus for evidence. Returns top passages.
    Call this ONCE with your best query. Do not call it multiple times."""
    results = retrieve(query)
    if not results:
        return "No relevant evidence found."
    lines = []
    for r in results:
        source = r["metadata"].get("source", "unknown")
        # truncate each chunk to 200 chars so the context stays small
        text = r["text"][:200]
        lines.append(f"[{source}]: {text}")
    return "\n---\n".join(lines)


# ===========================================================
# Result container
# ===========================================================

@dataclass
class AssessmentResult:
    """The full result of one assessment turn.

    Kept as separate fields so the UI can render each block in its correct
    visual register: deterministic results as "System Check" (shield),
    LLM output as "Suggested - review before use" (sparkle).
    """
    # Deterministic block ("System Check")
    red_flag_positive: bool
    red_flag_message: str
    ottawa_result: object = None       # OttawaResult or None
    pittsburgh_result: object = None   # PittsburghResult or None

    # LLM block ("Suggested")
    agent_suggestion: str = ""

    # True if we stopped at the red-flag gate and never ran the LLM.
    stopped_at_red_flag: bool = False


# ===========================================================
# The orchestrator
# ===========================================================

def run_assessment(red_flag_input, ottawa_input, pittsburgh_input, clinician_query):
    """
    Run one acute-knee assessment turn.

    Args:
        red_flag_input:  a RedFlagInput with the screening answers
        ottawa_input:    an OttawaInput with the Ottawa criteria answers
        pittsburgh_input: a PittsburghInput with the Pittsburgh criteria answers
        clinician_query: the free-text question from the clinician, e.g.
                         "which special tests should I prioritise?"

    Returns:
        AssessmentResult
    """

    # -------------------------------------------------------
    # STEP 1: Red-flag screen (deterministic, runs first)
    # -------------------------------------------------------
    # If a red flag fires, we STOP here. The LLM never runs. This is the
    # hard safety gate: nothing the LLM could say should override a red flag,
    # so we don't even give it the chance.

    red_flag_result = screen_red_flags(red_flag_input)

    # RedFlagResult fields: escalate_immediately (bool), triggered_flags (list),
    # rationale (str), action (str). If a flag fires, STOP here - no LLM.
    if red_flag_result.escalate_immediately:
        return AssessmentResult(
            red_flag_positive=True,
            red_flag_message=red_flag_result.rationale + "\n\n" + red_flag_result.action,
            stopped_at_red_flag=True,
        )

    # -------------------------------------------------------
    # STEP 2: Imaging rules (deterministic, computed as facts)
    # -------------------------------------------------------

    ottawa_result = apply_ottawa_knee_rule(ottawa_input)
    pittsburgh_result = apply_pittsburgh_knee_rule(pittsburgh_input)

    # -------------------------------------------------------
    # STEP 3: Agent loop (LLM reasons, with safety facts injected)
    # -------------------------------------------------------
    # We build the LLM, give it the one retrieval tool, and inject the
    # deterministic results into its context as established facts. The agent
    # reasons about test prioritisation and pulls evidence, but cannot touch
    # the safety conclusions.

    llm = ChatOllama(model=LLM_MODEL, temperature=0,
        num_predict=200,      # cap output length so it can't ramble for minutes, changed fro ,350 to 200
        num_ctx=1024,    )

    agent = create_agent(
        model=llm,
        tools=[search_corpus],
        system_prompt=AGENT_SYSTEM_PROMPT,
    )
    
    # Build a patient context string from what we already know deterministically.
# The agent needs this so it can scope its suggestions appropriately.
    patient_context_lines = [
        "PATIENT CONTEXT (use to scope your suggestions):",
        f"- Clinical phase: Acute triage",
        f"- Ottawa age criterion met: {'Yes' if ottawa_input.age_55_or_older else 'No'}",
    ]
    # If the Ottawa age criterion fired, make the implication explicit.
    if ottawa_input.age_55_or_older:
        patient_context_lines.append(
            "- Patient is 55 or older: do NOT suggest return-to-sport tests, "
            "hop tests, or high-performance athletic screening."
        )

    patient_context = "\n".join(patient_context_lines)


    # Facts block handed to the agent. These are the deterministic outputs,
    # phrased as settled context the agent must respect.
    facts_block = (
        "ESTABLISHED FACTS (do not override):\n"
        f"- Ottawa Knee Rule: {ottawa_result.rationale}\n"
        f"- Pittsburgh Knee Rule: {pittsburgh_result.rationale}\n"
        "- Red-flag screen: negative (no red flags detected).\n\n"
        + patient_context + "\n\n"
        + f"CLINICIAN QUESTION: {clinician_query}"
    )

    # create_agent returns a runnable that takes a messages list.
    response = agent.invoke({
        "messages": [{"role": "user", "content": facts_block}]}, config={"recursion_limit": 4},
    )

    # Extract the agent's final text. The exact shape of the response may
    # need a small adjustment depending on the langchain 1.3.10 return type;
    # this pulls the content of the last message.
    # extract the agent's final text
    agent_suggestion = _extract_final_text(response)

    # -------------------------------------------------------
    # STEP 4: Assemble the separated result
    # -------------------------------------------------------
    return AssessmentResult(
        red_flag_positive=False,
        red_flag_message=red_flag_result.rationale,
        ottawa_result=ottawa_result,
        pittsburgh_result=pittsburgh_result,
        agent_suggestion=agent_suggestion,
        stopped_at_red_flag=False,
    )


def _extract_final_text(response):
    """Pull the final assistant text out of a create_agent response.

    create_agent returns a dict with a "messages" list. The last message is
    the agent's final answer. This helper isolates that so the main flow
    stays readable, and gives one place to adjust if the return shape differs.
    """
    messages = response.get("messages", [])
    if not messages:
        return "(no response produced)"

    last = messages[-1]
    # Messages may be dicts or objects depending on version; handle both.
    if isinstance(last, dict):
        return last.get("content", "(no content)")
    return getattr(last, "content", "(no content)")


# ===========================================================
# Quick manual test
# ===========================================================
def run_agent_only(query, safety_facts):
    # thin wrapper so main.py can call the agent without re-running the
    # deterministic gates (those already ran when the case was created)
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,
        num_predict=350,
        num_ctx=1024,
    ) # changed num_ctx from 2048 to 1024 to reduce context window and prevent potential memory issues

    agent = create_agent(
        model=llm,
        tools=[search_corpus],
        system_prompt=AGENT_SYSTEM_PROMPT,
    )

    # inject the pre-computed safety facts plus the clinician question
    full_prompt = safety_facts + "\n\nCLINICIAN QUESTION: " + query

    response = agent.invoke(
        {"messages": [{"role": "user", "content": full_prompt}]},
        config={"recursion_limit": 4},
    ) # recursion_limit=4 is the minimum needed for one tool call + final answer

    suggestion = _extract_final_text(response)

    # return shape matches what main.py expects: suggestion + retrieved stub
    # (retrieved is empty here because retrieval happens inside the agent loop)
    return {
        "suggestion": suggestion,
        "retrieved": [],
    }

if __name__ == "__main__":

    # Example inputs. Replace field values with whatever your dataclasses
    # actually require - these are illustrative.
    #
    # This block is for eyeballing the flow end to end. It will only run
    # once the field names below match your real dataclasses.

    print("This is a manual smoke test. Edit the inputs to match your "
          "dataclass fields, then run:  python -m backend.agent.orchestrator")
