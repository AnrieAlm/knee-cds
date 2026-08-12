# orchestrator.py
#
# The Cygnus agent orchestrator. This is the spine of the system: it runs
# a single acute-knee assessment turn, combining the DETERMINISTIC safety
# layer with the LLM agent, and keeps them architecturally separate.
#
# Design ( safety as a hard gate, NOT as agent tools):
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
# Why this structure matters (my argument): the safety-critical
# outputs are produced by deterministic Python, outside the LLM's control.
# The LLM cannot forget to run a check, cannot run it wrong, and cannot
# talk itself out of a red flag. That is the core safety claim of Cygnus.

from dataclasses import dataclass, field
import os

# --- Deterministic layer imports (no LLM) ---
from backend.rules.ottawa import apply_ottawa_knee_rule, OttawaInput
from backend.rules.pittsburgh import apply_pittsburgh_knee_rule, PittsburghInput
from backend.safety.red_flags import screen_red_flags, RedFlagInput
# addition 1
from backend.agent.physical_context import formatPhysicalForAgent
from backend.investigation_context import build_investigation_context_from_list
# --- Retrieval (used by the agent as a tool) ---
from backend.rag.retriever import retrieve

# --- LangChain v1 agent construction ---
# NOTE: create_agent is the LangChain v1 API (langchain 1.3.10). It replaces
# the older create_react_agent / initialize_agent seen in most tutorials.
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq


# ===========================================================
# Settings
# ===========================================================


LLM_MODEL = "llama-3.3-70b-versatile"


# addition 2
# the agent runs on either backend, the choice is a deployment decision not an architectural one
# local keeps all data on the machine, hosted gives better reasoning at the cost of sending
# case data and retrieved corpus text to a third party
LLM_BACKEND = os.getenv('LLM_BACKEND', 'groq')


def buildLlm():
    if LLM_BACKEND == 'local':
        # imported here so langchain-ollama is only needed when the local path is used
        from langchain_ollama import ChatOllama
        print('using local backend: llama3.2:3b')
        return ChatOllama(
            model='llama3.2:3b',
            temperature=0,
            num_predict=350,
            num_ctx=2048,
        )

    print('using hosted backend:', LLM_MODEL)
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        max_tokens=500,
        api_key=os.getenv('GROQ_API_KEY'),
    )   # groq hosted - much bigger model LLM_MODEL = "llama3.2:3b"     # was "llama3.1:8b"

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

RETRIEVAL INSTRUCTIONS:
- Use search_corpus with specific test names, one at a time if needed. \
For example: 'Lachman test ACL', 'McMurray test meniscus', \
'anterior drawer test', 'valgus stress test collateral ligament'.
- After retrieving, you MUST cite the source tag shown in brackets \
e.g. [magee_ch12] or [jospt_acl_cpg] next to each claim you make.
- If a test is not found in the corpus, say so explicitly. Do NOT \
invent descriptions from general knowledge.

CLINICAL REASONING WITH PHYSICAL FINDINGS:
- If physical examination findings are provided, use them to prioritise your \
test suggestions. A muscle grade marked "(pain limited)" or "(effusion limited)" \
reflects inhibition, NOT neurological weakness - do not interpret it as a \
neurological finding.
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
        # relevance is carried through so the audit log can record how well
        # each chunk actually matched — a citation backed by a 0.55 score is
        # a different claim from one backed by 0.85
        score = r.get("score", r.get("distance", ""))
        text = r["text"][:200]
        if score != "":
            lines.append(f"[{source} | relevance {score}]: {text}")
        else:
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

    llm = buildLlm()
   # replacing with  llm = ChatGroq(
   # model=LLM_MODEL,
  ## max_tokens=500,         # replaces num_predict
   # api_key=os.getenv##('GROQ_API_KEY'),
#)

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
        "messages": [{"role": "user", "content": facts_block}]}, config={"recursion_limit": 10},
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

def _extract_retrieved(response):
    """Pull the retrieval chain out of a create_agent response.

    This is the audit trail. The ReAct loop calls search_corpus internally,
    so without capturing the tool messages here there is no independent
    record of what the model was actually shown — only the citations it
    chose to write into its prose, which cannot be verified against
    anything. Every log entry written before this existed has an empty
    retrieved list for that reason.

    Returns a list of {query, result} dicts, one per tool call, in the
    order the agent made them.
    """
    messages = response.get("messages", [])
    retrieved = []
    pending_query = None

    for msg in messages:
        # tool CALL — carries the query the agent chose
        tool_calls = None
        if isinstance(msg, dict):
            tool_calls = msg.get("tool_calls")
        else:
            tool_calls = getattr(msg, "tool_calls", None)

        if tool_calls:
            for call in tool_calls:
                if isinstance(call, dict):
                    args = call.get("args", {})
                else:
                    args = getattr(call, "args", {})
                pending_query = args.get("query", "") if isinstance(args, dict) else ""

        # tool RESULT — carries what came back
        msg_type = None
        if isinstance(msg, dict):
            msg_type = msg.get("type") or msg.get("role")
        else:
            msg_type = getattr(msg, "type", None)

        if msg_type == "tool":
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            retrieved.append({
                "query": pending_query or "(query not captured)",
                "result": content,
            })
            pending_query = None

    return retrieved
# ===========================================================
# Quick manual test
# ===========================================================
def run_agent_only(query, safety_facts, physical_dict=None, investigations=None):
    # thin wrapper so main.py can call the agent without re-running the
    # deterministic gates (those already ran when the case was created)
    llm = buildLlm() 
    # changed num_ctx from 2048 to 1024 to reduce context window and prevent potential memory issues

    agent = create_agent(
        model=llm,
        tools=[search_corpus],
        system_prompt=AGENT_SYSTEM_PROMPT,
    )

    

    # addition 3
    # physical findings are recorded observations, not rule outputs, so they go in their own
    # labelled block rather than inside the ESTABLISHED FACTS header
    physical_block = ''
    if physical_dict:
        physical_text = formatPhysicalForAgent(physical_dict)
        physical_block = (
            '\n\nPHYSICAL EXAMINATION (recorded by the clinician, do not contradict):\n'
            + physical_text
        )

    # Prior investigations. Placed after the physical block so the agent reads
    # the clinician's own findings before any radiology report, mirroring the
    # tab order. Only clinician-verified transcriptions are included; that gate
    # is enforced in investigation_context, not here.
    investigation_block = "\n\n" + build_investigation_context_from_list(investigations)

    # inject the pre-computed safety facts, the physical findings, the prior
    # investigations and the clinician question
    full_prompt = (
        safety_facts
        + physical_block
        + investigation_block
        + "\n\nCLINICIAN QUESTION: "
        + query
    )


    response = agent.invoke(
        {"messages": [{"role": "user", "content": full_prompt}]},
        config={"recursion_limit": 10},
    ) # recursion_limit=10 allows several retrieval steps before answering

    suggestion = _extract_final_text(response)
    retrieved = _extract_retrieved(response)

    print(f'[agent] {len(retrieved)} retrieval call(s) captured')

    return {
        "suggestion": suggestion,
        "retrieved": retrieved,
    }

if __name__ == "__main__":

    # Example inputs. Replace field values with whatever your dataclasses
    # actually require - these are illustrative.
    #
    # This block is for eyeballing the flow end to end. It will only run
    # once the field names below match your real dataclasses.

    print("This is a manual smoke test. Edit the inputs to match your "
          "dataclass fields, then run:  python -m backend.agent.orchestrator")
