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
import re

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
from langgraph.errors import GraphRecursionError
from langchain.tools import tool
from langchain_groq import ChatGroq
from groq import APIStatusError as GroqAPIStatusError
import time  

# ===========================================================
# Settings
# ===========================================================
from backend.constants import GEN_MODEL
LLM_MODEL = GEN_MODEL

#LLM_MODEL = "llama-3.3-70b-versatile"


# addition 2
# the agent runs on either backend, the choice is a deployment decision not an architectural one
# local keeps all data on the machine, hosted gives better reasoning at the cost of sending
# case data and retrieved corpus text to a third party
LLM_BACKEND = os.getenv('LLM_BACKEND', 'groq')
LOCAL_MODEL = os.getenv('LOCAL_MODEL', 'llama3.1:8b')

# Retrieval budget, evidence length, and results-per-call. All three were
# tuned empirically against two real failures observed in testing:
#
# 1. The original 200-character evidence snippet and unbounded retrieval
#    loop caused the ReAct agent to repeatedly re-query the same or
#    near-duplicate tests rather than reaching a final answer (observed
#    via a temporary debug print of every search_corpus query).
#
# 2. Fixing (1) by raising evidence length and adding a 5-call budget
#    then produced a second failure: retrieve() returns TOP_K=5 chunks
#    per call (retriever.py), and a single search_corpus call was
#    returning ALL of them, so one call could return up to
#    TOP_K * MAX_EVIDENCE_CHARS characters. Since ReAct history keeps
#    every prior tool result in context for every subsequent model call,
#    5 tool calls compounded to roughly 7500 tokens of evidence alone,
#    crossing Groq's free-tier 8000 TPM limit on the 5th call (observed
#    directly: groq.APIStatusError, "Requested 8147" vs "Limit 8000").
#
# MAX_RESULTS_PER_CALL caps the multiplicative factor directly, rather
# than only shrinking MAX_EVIDENCE_CHARS, which only reduces one side of
# the multiplication. The top 3 of 5 results by relevance carry nearly
# all the useful signal, since retrieve() already ranks by score.
MAX_RETRIEVAL_CALLS = 4
MAX_EVIDENCE_CHARS = 800
MAX_RESULTS_PER_CALL = 3


def buildLlm():
    if LLM_BACKEND == 'local':
        from langchain_ollama import ChatOllama
        print('using local backend:', LOCAL_MODEL)
        return ChatOllama(
            model=LOCAL_MODEL,
            temperature=0,
            num_predict=1200,
            num_ctx=4096,
        )
    print('using hosted backend:', LLM_MODEL)
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        # Raised from 700, then settled at 1500 on 23 August 2026. The
        # replacement generation model emits reasoning tokens drawn from the
        # same completion budget, so the original ceiling truncated the ReAct
        # trace. A 2500 ceiling in turn exceeded the free-tier limit of 8000
        # tokens per minute once retrieval context was included.
        max_tokens=1500,
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
- First decide which 3-5 special tests are most relevant to this \
presentation. Then use search_corpus once per test, e.g. \
'Lachman test ACL', 'McMurray test meniscus', 'anterior drawer test', \
'valgus stress test collateral ligament'.
- You have a maximum retrieval budget of 5 search_corpus calls. Do NOT \
search for the same test twice, and do NOT issue a reworded or \
paraphrased repeat of a search you have already made.
- Once you have retrieved evidence for your shortlisted tests, STOP \
using tools and write the final answer. Do not search for additional \
tests merely because a result feels incomplete.
- After retrieving, you MUST cite the source tag shown in brackets \
e.g. [magee_ch12] or [jospt_acl_cpg] next to each claim you make.
- If a test is not found in the corpus, say so explicitly. Do NOT \
invent descriptions from general knowledge.

CLINICAL REASONING WITH PHYSICAL FINDINGS:
- If physical examination findings are provided, use them to prioritise your \
test suggestions. A muscle grade marked "(pain limited)" or "(effusion limited)" \
reflects inhibition, NOT neurological weakness - do not interpret it as a \
neurological finding.
OUTPUT FORMAT:
- Suggest 3-5 tests, highest priority first.
- One sentence of rationale per test, with its citation inline.
- Stop after the last test. Do NOT write a closing summary, do NOT restate
  the constraints above, and do NOT list what you excluded or why.
"""


# ===========================================================
# The agent's one tool: corpus retrieval
#
# search_corpus is built fresh inside run_agent_only, NOT defined once at
# module level. This gives each call its own private retrieval counter,
# evidence list, and seen-queries set via closure, rather than a
# module-level global. A global would be shared across every concurrent
# request the live FastAPI server handles (main.py runs run_agent_only
# inside a background thread per /suggest-async call), so two clinicians
# generating suggestions for two different cases at the same time would
# silently share, and corrupt, each other's retrieval budget. Since
# create_agent() already builds a fresh agent on every call, building
# search_corpus fresh alongside it costs nothing extra and removes the
# concurrency hazard entirely.
# ===========================================================

def _normalize_query(query: str) -> str:
    """Collapse whitespace/case/filler so near-duplicate queries
    ('patellar grind test evidence' vs 'patellar grind test') are
    recognised as the same request rather than two distinct ones."""
    q = query.lower().strip()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"\bevidence\b", "", q)
    return re.sub(r"\s+", " ", q).strip()


def _build_search_corpus_tool(collected_evidence: list, seen_queries: set):
    """Builds a search_corpus tool bound to this call's own counter,
    evidence list, and seen-queries set. A fresh one is built per
    run_agent_only invocation, so state never leaks across requests."""

    @tool
    def search_corpus(query: str) -> str:
        """Search the clinical knowledge corpus for evidence. Returns top passages.
        Call once per structure or test you are considering — e.g. separately for
        'McMurray test meniscus' and 'valgus stress MCL'. A single query anchored to
        one hypothesis will only return evidence for that hypothesis."""

        if len(collected_evidence) >= MAX_RETRIEVAL_CALLS:
            return (
                "RETRIEVAL BUDGET EXHAUSTED. Do not call search_corpus again. "
                "Write the final answer now using the evidence already retrieved."
            )

        normalized = _normalize_query(query)
        if normalized in seen_queries:
            return (
                "This test has already been searched. Use the evidence already "
                "retrieved for it rather than searching again."
            )
        seen_queries.add(normalized)

        print(f"[agent tool] search_corpus query = {query!r} "
              f"({len(collected_evidence) + 1}/{MAX_RETRIEVAL_CALLS})")

        results = retrieve(query)
        if not results:
            collected_evidence.append({"query": query, "text": "No relevant evidence found."})
            return "No relevant evidence found."

        # retrieve() returns up to TOP_K=5 chunks, already ranked by
        # relevance. Only the top MAX_RESULTS_PER_CALL are sent to the
        # model; see the comment on the constants above for why.
        results = results[:MAX_RESULTS_PER_CALL]

        lines = []
        for r in results:
            source = r["metadata"].get("source", "unknown")
            # relevance is carried through so the audit log can record how well
            # each chunk actually matched — a citation backed by a 0.55 score is
            # a different claim from one backed by 0.85
            score = r.get("score", r.get("distance", ""))
            text = r["text"][:MAX_EVIDENCE_CHARS]
            if score != "":
                lines.append(f"[{source} | relevance {score}]: {text}")
            else:
                lines.append(f"[{source}]: {text}")

        formatted = "\n---\n".join(lines)
        collected_evidence.append({"query": query, "text": formatted})
        return formatted

    return search_corpus


# ===========================================================
# Result container
# ===========================================================

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


def _format_fallback_evidence(collected_evidence: list) -> str:
    """Turns the evidence gathered before a recursion-limit fallback into
    plain text the LLM can synthesize from directly."""
    sections = []
    for i, item in enumerate(collected_evidence, 1):
        sections.append(f"RETRIEVAL {i}\nQuery: {item['query']}\n{item['text']}")
    return "\n\n".join(sections) if sections else "(no evidence was retrieved before the limit was reached)"


def _run_synthesis_fallback(llm, patient_prompt: str, collected_evidence: list) -> str:
    """Called only if the ReAct loop hits GraphRecursionError. Makes one
    direct, tool-free call to the LLM using whatever evidence was already
    gathered, so a demo run always produces a suggestion (or an honest
    statement that evidence was insufficient) rather than an unhandled
    error reaching the clinician."""
    evidence_text = _format_fallback_evidence(collected_evidence)

    fallback_prompt = f"""You are the final clinical reasoning stage. The normal \
agent exceeded its tool-use budget before producing a final answer. Do NOT \
request any further retrieval — none is available.

Answer the clinician's question using ONLY the established facts, patient \
context, and retrieved evidence below. If the retrieved evidence is \
insufficient to support a specific claim, say so explicitly rather than \
inventing it.

{patient_prompt}

RETRIEVED EVIDENCE:
{evidence_text}

Provide the best clinically appropriate answer now, in the same format as \
normal (3-5 prioritised tests, one sentence of rationale each, citation \
inline). Do not mention the recursion limit or any internal agent failure.
"""
    response = llm.invoke(fallback_prompt)
    return getattr(response, "content", str(response))


# ===========================================================
# Quick manual test
# ===========================================================
# deferrals arrives separately from physical_dict because it lives on the case
# document rather than inside physical. Keyword with a None default so the
# existing positional calls keep working.
def run_agent_only(query, safety_facts, physical_dict=None, investigations=None,
                   deferrals=None, involved_side=None):
    # thin wrapper so main.py can call the agent without re-running the
    # deterministic gates (those already ran when the case was created)
    llm = buildLlm()
    # changed num_ctx from 2048 to 1024 to reduce context window and prevent potential memory issues

    # Per-call state, isolated by closure — see the note above
    # _build_search_corpus_tool for why this replaces a module-level global.
    collected_evidence = []
    seen_queries = set()
    search_corpus = _build_search_corpus_tool(collected_evidence, seen_queries)

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
        physical_text = formatPhysicalForAgent(physical_dict, deferrals=deferrals)
        physical_block = (
            '\n\nPHYSICAL EXAMINATION (recorded by the clinician, do not contradict):\n'
            + physical_text
        )

    # Prior investigations. Placed after the physical block so the agent reads
    # the clinician's own findings before any radiology report, mirroring the
    # tab order. Only clinician-verified transcriptions are included; that gate
    # is enforced in investigation_context, not here.
    investigation_block = "\n\n" + build_investigation_context_from_list(
        investigations, involved_side=involved_side
    )

    # inject the pre-computed safety facts, the physical findings, the prior
    # investigations and the clinician question
    full_prompt = (
        safety_facts
        + physical_block
        + investigation_block
        + "\n\nCLINICIAN QUESTION: "
        + query
    )
    print(f"[agent] full prompt ({len(full_prompt)} chars):\n{full_prompt}\n[/agent]")

    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": full_prompt}]},
            config={"recursion_limit": 25},
        )
        suggestion = _extract_final_text(response)
        retrieved = _extract_retrieved(response)

    except (GraphRecursionError, GroqAPIStatusError) as exc:
        # Real safety net, not a prompt hope. The retrieval budget,
        # duplicate guard, and per-call results cap above make both
        # failures below rare, but neither an LLM following instructions
        # nor a third-party API's rate limit is a guarantee, and a live
        # demo cannot show the clinician an unhandled error. Fall back to
        # a single direct synthesis call using whatever evidence was
        # already gathered before the failure.
        #
        # GraphRecursionError: the ReAct loop failed to converge within
        # the step budget (observed during testing on a clean case with
        # the original 200-char / unbounded-call configuration).
        #
        # GroqAPIStatusError (code 413): the accumulated ReAct message
        # history, including all retrieved evidence from prior tool
        # calls, exceeded Groq's free-tier 8000 TPM limit (observed
        # directly: "Requested 8147" on the 5th retrieval call, before
        # MAX_RESULTS_PER_CALL was added).
        print(f"[agent] {type(exc).__name__} during agent.invoke; using synthesis fallback: {exc}")
        suggestion = _run_synthesis_fallback(llm, full_prompt, collected_evidence)
        retrieved = [{"query": item["query"], "result": item["text"]} for item in collected_evidence]

    print(f'[agent] {len(retrieved)} retrieval call(s) captured')

    return {
        "suggestion": suggestion,
        "retrieved": retrieved,
    }