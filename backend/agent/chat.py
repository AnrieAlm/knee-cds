# backend/agent/chat.py
# Lightweight RAG chat handler for the Cygnus clinical assistant.
# Unlike the ReAct orchestrator, this does a single retrieval pass
# and a single LLM call — no agent loop. Fast enough for synchronous
# use in the chat panel on the Exam tab.

from backend.rag.retriever import retrieve
from langchain_ollama import OllamaLLM

# same model as the agent — keeps hardware requirements identical
_llm = OllamaLLM(
    model='llama3.2:3b',
    num_predict=350,
    num_ctx=2048,
    temperature=0.2,
)


def run_chat(message: str, case: dict) -> str:
    # retrieve relevant passages from the corpus
    results = retrieve(message)

    # build a context string from retrieved chunks
    corpus_context = '\n\n'.join(
        f'[Source {i+1}]: {r["text"]}'
        for i, r in enumerate(results)
    ) if results else 'No relevant passages found in corpus.'

    # build a case context string from saved history if it exists
    history = case.get('history', {})
    if history:
        case_context = (
            f"Current case: {history.get('activity_level', 'unknown')} patient, "
            f"{history.get('involved_side', 'unknown')} knee. "
            f"Mechanism: {history.get('mechanism_type', 'unknown')} — "
            f"{history.get('mechanism_description', 'no description')}. "
            f"Pain location: {history.get('pain_location', 'not recorded')}. "
            f"Swelling: {history.get('swelling_present', 'unknown')} "
            f"({history.get('swelling_onset', 'none')}). "
            f"Weight-bearing: {history.get('weight_bearing', 'unknown')}. "
            f"Goal: {history.get('patient_goal', 'not recorded')}."
        )
    else:
        case_context = 'No case history recorded yet.'

    # build the prompt
    prompt = (
        'You are a clinical assistant supporting a physiotherapist assessing an acute knee injury.\n'
        'Answer the question below using the corpus passages provided.\n'
        'Be concise and clinically specific. Do not invent tests or facts not in the passages.\n\n'
        f'CASE CONTEXT:\n{case_context}\n\n'
        f'CORPUS PASSAGES:\n{corpus_context}\n\n'
        f'QUESTION: {message}\n\n'
        'ANSWER:'
    )

    response = _llm.invoke(prompt)
    return response.strip()