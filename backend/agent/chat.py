# backend/agent/chat.py
# Lightweight RAG chat handler for the Cygnus clinical assistant.
# Unlike the ReAct orchestrator, this does a single retrieval pass
# and a single LLM call - no agent loop. Fast enough for synchronous
# use in the chat panel on the Exam tab.

from backend.rag.retriever import retrieve
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from backend.constants import GEN_MODEL
load_dotenv()

# same model as orchestrator.py - keep this in sync if you change it there
#LLM_MODEL = 'llama-3.3-70b-versatile'
LLM_MODEL = GEN_MODEL
LLM_BACKEND = os.getenv('LLM_BACKEND', 'groq')
LOCAL_MODEL = os.getenv('LOCAL_MODEL', 'llama3.1:8b')
# build the LLM once at module load - reused for every 
# 
# chat call

def _build_llm():
    if LLM_BACKEND == 'local':
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=LOCAL_MODEL,
            temperature=0,
            num_predict=800,
            num_ctx=4096,
        )
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        max_tokens=1500,
        api_key=os.getenv('GROQ_API_KEY'),
    )


_llm = _build_llm()
_llm = ChatGroq(
    model=LLM_MODEL,
    temperature=0,
    max_tokens=500,
    api_key=os.getenv('GROQ_API_KEY'),
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
            f"Mechanism: {history.get('mechanism_type', 'unknown')} - "
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

    # .content pulls the text out of the LangChain message object
    return response.content.strip()