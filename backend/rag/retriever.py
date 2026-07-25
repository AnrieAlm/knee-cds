# retriever.py
# Retrieves the most relevant chunks from the ChromaDB collection
# for a given clinical query. Applies a relevance floor of 0.3.
# Returns chunk IDs + scores + text + metadata so the agent can
# use them and the agentLog can store IDs and scores.

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# -----------------------------------------------------------
# Settings (all in one place so they're easy to change later)
# -----------------------------------------------------------

# Where ChromaDB stores its files on the external drive.
# This must match the path used in ingest.py.
CHROMA_PATH = "/media/anriel/LENOVO_USB_HDD/knee_cds/chroma_db"

# The name of the single collection where all chunks live.
# This must match the name used in ingest.py.
COLLECTION_NAME = "cygnus_corpus"

# The embedding model Ollama runs locally.
# Must be the same model used during ingestion, otherwise
# the query vector and stored vectors won't be comparable.
EMBEDDING_MODEL = "nomic-embed-text"

# How many results to fetch from ChromaDB before filtering.
# We ask for a few extra because some may fall below the floor.
TOP_K = 5

# The relevance floor. Any chunk with a score below this is
# discarded. 0.3 is the value locked in the spec.
RELEVANCE_FLOOR = 0.3


# -----------------------------------------------------------
# Load the vector store once when this file is imported
# -----------------------------------------------------------

def load_vector_store():
    """
    Connects to the existing ChromaDB collection on disk.
    Returns a Chroma object we can search against.
    """

    # Set up the embedding function. This is what turns
    # the user's query text into a vector at search time.
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Open the persisted collection. Because we pass
    # persist_directory, Chroma reads the existing files
    # rather than creating a new empty collection.
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

    return vector_store


# We load the store once at module level. This means every
# call to retrieve() reuses the same connection instead of
# opening a new one each time.
VECTOR_STORE = load_vector_store()


# -----------------------------------------------------------
# The main function the agent will call
# -----------------------------------------------------------

def retrieve(query):
    """
    Takes a plain-text clinical query (e.g. "anterior drawer test").
    Returns a list of dicts, one per relevant chunk.

    Each dict looks like:
    {
        "chunk_id": "some-id-string",
        "score": 0.72,
        "text": "the chunk text ...",
        "metadata": { "source": "magee_ch12_part1.md", ... }
    }

    If no chunk passes the relevance floor, returns an empty list.
    """

    # Ask Chroma for the top K most similar chunks along with
    # their relevance scores. Scores are normalised between 0 and 1,
    # where higher means more relevant.
    raw_results = VECTOR_STORE.similarity_search_with_relevance_scores(
        query=query,
        k=TOP_K,
    )

    # raw_results is a list of tuples: (Document, score).
    # We'll walk through them one by one and keep only the good ones.
    filtered_results = []

    for document, score in raw_results:

        # Skip anything below the relevance floor.
        if score < RELEVANCE_FLOOR:
            continue

        # Pull the chunk's ID out of its metadata.
        # ingest.py must have stored an "id" field for every chunk.
        # If it's missing for some reason, fall back to "unknown".
        chunk_id = document.metadata.get("id", "unknown")

        # Build a simple dict for this result.
        result = {
            "chunk_id": chunk_id,
            "score": score,
            "text": document.page_content,
            "metadata": document.metadata,
        }

        # Add it to the list we'll return.
        filtered_results.append(result)

    return filtered_results


# -----------------------------------------------------------
# Quick manual test
# Run this file directly with: python retriever.py
# -----------------------------------------------------------

if __name__ == "__main__":

    # A couple of test queries a physio might actually ask.
    test_queries = [
        "anterior drawer test for ACL",
        "Ottawa knee rule criteria",
        "meniscal tear special tests",
    ]

    # Run each query and print the results in a readable way.
    for query in test_queries:

        print("=" * 60)
        print("QUERY:", query)
        print("=" * 60)

        results = retrieve(query)

        # If nothing passed the floor, say so clearly.
        if len(results) == 0:
            print("No chunks passed the relevance floor of", RELEVANCE_FLOOR)
            print()
            continue

        # Otherwise print each result.
        for i, result in enumerate(results):
            print()
            print("Result", i + 1)
            print("  chunk_id :", result["chunk_id"])
            print("  score    :", round(result["score"], 3))
            print("  source   :", result["metadata"].get("source", "unknown"))
            print("  text     :", result["text"][:200], "...")

        print()