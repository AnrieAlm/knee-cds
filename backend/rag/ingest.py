# ingest.py
# Builds the ChromaDB collection from the corpus on disk.
# This is an idempotent full rebuild: every time you run it,
# the old collection is wiped and rebuilt from scratch.
# That way we never end up with stale or duplicated chunks.
#
# Two chunking paths:
#   1. Prose sources (Magee chapters, JOSPT CPGs)
#      -> split by markdown headers first
#      -> then use a length guard to break up any oversized section
#   2. Special-test stubs in corpus/tests/
#      -> one chunk per file, no splitting (atomic)

import os
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


# -----------------------------------------------------------
# Settings (all in one place so they're easy to change later)
# -----------------------------------------------------------

# Where ChromaDB stores its files on the external drive.
# Must match retriever.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = str(PROJECT_ROOT / "chroma_db")

# The name of the single collection where all chunks live.
# Must match retriever.py.
COLLECTION_NAME = "cygnus_corpus"

# The embedding model Ollama runs locally.
# Must match retriever.py so the vectors are comparable.
EMBEDDING_MODEL = "nomic-embed-text"

# Root folder of the corpus. Adjust if your layout is different.
CORPUS_ROOT = PROJECT_ROOT / "corpus" / "guidelines"

# Subfolder that contains the atomic special-test stub files.
# Every .md file inside this folder becomes exactly one chunk.
TESTS_FOLDER = CORPUS_ROOT / "test_stubs"

# Length guard for the prose splitter.
# If a header-based section is longer than this many characters,
# we break it into smaller pieces so no single chunk is too big
# for the embedding model to represent well.
MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150

# The headers we tell MarkdownHeaderTextSplitter to split on.
# Each tuple is (markdown syntax, metadata field name).
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


# -----------------------------------------------------------
# Helper: split one prose markdown file into chunks
# -----------------------------------------------------------

def split_prose_file(file_path):
    """
    Reads a markdown file and returns a list of chunk dicts.

    Each chunk dict looks like:
    {
        "id": "unique-id-string",
        "text": "the chunk text",
        "metadata": { "source": "filename.md", "h1": "...", ... }
    }
    """

    # Read the whole file into a string.
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Step 1: split by markdown headers.
    # This gives us sections that respect the document's own structure.
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
    )
    header_sections = header_splitter.split_text(full_text)

    # Step 2: length guard.
    # If any header section is longer than MAX_CHUNK_CHARS,
    # break it into smaller pieces with a bit of overlap
    # so context isn't lost at chunk boundaries.
    length_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
    )
    final_sections = length_splitter.split_documents(header_sections)

    # Step 3: turn each section into our chunk dict format.
    chunks = []

    for section in final_sections:

        # Every chunk needs a unique id so retriever.py can
        # report it and agentLog can store it.
        chunk_id = str(uuid.uuid4())

        # Copy over the metadata that MarkdownHeaderTextSplitter added
        # (the h1/h2/h3 fields), then add the source filename.
        metadata = dict(section.metadata)
        metadata["source"] = file_path.name
        metadata["id"] = chunk_id

        chunks.append({
            "id": chunk_id,
            "text": section.page_content,
            "metadata": metadata,
        })

    return chunks


# -----------------------------------------------------------
# Helper: turn one special-test stub file into a single chunk
# -----------------------------------------------------------

def load_test_stub(file_path):
    """
    Reads a special-test stub markdown file and returns one chunk dict.
    No splitting is done because these files are already short and
    each file describes exactly one test.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunk_id = str(uuid.uuid4())

    metadata = {
        "source": file_path.name,
        "id": chunk_id,
        # Mark this chunk as a test stub so we can filter on it later
        # if we ever want to (e.g. "only search test stubs").
        "chunk_type": "test_stub",
    }

    return {
        "id": chunk_id,
        "text": text,
        "metadata": metadata,
    }


# -----------------------------------------------------------
# Wipe the old collection so we start fresh every run
# -----------------------------------------------------------

def wipe_collection(embeddings):
    """
    Deletes the existing collection if it exists.
    This is what makes the ingest idempotent: run it twice
    and you get the same result, not duplicates.
    """

    # Open the persisted store just so we can drop the collection.
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

    # delete_collection removes all chunks in this collection.
    store.delete_collection()

    print("Old collection wiped (if it existed).")


# -----------------------------------------------------------
# Main ingestion routine
# -----------------------------------------------------------

def main():

    # Set up the embedding function once and reuse it.
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Wipe first so we're starting from a clean slate.
    wipe_collection(embeddings)

    # We'll collect every chunk from every file into these lists,
    # then send them to ChromaDB in one batch at the end.
    all_texts = []
    all_metadatas = []
    all_ids = []

    # -----------------------------------------------------------
    # Path 1: prose files at the top level of the corpus.
    # We look for every .md file in CORPUS_ROOT itself
    # (not recursing yet) and treat them all as prose.
    # -----------------------------------------------------------

    prose_files = [
        f for f in CORPUS_ROOT.iterdir()
        if f.is_file() and f.suffix == ".md"
    ]

    print("Found", len(prose_files), "prose files.")

    for file_path in prose_files:

        print("Splitting prose file:", file_path.name)
        chunks = split_prose_file(file_path)
        print("  -> produced", len(chunks), "chunks")

        for chunk in chunks:
            all_texts.append(chunk["text"])
            all_metadatas.append(chunk["metadata"])
            all_ids.append(chunk["id"])

    # -----------------------------------------------------------
    # Path 2: atomic test-stub files in corpus/tests/.
    # Each file becomes exactly one chunk, no splitting.
    # -----------------------------------------------------------

    if TESTS_FOLDER.exists():

        test_files = [
            f for f in TESTS_FOLDER.iterdir()
            if f.is_file() and f.suffix == ".md"
        ]

        print("Found", len(test_files), "test-stub files.")

        for file_path in test_files:

            print("Loading test stub:", file_path.name)
            chunk = load_test_stub(file_path)

            all_texts.append(chunk["text"])
            all_metadatas.append(chunk["metadata"])
            all_ids.append(chunk["id"])

    else:
        print("No tests folder found at", TESTS_FOLDER)

    # -----------------------------------------------------------
    # Send everything to ChromaDB in one go
    # -----------------------------------------------------------

    print()
    print("Total chunks to ingest:", len(all_texts))

    if len(all_texts) == 0:
        print("Nothing to ingest. Check your corpus paths.")
        return

    # Open a fresh Chroma store (the old one was wiped above).
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

    # add_texts embeds every text with Ollama and stores the
    # vectors along with the metadata and ids in Chroma.
    # This is the slow step, since it calls the embedding model
    # once per chunk.
    store.add_texts(
        texts=all_texts,
        metadatas=all_metadatas,
        ids=all_ids,
    )

    print("Ingestion complete.")
    print("Chunks stored:", len(all_texts))


# -----------------------------------------------------------
# Standard Python entry point
# -----------------------------------------------------------

if __name__ == "__main__":
    main()