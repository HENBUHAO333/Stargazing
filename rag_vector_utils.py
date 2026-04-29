import json
import os
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

VECTOR_DIR = Path("vector_store")
INDEX_PATH = VECTOR_DIR / "stargazing.faiss"
CHUNKS_PATH = VECTOR_DIR / "chunks.json"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def get_openai_client() -> OpenAI:
    """
    Works locally with .env and on Streamlit Cloud with st.secrets.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY for vector retrieval.")

    return OpenAI(api_key=api_key)


def load_vector_store():
    """
    Load FAISS index and chunk metadata.
    """

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Vector index not found at {INDEX_PATH}. "
            "Run `python3 build_vector_index.py` first."
        )

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunk metadata not found at {CHUNKS_PATH}. "
            "Run `python3 build_vector_index.py` first."
        )

    index = faiss.read_index(str(INDEX_PATH))

    chunks = json.loads(
        CHUNKS_PATH.read_text(encoding="utf-8")
    )

    return index, chunks


def embed_query(query: str) -> np.ndarray:
    """
    Embed the user query / forecast query using the same embedding model
    used in build_vector_index.py.
    """

    client = get_openai_client()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )

    vector = np.array([response.data[0].embedding], dtype="float32")

    # Normalize for cosine similarity with IndexFlatIP.
    faiss.normalize_L2(vector)

    return vector


def retrieve_vector_context(query: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieve top-k relevant chunks from the FAISS vector store.
    """

    index, chunks = load_vector_store()
    query_vector = embed_query(query)

    scores, indices = index.search(query_vector, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        chunk = chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results


def format_vector_context(results: List[Dict]) -> str:
    """
    Format retrieved chunks for LLM context.
    """

    if not results:
        return "No vector knowledge context was retrieved."

    blocks = []

    for item in results:
        source = item.get("source", "unknown")
        chunk_id = item.get("chunk_id", "unknown")
        score = item.get("score", 0)
        text = item.get("text", "")

        blocks.append(
            f"[Source: {source} | Chunk ID: {chunk_id} | Similarity: {score:.3f}]\n"
            f"{text}"
        )

    return "\n\n---\n\n".join(blocks)
