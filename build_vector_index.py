import json
import os
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

# ============================================================
# CONFIG
# ============================================================

KNOWLEDGE_DIR = Path("data/knowledge_curated")
VECTOR_DIR = Path("vector_store")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = VECTOR_DIR / "stargazing.faiss"
CHUNKS_PATH = VECTOR_DIR / "chunks.json"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


# ============================================================
# OPENAI CLIENT
# ============================================================

def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing OPENAI_API_KEY. Add it to your .env file before building the vector index."
        )

    return OpenAI(api_key=api_key)


# ============================================================
# READ DOCUMENTS
# ============================================================

def read_markdown_files() -> List[Dict]:
    docs = []

    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"{KNOWLEDGE_DIR} does not exist. Create data/knowledge_curated first."
        )

    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()

        if not text:
            continue

        docs.append(
            {
                "source": path.name,
                "path": str(path),
                "text": text,
            }
        )

    return docs


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = 700,
    overlap: int = 120,
) -> List[str]:
    """
    Word-based chunking.

    chunk_size and overlap are in words, not characters.
    700 words is enough context without making retrieval too broad.
    """

    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def build_chunks(docs: List[Dict]) -> List[Dict]:
    all_chunks = []

    for doc in docs:
        chunks = chunk_text(doc["text"])

        for local_chunk_id, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "chunk_id": len(all_chunks),
                    "source": doc["source"],
                    "path": doc["path"],
                    "local_chunk_id": local_chunk_id,
                    "text": chunk,
                }
            )

    return all_chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def embed_texts(
    client: OpenAI,
    texts: List[str],
    batch_size: int = 50,
) -> np.ndarray:
    vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        batch_vectors = [item.embedding for item in response.data]
        vectors.extend(batch_vectors)

        print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")

    arr = np.array(vectors, dtype="float32")

    # Normalize vectors so FAISS inner product behaves like cosine similarity.
    faiss.normalize_L2(arr)

    return arr


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index


# ============================================================
# MAIN
# ============================================================

def main():
    docs = read_markdown_files()

    if not docs:
        raise ValueError("No markdown files found in data/knowledge_curated.")

    chunks = build_chunks(docs)

    if not chunks:
        raise ValueError("No chunks were created from the curated documents.")

    print(f"Loaded {len(docs)} curated documents.")
    print(f"Created {len(chunks)} chunks.")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    client = get_client()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(client, texts)

    index = build_faiss_index(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Saved FAISS index to: {INDEX_PATH}")
    print(f"Saved chunk metadata to: {CHUNKS_PATH}")
    print(f"Vector count: {index.ntotal}")


if __name__ == "__main__":
    main()
    
