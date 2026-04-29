from pathlib import Path
import re
from typing import List, Dict


KNOWLEDGE_DIR = Path(__file__).parent / "data" / "knowledge"


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_knowledge_chunks(chunk_size: int = 900, overlap: int = 120) -> List[Dict]:
    chunks = []

    if not KNOWLEDGE_DIR.exists():
        return chunks

    files = list(KNOWLEDGE_DIR.glob("*.md")) + list(KNOWLEDGE_DIR.glob("*.txt"))

    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        text = _clean_text(text)

        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append(
                    {
                        "source": file_path.name,
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                    }
                )

            chunk_id += 1
            start += chunk_size - overlap

    return chunks


def _tokenize(text: str) -> set:
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text)

    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "into",
        "are", "is", "of", "to", "a", "an", "in", "on", "it", "as",
        "by", "or", "be", "can", "usually", "what", "why", "how",
    }

    return {t for t in tokens if t not in stopwords}


def retrieve_context(query: str, top_k: int = 4) -> List[Dict]:
    chunks = load_knowledge_chunks()

    if not chunks:
        return []

    query_tokens = _tokenize(query)
    scored = []

    for chunk in chunks:
        chunk_tokens = _tokenize(chunk["text"])
        overlap = query_tokens.intersection(chunk_tokens)

        score = len(overlap) / max(len(query_tokens), 1)

        scored.append(
            {
                **chunk,
                "score": score,
            }
        )

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)

    return [s for s in scored[:top_k] if s["score"] > 0]


def format_retrieved_context(retrieved: List[Dict]) -> str:
    if not retrieved:
        return "No external knowledge context was retrieved."

    blocks = []

    for item in retrieved:
        blocks.append(
            f"[Source: {item['source']} | Chunk: {item['chunk_id']}]\n"
            f"{item['text']}"
        )

    return "\n\n---\n\n".join(blocks)
