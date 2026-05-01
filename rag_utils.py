from pathlib import Path
import re
from typing import List, Dict, Optional


BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIRS = [
    BASE_DIR / "data" / "knowledge_curated",
    BASE_DIR / "data" / "knowledge",
]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _parse_front_matter_like_lines(block: str) -> Dict[str, str]:
    metadata = {}
    for line in block.splitlines()[:8]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        if key in {"category", "keywords"}:
            metadata[key] = value.strip()
    return metadata


def _load_section_chunks(file_path: Path) -> List[Dict]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    parts = re.split(r"(?=^##\s+CHUNK\s+\d+)", text, flags=re.MULTILINE)
    chunks = []

    for part in parts:
        part = part.strip()
        if not part.startswith("## CHUNK"):
            continue

        lines = part.splitlines()
        title = lines[0].replace("##", "", 1).strip()
        metadata = _parse_front_matter_like_lines(part)
        body_lines = [
            line for line in lines[1:]
            if not line.strip().lower().startswith(("category:", "keywords:"))
        ]

        chunks.append(
            {
                "source": file_path.name,
                "path": str(file_path),
                "chunk_id": len(chunks),
                "title": title,
                "category": metadata.get("category", "general"),
                "keywords": metadata.get("keywords", ""),
                "text": _clean_text("\n".join(body_lines)),
            }
        )

    return chunks


def load_knowledge_chunks(chunk_size: int = 900, overlap: int = 120) -> List[Dict]:
    chunks = []

    files = []
    seen_paths = set()

    for directory in KNOWLEDGE_DIRS:
        if not directory.exists():
            continue
        for file_path in list(directory.glob("*.md")) + list(directory.glob("*.txt")):
            resolved = str(file_path.resolve())
            if resolved not in seen_paths:
                files.append(file_path)
                seen_paths.add(resolved)

    for file_path in files:
        if file_path.name.startswith("project_stargazing_rag_chunks"):
            chunks.extend(_load_section_chunks(file_path))
            continue

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
                        "path": str(file_path),
                        "chunk_id": chunk_id,
                        "title": file_path.stem.replace("_", " ").title(),
                        "category": "source_material",
                        "keywords": "",
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


def _score_chunk(query_tokens: set, query_text: str, chunk: Dict, intent: Optional[str]) -> float:
    text = chunk.get("text", "")
    keywords = chunk.get("keywords", "")
    title = chunk.get("title", "")
    category = chunk.get("category", "")

    body_tokens = _tokenize(text)
    keyword_tokens = _tokenize(keywords)
    title_tokens = _tokenize(title)
    category_tokens = _tokenize(category)

    overlap = query_tokens.intersection(body_tokens)
    keyword_overlap = query_tokens.intersection(keyword_tokens)
    title_overlap = query_tokens.intersection(title_tokens)
    category_overlap = query_tokens.intersection(category_tokens)

    score = len(overlap) / max(len(query_tokens), 1)
    score += 0.22 * len(keyword_overlap)
    score += 0.18 * len(title_overlap)
    score += 0.12 * len(category_overlap)

    if intent and intent.lower() in category.lower():
        score += 0.55

    lowered = f"{title} {keywords} {text}".lower()
    for phrase in [
        "bortle scale",
        "light pollution",
        "cloud cover",
        "moon illumination",
        "dark adaptation",
        "travel plan",
        "observing site",
        "transparency",
        "seeing",
    ]:
        if phrase in query_text and phrase in lowered:
            score += 0.2

    return score


def retrieve_context(query: str, top_k: int = 4, intent: Optional[str] = None) -> List[Dict]:
    chunks = load_knowledge_chunks()

    if not chunks:
        return []

    query_tokens = _tokenize(query)
    query_text = query.lower()
    scored = []

    for chunk in chunks:
        score = _score_chunk(query_tokens, query_text, chunk, intent)

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
        title = item.get("title", "Untitled")
        category = item.get("category", "general")
        score = item.get("score", 0)
        blocks.append(
            f"[Source: {item['source']} | Chunk: {item['chunk_id']} | "
            f"Title: {title} | Category: {category} | Score: {score:.3f}]\n"
            f"{item['text']}"
        )

    return "\n\n---\n\n".join(blocks)
