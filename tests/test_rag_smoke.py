from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag_utils import load_knowledge_chunks, retrieve_context
from backend import _retrieve_stargazing_knowledge, generate_stargazing_travel_plan


def test_project_chunks_load():
    chunks = load_knowledge_chunks()
    project_chunks = [
        chunk for chunk in chunks
        if chunk["source"].startswith("project_stargazing_rag_chunks")
    ]

    assert len(project_chunks) >= 45
    assert all(chunk.get("category") for chunk in project_chunks)
    assert all(chunk.get("keywords") for chunk in project_chunks)


def test_intent_retrieval_prefers_curated_travel_chunks():
    results = retrieve_context(
        "where should I travel for stargazing and what should I bring",
        top_k=3,
        intent="travel_planning",
    )

    assert results
    assert results[0]["source"].startswith("project_stargazing_rag_chunks")
    assert "travel" in results[0]["category"].lower()


def test_rag_fallback_returns_sources():
    context, mode, sources = _retrieve_stargazing_knowledge(
        "why is my stargazing score low because of clouds and moonlight",
        top_k=4,
        intent="scoring_logic",
    )

    assert "retrieval" in mode.lower()
    assert "project_stargazing_rag_chunks" in context
    assert "CHUNK" in sources


def test_travel_plan_uses_rag_without_ai_text():
    context = {
        "city": "Demo City",
        "lat": 45.0,
        "lon": -121.0,
        "top_windows": [
            {
                "stargazing_score": 82.0,
                "time_label": "Tonight 10 PM",
                "recommendation": "Excellent",
            }
        ],
    }
    search_result = {
        "status": "eligible",
        "current_score": 82.0,
        "best_candidate": {
            "lat": 45.2,
            "lon": -121.4,
            "distance_km": 35.0,
            "best_score": 91.0,
            "estimated_bortle_index": 3.5,
            "time_label": "Tonight 11 PM",
            "recommendation": "Excellent",
            "cloud_value": 4,
            "transparency_value": 8,
            "seeing_value": 7,
        },
        "destination": {
            "name": "Demo Viewpoint",
            "lat": 45.21,
            "lon": -121.39,
            "distance_from_candidate_km": 1.2,
        },
    }

    plan = generate_stargazing_travel_plan(
        context=context,
        search_result=search_result,
        use_ai_text=False,
    )

    assert "Knowledge-Backed Planning Notes" in plan
    assert "Retrieved Sources" in plan
    assert "Demo Viewpoint" in plan


if __name__ == "__main__":
    test_project_chunks_load()
    test_intent_retrieval_prefers_curated_travel_chunks()
    test_rag_fallback_returns_sources()
    test_travel_plan_uses_rag_without_ai_text()
    print("rag smoke tests passed")
