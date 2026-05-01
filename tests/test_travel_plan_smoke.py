from datetime import datetime, timezone as dt_timezone, timedelta
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend


def _high_score_context(score=80):
    return {
        "city": "Test City",
        "lat": 40.7128,
        "lon": -74.0060,
        "timezone": "America/New_York",
        "weather_source": "test",
        "astronomy_source": "test",
        "top_windows": [
            {
                "stargazing_score": score,
                "time_label": "Tonight 10 PM",
                "recommendation": "Good",
            }
        ],
        "daily_summary": [],
    }


def _fake_weather(lat, lon, timezone="UTC", days=1):
    start = datetime.now(dt_timezone.utc).replace(minute=0, second=0, microsecond=0)
    rows = []
    for hour in range(24):
        rows.append(
            {
                "utcForecastHour": start + timedelta(hours=hour),
                "hourOffset": hour,
                "seeing_value": 2,
                "seeing_color": None,
                "transparency_value": 2,
                "transparency_color": None,
                "cloud_value": 8,
                "cloud_color": None,
                "temperature_value": 10,
                "temperature_color": None,
                "dewPoint_value": 2,
                "dewPoint_color": None,
                "wind_value": 5,
                "wind_color": None,
            }
        )
    return pd.DataFrame(rows), "Fake weather"


def test_below_threshold_gates_travel_plan():
    result = backend.generate_travel_plan_for_current_forecast(
        _high_score_context(score=55),
        bortle_index=5,
        days=1,
        radius_km=25,
        max_candidates=2,
    )
    assert result["search"]["status"] == "not_eligible"
    assert result["travel_plan"] is None


def test_osm_resolver_falls_back_on_network_failure():
    original_post = backend.requests.post
    backend._DESTINATION_CACHE.clear()

    def fail_post(*args, **kwargs):
        raise RuntimeError("network disabled")

    try:
        backend.requests.post = fail_post
        destinations = backend.resolve_nearby_public_destinations(40.0, -74.0, 5)
    finally:
        backend.requests.post = original_post

    assert destinations == []


def test_high_score_search_and_deterministic_plan_without_openai():
    original_weather = backend.fetch_weather_forecast_with_fallback
    original_resolver = backend.resolve_nearby_public_destinations
    original_key = backend.OPENAI_API_KEY
    backend._TRAVEL_SCORE_CACHE.clear()

    def fake_resolver(lat, lon, radius_km=10):
        return [
            {
                "name": "Mock Dark Park",
                "lat": lat,
                "lon": lon,
                "osm_type": "node",
                "osm_id": 1,
                "tags": {"leisure": "park", "name": "Mock Dark Park"},
                "distance_from_candidate_km": 0,
                "suitability_score": 20,
            }
        ]

    try:
        backend.fetch_weather_forecast_with_fallback = _fake_weather
        backend.resolve_nearby_public_destinations = fake_resolver
        backend.OPENAI_API_KEY = None
        result = backend.generate_travel_plan_for_current_forecast(
            _high_score_context(score=80),
            bortle_index=5,
            days=1,
            radius_km=25,
            max_candidates=4,
            score_threshold=70,
        )
    finally:
        backend.fetch_weather_forecast_with_fallback = original_weather
        backend.resolve_nearby_public_destinations = original_resolver
        backend.OPENAI_API_KEY = original_key

    assert result["search"]["status"] == "eligible"
    assert result["search"]["destination"]["name"] == "Mock Dark Park"
    assert "Mock Dark Park" in result["travel_plan"]


def test_high_score_search_falls_back_to_coordinates_without_osm_destination():
    original_weather = backend.fetch_weather_forecast_with_fallback
    original_resolver = backend.resolve_nearby_public_destinations
    original_key = backend.OPENAI_API_KEY
    backend._TRAVEL_SCORE_CACHE.clear()

    try:
        backend.fetch_weather_forecast_with_fallback = _fake_weather
        backend.resolve_nearby_public_destinations = lambda lat, lon, radius_km=10: []
        backend.OPENAI_API_KEY = None
        result = backend.generate_travel_plan_for_current_forecast(
            _high_score_context(score=80),
            bortle_index=5,
            days=1,
            radius_km=25,
            max_candidates=4,
            score_threshold=70,
        )
    finally:
        backend.fetch_weather_forecast_with_fallback = original_weather
        backend.resolve_nearby_public_destinations = original_resolver
        backend.OPENAI_API_KEY = original_key

    assert result["search"]["status"] == "eligible"
    assert result["search"]["destination"] is None
    assert "nearby candidate area" in result["travel_plan"].lower()


if __name__ == "__main__":
    test_below_threshold_gates_travel_plan()
    test_osm_resolver_falls_back_on_network_failure()
    test_high_score_search_and_deterministic_plan_without_openai()
    test_high_score_search_falls_back_to_coordinates_without_osm_destination()
    print("travel plan smoke tests passed")
