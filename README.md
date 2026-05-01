# Stargazing Assistant

A live Streamlit-based stargazing recommendation system that combines weather forecasts, astronomy data, light pollution, and scoring logic to recommend the best observing windows.

## Features

- City preset, custom latitude/longitude input, and browser-based current location detection
- Live weather and astronomy data pipeline
- Astrospheric forecast with Open-Meteo fallback
- IPGeolocation astronomy data with fallback logic
- Bortle light pollution adjustment
- Stargazing score and recommendation labels
- Nearby stargazing trip planning with score-gated candidate search
- OpenStreetMap/Overpass public destination lookup with coordinate fallback
- Interactive map for current location, search radius, candidate points, and selected destination
- Hourly score chart
- Top observing windows
- Daily summary
- "Why this score?" model explanation panel
- Feature diagnostics
- Optional Sun/Moon sky path visualization
- Optional AI-generated recommendation explanation

## Demo Flow

1. Choose a city preset, custom coordinates, or browser current location.
2. Set the city lights/Bortle index and run the pipeline.
3. Review the best observing windows and the "Why this score?" explanation.
4. Open AI Insight -> Travel Plan.
5. If the best score is at least 70/100, search nearby locations.
6. Review the map, candidate table, selected public destination, and generated travel plan.

Travel planning still works without an OpenAI key. In that case, the app returns a deterministic trip plan based on the highest-scoring nearby candidate.

## Travel Plan Caveats

- OpenStreetMap/Overpass is used as a no-key public destination source.
- If Overpass is unavailable or no suitable destination is found, the app falls back to the highest-scoring coordinate.
- Nearby Bortle/light-pollution is estimated from the user-selected baseline, distance, and OSM outdoor-place signals; it is not a live light-pollution raster.
- Nearby astronomy values use the app's fallback approximation because moon/twilight timing changes only modestly across the short travel radius.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

Run lightweight smoke tests:

```bash
python3 tests/test_travel_plan_smoke.py
```
