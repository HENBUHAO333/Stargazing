# Stargazing Assistant

A live Streamlit-based stargazing recommendation system that combines weather forecasts, astronomy data, light pollution, and scoring logic to recommend the best observing windows.

## Features

- City preset and custom latitude/longitude input
- Live weather and astronomy data pipeline
- Astrospheric forecast with Open-Meteo fallback
- IPGeolocation astronomy data with fallback logic
- Bortle light pollution adjustment
- Stargazing score and recommendation labels
- Hourly score chart
- Top observing windows
- Daily summary
- Feature diagnostics
- Optional Sun/Moon sky path visualization
- Optional AI-generated recommendation explanation

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate