# Stargazing Assistant 🌌

Stargazing Assistant is a live Streamlit-based data product that helps users find the best stargazing windows based on weather, astronomy, light pollution, and model-based scoring logic.

Users can select a preset city or enter custom latitude/longitude coordinates. The app then fetches live weather and astronomy data, calculates a stargazing score for each forecast hour, ranks the best observing windows, visualizes sky condition trends, and generates AI/RAG-enhanced explanations.

---

## Project Overview

The goal of this project is to make stargazing planning easier for non-expert users. Instead of requiring users to manually interpret cloud cover, transparency, seeing, moon phase, moon altitude, and light pollution, the app combines these factors into a structured recommendation system.

The app answers questions such as:

- When is the best time to go stargazing?
- Are the next few nights worth observing?
- What factors are limiting visibility?
- Should I focus on the Moon, bright stars, planets, or deep-sky objects?
- How does city light pollution affect the recommendation?

---

## Key Features

### Live Location-Based Forecast

Users can choose from preset North American cities or enter custom coordinates.

Supported preset cities include:

- New York City
- Hoboken
- Chicago
- Denver
- Los Angeles
- San Francisco
- Seattle
- Boston
- Toronto
- Vancouver

---

### Weather and Astronomy Data Pipeline

The backend fetches and combines multiple live data sources:

- **Astrospheric** as the primary astronomy-weather forecast source
- **Open-Meteo** as a global weather fallback
- **IPGeolocation Astronomy API** for moonrise, moonset, twilight, and moon phase data
- **Timeanddate API** for optional detailed Sun/Moon event and position data

The app includes fallback logic so that the frontend remains usable even if one external API fails or returns limited data.

---

### Stargazing Scoring Model

Each forecast hour receives a `stargazing_score` from 0 to 100.

The scoring model considers:

- Cloud cover
- Atmospheric transparency
- Seeing conditions
- Temperature and dew point spread
- Moon illumination
- Moon altitude
- Whether the moon is up
- Whether the sky is dark enough
- City light pollution / sky darkness level

Recommendation labels include:

- Excellent
- Good
- Marginal
- Poor
- No-Go

The score is deterministic and generated before any AI explanation is created.

---

### Visual Dashboard

The Streamlit frontend includes multiple product-style pages:

- **Dashboard**: summary cards, best window, and top recommendations
- **Forecast Timeline**: hourly score chart and date-hour heatmap
- **Best Windows**: ranked observing windows
- **Sky Conditions**: factor trends and diagnostic charts
- **Sky Path**: optional Sun/Moon position visualization
- **AI Insight**: natural-language explanation using LLM and RAG
- **Methodology**: data sources, scoring logic, and limitations
- **Raw Data**: debugging tables for transparency

---

### K-Means Cluster Explanation

The app also includes a clustering layer to help explain why different observing windows receive similar scores.

Forecast windows may be grouped into condition-based clusters such as hazy, moonlit, cloudy, or better observing windows. This makes the recommendation more interpretable for users.

---

### AI Insight with Lightweight RAG

The AI Insight page explains the model output in user-friendly language.

The current prototype includes a lightweight Retrieval-Augmented Generation layer. Instead of relying only on the LLM’s general knowledge, the app retrieves relevant passages from a local stargazing knowledge base.

The local knowledge base includes explanations of:

- City light pollution / Bortle scale
- Cloud cover
- Atmospheric transparency
- Seeing
- Moon illumination
- What objects are worth observing under different conditions
- Data source limitations

The RAG layer does **not** change the score. It only helps explain the fixed model output.

---

## Current Architecture

```text
User Input
    ↓
Location / Coordinates
    ↓
Live Weather + Astronomy APIs
    ↓
Master DataFrame
    ↓
Scoring Model
    ↓
Top Windows + Daily Summary + Diagnostics
    ↓
AI / RAG Explanation Layer
    ↓
Streamlit Frontend
