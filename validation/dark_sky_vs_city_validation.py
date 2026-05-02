import json
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend import run_pipeline


VALIDATION_SITES = [
    # Dark-sky parks selected from the NPCA certified dark-sky parks article.
    {
        "name": "Arches National Park",
        "group": "NPCA certified dark-sky park",
        "lat": 38.7331,
        "lon": -109.5925,
        "timezone": "America/Denver",
        "bortle_index": 2,
    },
    {
        "name": "Big Bend National Park",
        "group": "NPCA certified dark-sky park",
        "lat": 29.1275,
        "lon": -103.2425,
        "timezone": "America/Chicago",
        "bortle_index": 1,
    },
    {
        "name": "Bryce Canyon National Park",
        "group": "NPCA certified dark-sky park",
        "lat": 37.5930,
        "lon": -112.1871,
        "timezone": "America/Denver",
        "bortle_index": 2,
    },
    {
        "name": "Canyonlands National Park",
        "group": "NPCA certified dark-sky park",
        "lat": 38.3269,
        "lon": -109.8783,
        "timezone": "America/Denver",
        "bortle_index": 2,
    },
    {
        "name": "Death Valley National Park",
        "group": "NPCA certified dark-sky park",
        "lat": 36.5054,
        "lon": -117.0794,
        "timezone": "America/Los_Angeles",
        "bortle_index": 1,
    },
    {
        "name": "Great Basin National Park",
        "group": "NPCA certified dark-sky park",
        "lat": 38.9833,
        "lon": -114.3000,
        "timezone": "America/Los_Angeles",
        "bortle_index": 1,
    },
    {
        "name": "Grand Canyon National Park",
        "group": "NPCA certified dark-sky park",
        "lat": 36.2679,
        "lon": -112.3535,
        "timezone": "America/Phoenix",
        "bortle_index": 2,
    },
    {
        "name": "Joshua Tree National Park",
        "group": "NPCA certified dark-sky park",
        "lat": 33.8734,
        "lon": -115.9010,
        "timezone": "America/Los_Angeles",
        "bortle_index": 3,
    },
    {
        "name": "Natural Bridges National Monument",
        "group": "NPCA certified dark-sky park",
        "lat": 37.6044,
        "lon": -110.0028,
        "timezone": "America/Denver",
        "bortle_index": 1,
    },
    {
        "name": "Voyageurs National Park",
        "group": "NPCA certified dark-sky park",
        "lat": 48.4833,
        "lon": -92.8389,
        "timezone": "America/Chicago",
        "bortle_index": 2,
    },
    # Large metros for contrast.
    {
        "name": "New York City",
        "group": "large metro",
        "lat": 40.7128,
        "lon": -74.0060,
        "timezone": "America/New_York",
        "bortle_index": 9,
    },
    {
        "name": "Los Angeles",
        "group": "large metro",
        "lat": 34.0522,
        "lon": -118.2437,
        "timezone": "America/Los_Angeles",
        "bortle_index": 9,
    },
    {
        "name": "Chicago",
        "group": "large metro",
        "lat": 41.8781,
        "lon": -87.6298,
        "timezone": "America/Chicago",
        "bortle_index": 9,
    },
    {
        "name": "Houston",
        "group": "large metro",
        "lat": 29.7604,
        "lon": -95.3698,
        "timezone": "America/Chicago",
        "bortle_index": 9,
    },
    {
        "name": "Phoenix",
        "group": "large metro",
        "lat": 33.4484,
        "lon": -112.0740,
        "timezone": "America/Phoenix",
        "bortle_index": 8,
    },
    {
        "name": "Miami",
        "group": "large metro",
        "lat": 25.7617,
        "lon": -80.1918,
        "timezone": "America/New_York",
        "bortle_index": 9,
    },
    {
        "name": "Seattle",
        "group": "large metro",
        "lat": 47.6062,
        "lon": -122.3321,
        "timezone": "America/Los_Angeles",
        "bortle_index": 8,
    },
    {
        "name": "Boston",
        "group": "large metro",
        "lat": 42.3601,
        "lon": -71.0589,
        "timezone": "America/New_York",
        "bortle_index": 9,
    },
]


def summarize_result(site: dict, result: dict) -> dict:
    score_df = result.get("score_df", pd.DataFrame())
    top_windows = result.get("top_windows", pd.DataFrame())
    night_df = score_df
    if score_df is not None and not score_df.empty and "is_dark_enough" in score_df.columns:
        night_df = score_df[score_df["is_dark_enough"].fillna(False).astype(bool)]

    best = top_windows.iloc[0].to_dict() if top_windows is not None and not top_windows.empty else {}

    def safe_stat(df, col, fn):
        if df is None or df.empty or col not in df.columns:
            return None
        return round(float(getattr(df[col], fn)()), 2)

    return {
        **site,
        "weather_source": result.get("weather_source"),
        "astronomy_source": result.get("astronomy_source"),
        "rows": int(len(score_df)) if score_df is not None else 0,
        "night_rows": int(len(night_df)) if night_df is not None else 0,
        "best_score": round(float(best.get("stargazing_score", 0)), 2) if best else None,
        "best_label": best.get("recommendation"),
        "best_time": best.get("time_label"),
        "median_score_all": safe_stat(score_df, "stargazing_score", "median"),
        "mean_score_all": safe_stat(score_df, "stargazing_score", "mean"),
        "median_score_night": safe_stat(night_df, "stargazing_score", "median"),
        "mean_score_night": safe_stat(night_df, "stargazing_score", "mean"),
        "peak_score_night": safe_stat(night_df, "stargazing_score", "max"),
        "good_or_better_night_share": (
            round(float((night_df["stargazing_score"] >= 70).mean()), 3)
            if night_df is not None and not night_df.empty and "stargazing_score" in night_df.columns
            else None
        ),
    }


def main():
    out_dir = Path("validation")
    out_dir.mkdir(exist_ok=True)

    rows = []
    errors = []

    for site in VALIDATION_SITES:
        print(f"Running {site['group']}: {site['name']}")
        try:
            result = run_pipeline(
                city_name=site["name"],
                lat=site["lat"],
                lon=site["lon"],
                timezone=site["timezone"],
                days=4,
                bortle_index=site["bortle_index"],
                include_tad=False,
                include_positions=False,
                include_llm=False,
            )
            rows.append(summarize_result(site, result))
        except Exception as exc:
            errors.append({**site, "error": str(exc)})
            print(f"  failed: {exc}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / "dark_sky_vs_city_validation.csv", index=False)

    summary = (
        result_df.groupby("group")
        .agg(
            sites=("name", "count"),
            median_best_score=("best_score", "median"),
            mean_best_score=("best_score", "mean"),
            median_night_score=("median_score_night", "median"),
            mean_night_score=("mean_score_night", "mean"),
            median_good_or_better_night_share=("good_or_better_night_share", "median"),
        )
        .round(3)
        .reset_index()
        if not result_df.empty
        else pd.DataFrame()
    )
    summary.to_csv(out_dir / "dark_sky_vs_city_validation_summary.csv", index=False)

    (out_dir / "dark_sky_vs_city_validation_errors.json").write_text(
        json.dumps(errors, indent=2),
        encoding="utf-8",
    )

    print("\nSummary")
    print(summary.to_string(index=False))
    if errors:
        print(f"\nErrors: {len(errors)}")


if __name__ == "__main__":
    main()
