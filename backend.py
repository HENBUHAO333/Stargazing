import os
import json
import warnings
import math
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

from rag_utils import retrieve_context, format_retrieved_context

try:
    from rag_vector_utils import retrieve_vector_context, format_vector_context
except Exception as rag_vector_import_error:
    RAG_VECTOR_IMPORT_ERROR = str(rag_vector_import_error)
    retrieve_vector_context = None
    format_vector_context = None

load_dotenv()

# ============================================================
# STREAMLIT CLOUD + LOCAL SECRET LOADER
# ============================================================

try:
    import streamlit as st
except Exception:
    st = None


def get_secret(name):
    """
    Read secrets from local .env first, then Streamlit Cloud secrets.
    """
    value = os.getenv(name)
    if value:
        return value

    try:
        with open(".env", "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, raw_value = line.split("=", 1)
                if key.strip() == name:
                    return raw_value.strip().strip('"').strip("'")
    except Exception:
        pass

    if st is not None:
        try:
            return st.secrets.get(name)
        except Exception:
            return None

    return None


# ============================================================
# ENV KEYS
# ============================================================

TAD_ACCESS_KEY = get_secret("ASTRO_ACCESS_KEY")
TAD_SECRET_KEY = get_secret("ASTRO_SECRET_KEY")
IPGEOLOC_API_KEY = get_secret("IPGEOLOC_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
METEOBLUE_API_KEY = get_secret("METEOBLUE_API_KEY")

TRAVEL_PLAN_SCORE_THRESHOLD = 70.0
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"
_DESTINATION_CACHE = {}
_TRAVEL_SCORE_CACHE = {}


# ============================================================
# ASTROSPHERIC ENDPOINT
# ============================================================

ASTROSPHERIC_URL = (
    "https://forecast-isolate-fyakcsbwd5fkfwaw.westus-01.azurewebsites.net/"
    "api/GetForecast?code=fJoPn63j_auOYhEG04DpQKVmCK59UK0D4PU_sPDcQ_JJAzFuH7Hijw=="
)


# ============================================================
# CITY PRESETS — NORTH AMERICA FOCUSED
# ============================================================

CITY_PRESETS = {
    "New York City": (40.7128, -74.0060, "America/New_York"),
    "Hoboken": (40.7440, -74.0324, "America/New_York"),
    "Chicago": (41.8781, -87.6298, "America/Chicago"),
    "Denver": (39.7392, -104.9903, "America/Denver"),
    "Los Angeles": (34.0522, -118.2437, "America/Los_Angeles"),
    "San Francisco": (37.7749, -122.4194, "America/Los_Angeles"),
    "Seattle": (47.6062, -122.3321, "America/Los_Angeles"),
    "Boston": (42.3601, -71.0589, "America/New_York"),
    "Toronto": (43.6532, -79.3832, "America/Toronto"),
    "Vancouver": (49.2827, -123.1207, "America/Vancouver"),
}


# ============================================================
# LOCATION
# ============================================================

def get_city_coordinates(city_name: str) -> Tuple[float, float, str]:
    city_name = city_name.strip()

    if city_name in CITY_PRESETS:
        return CITY_PRESETS[city_name]

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city_name,
        "count": 1,
        "language": "en",
        "format": "json",
    }

    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()

    data = resp.json()
    results = data.get("results", [])

    if not results:
        raise ValueError(f"Could not find coordinates for city: {city_name}")

    result = results[0]

    lat = float(result["latitude"])
    lon = float(result["longitude"])
    tz = result.get("timezone", "UTC")

    return lat, lon, tz


# ============================================================
# ASTROSPHERIC WEATHER FORECAST
# ============================================================

def extract_metric_df(data: Dict, metric_name: str) -> pd.DataFrame:
    metric_data = data.get(metric_name)

    if metric_data is None:
        return pd.DataFrame(
            columns=[
                "utcForecastHour",
                "hourOffset",
                f"{metric_name}_value",
                f"{metric_name}_color",
            ]
        )

    rows = []

    for item in metric_data:
        row = {
            "utcForecastHour": item.get("utcForecastHour"),
            "hourOffset": item.get("hourOffset"),
        }

        value = item.get("value")

        if isinstance(value, dict):
            row[f"{metric_name}_value"] = value.get("actualValue")
            row[f"{metric_name}_color"] = value.get("valueColor")
        else:
            row[f"{metric_name}_value"] = value
            row[f"{metric_name}_color"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def fetch_astrospheric_forecast(lat: float, lon: float) -> pd.DataFrame:
    payload = {
        "Latitude": lat,
        "Longitude": lon,
    }

    resp = requests.post(ASTROSPHERIC_URL, json=payload, timeout=30)
    resp.raise_for_status()

    data = resp.json()

    seeing_df = extract_metric_df(data, "seeing")
    transparency_df = extract_metric_df(data, "transparency")
    cloud_df = extract_metric_df(data, "cloud")
    temperature_df = extract_metric_df(data, "temperature")
    dew_df = extract_metric_df(data, "dewPoint")
    wind_df = extract_metric_df(data, "wind")

    weather_df = (
        seeing_df
        .merge(transparency_df, on=["utcForecastHour", "hourOffset"], how="outer")
        .merge(cloud_df, on=["utcForecastHour", "hourOffset"], how="outer")
        .merge(temperature_df, on=["utcForecastHour", "hourOffset"], how="outer")
        .merge(dew_df, on=["utcForecastHour", "hourOffset"], how="outer")
        .merge(wind_df, on=["utcForecastHour", "hourOffset"], how="outer")
    )

    if weather_df.empty:
        return weather_df

    weather_df["utcForecastHour"] = pd.to_datetime(
        weather_df["utcForecastHour"],
        utc=True,
        errors="coerce",
    )

    weather_df = weather_df.dropna(subset=["utcForecastHour"])

    return weather_df


# ============================================================
# OPEN-METEO FALLBACK WEATHER
# ============================================================

def fetch_open_meteo_forecast(
    lat: float,
    lon: float,
    timezone: str = "UTC",
    days: int = 4,
) -> pd.DataFrame:
    """
    Global fallback.

    Important:
    Open-Meteo does not provide astronomy-specific seeing/transparency.
    We convert it into the same schema used by the notebook:
    - cloud_value: 0-100 cloud cover
    - transparency_value: synthetic 1-5 scale, lower = better
    - seeing_value: neutral fallback value
    """

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "cloud_cover",
            "temperature_2m",
            "dew_point_2m",
            "wind_speed_10m",
            "visibility",
        ],
        "forecast_days": int(days),
        "timezone": "UTC",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    hourly = data.get("hourly", {})

    if not hourly or "time" not in hourly:
        raise ValueError("Open-Meteo returned no hourly forecast data.")

    df = pd.DataFrame(hourly)

    df["utcForecastHour"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["hourOffset"] = range(len(df))

    df["cloud_value"] = pd.to_numeric(df.get("cloud_cover"), errors="coerce")

    visibility = pd.to_numeric(df.get("visibility"), errors="coerce")
    visibility_km = visibility / 1000

    # Convert visibility to notebook-style 1-good / 5-bad scale.
    df["transparency_value"] = 5 - (visibility_km.clip(0, 20) / 20 * 4)
    df["transparency_value"] = df["transparency_value"].clip(1, 5)

    # Open-Meteo has no astronomical seeing.
    # 3 is a neutral midpoint in the notebook's 1-5 logic.
    df["seeing_value"] = 3.0

    df["temperature_value"] = pd.to_numeric(df.get("temperature_2m"), errors="coerce")
    df["dewPoint_value"] = pd.to_numeric(df.get("dew_point_2m"), errors="coerce")
    df["wind_value"] = pd.to_numeric(df.get("wind_speed_10m"), errors="coerce")

    df["seeing_color"] = None
    df["transparency_color"] = None
    df["cloud_color"] = None
    df["temperature_color"] = None
    df["dewPoint_color"] = None
    df["wind_color"] = None

    keep_cols = [
        "utcForecastHour",
        "hourOffset",
        "seeing_value",
        "seeing_color",
        "transparency_value",
        "transparency_color",
        "cloud_value",
        "cloud_color",
        "temperature_value",
        "temperature_color",
        "dewPoint_value",
        "dewPoint_color",
        "wind_value",
        "wind_color",
    ]

    return df[keep_cols]


def fetch_open_meteo_air_quality(
    lat: float,
    lon: float,
    timezone: str = "UTC",
    days: int = 4,
) -> pd.DataFrame:
    """
    Free no-key haze validation layer from Open-Meteo Air Quality.

    Aerosol optical depth is a closer proxy for sky transparency/haze than
    surface visibility alone. PM2.5 is not an astronomy variable, but it helps
    flag smoke/pollution nights where nominal weather visibility can be
    misleading.
    """

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["aerosol_optical_depth", "pm2_5", "dust"],
        "forecast_days": int(days),
        "timezone": "UTC",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    hourly = resp.json().get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df["utcForecastHour"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["utcForecastHour"])

    for col in ["aerosol_optical_depth", "pm2_5", "dust"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["utcForecastHour", "aerosol_optical_depth", "pm2_5", "dust"]]


def _estimate_dewpoint_celsius(temp_c: pd.Series, relative_humidity: pd.Series) -> pd.Series:
    temp_c = pd.to_numeric(temp_c, errors="coerce")
    rh = pd.to_numeric(relative_humidity, errors="coerce").clip(1, 100)
    a = 17.625
    b = 243.04
    gamma = np.log(rh / 100.0) + (a * temp_c) / (b + temp_c)
    return (b * gamma) / (a - gamma)


def _meteoblue_seeing_proxy(windspeed_ms: pd.Series, lifted_index: pd.Series) -> pd.Series:
    """
    Conservative 1-good / 5-bad seeing proxy.

    The queried Meteoblue package gives stability and wind context, not the
    dedicated Astronomy Seeing index. Keep this as a proxy and expose that in
    validation metadata.
    """
    wind = pd.to_numeric(windspeed_ms, errors="coerce")
    lifted = pd.to_numeric(lifted_index, errors="coerce")

    wind_quality = 1 - ((wind.clip(1, 12) - 1) / 11)
    stability_quality = ((lifted.clip(-2, 8) + 2) / 10)
    quality = (0.60 * wind_quality + 0.40 * stability_quality).fillna(0.50)
    return (5 - 4 * quality).clip(1, 5)


def fetch_meteoblue_forecast(
    lat: float,
    lon: float,
    timezone: str = "UTC",
    days: int = 4,
) -> pd.DataFrame:
    """
    Realtime Meteoblue validation/enrichment feed.

    Uses packages available in the Forecast API:
    - basic-1h: temperature, humidity, wind, visibility
    - clouds-1h: total/low/mid/high clouds
    - air-1h: stability indices useful as a weak seeing proxy
    """
    if not METEOBLUE_API_KEY:
        return pd.DataFrame()

    url = "https://my.meteoblue.com/packages/basic-1h_clouds-1h_air-1h"
    params = {
        "lat": lat,
        "lon": lon,
        "apikey": METEOBLUE_API_KEY,
        "forecast_days": int(days),
        "tz": timezone or "UTC",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("error_message"):
        raise ValueError(f"Meteoblue error: {payload.get('error_message')}")

    data = payload.get("data_1h", {})
    if not data or "time" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["utcForecastHour"] = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce",
    )
    df = df.dropna(subset=["utcForecastHour"])

    if df.empty:
        return pd.DataFrame()

    for col in [
        "totalcloudcover",
        "lowclouds",
        "midclouds",
        "highclouds",
        "visibility",
        "relativehumidity",
        "temperature",
        "windspeed",
        "liftedindex",
        "k_index",
        "fog_probability",
    ]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    visibility_km = df["visibility"] / 1000.0
    df["meteoblue_transparency_value"] = (
        5 - (visibility_km.clip(0, 25) / 25 * 4)
    ).clip(1, 5)
    df["meteoblue_seeing_proxy_value"] = _meteoblue_seeing_proxy(
        df["windspeed"],
        df["liftedindex"],
    )
    df["meteoblue_dewpoint_value"] = _estimate_dewpoint_celsius(
        df["temperature"],
        df["relativehumidity"],
    )

    out = pd.DataFrame(
        {
            "utcForecastHour": df["utcForecastHour"],
            "meteoblue_cloud_value": df["totalcloudcover"],
            "meteoblue_low_clouds": df["lowclouds"],
            "meteoblue_mid_clouds": df["midclouds"],
            "meteoblue_high_clouds": df["highclouds"],
            "meteoblue_visibility_m": df["visibility"],
            "meteoblue_transparency_value": df["meteoblue_transparency_value"],
            "meteoblue_seeing_proxy_value": df["meteoblue_seeing_proxy_value"],
            "meteoblue_relativehumidity": df["relativehumidity"],
            "meteoblue_temperature_value": df["temperature"],
            "meteoblue_dewpoint_value": df["meteoblue_dewpoint_value"],
            "meteoblue_wind_value": df["windspeed"],
            "meteoblue_liftedindex": df["liftedindex"],
            "meteoblue_k_index": df["k_index"],
            "meteoblue_fog_probability": df["fog_probability"],
        }
    )

    return out


def build_weather_from_meteoblue(meteoblue_df: pd.DataFrame) -> pd.DataFrame:
    if meteoblue_df is None or meteoblue_df.empty:
        return pd.DataFrame()

    df = meteoblue_df.copy()
    df["hourOffset"] = range(len(df))
    df["cloud_value"] = df["meteoblue_cloud_value"]
    df["transparency_value"] = df["meteoblue_transparency_value"]
    df["seeing_value"] = df["meteoblue_seeing_proxy_value"]
    df["temperature_value"] = df["meteoblue_temperature_value"]
    df["dewPoint_value"] = df["meteoblue_dewpoint_value"]
    df["wind_value"] = df["meteoblue_wind_value"]

    for col in [
        "seeing_color",
        "transparency_color",
        "cloud_color",
        "temperature_color",
        "dewPoint_color",
        "wind_color",
    ]:
        df[col] = None

    keep_cols = [
        "utcForecastHour",
        "hourOffset",
        "seeing_value",
        "seeing_color",
        "transparency_value",
        "transparency_color",
        "cloud_value",
        "cloud_color",
        "temperature_value",
        "temperature_color",
        "dewPoint_value",
        "dewPoint_color",
        "wind_value",
        "wind_color",
    ] + [c for c in df.columns if c.startswith("meteoblue_")]

    return df[keep_cols]


def enrich_weather_with_meteoblue(
    weather_df: pd.DataFrame,
    lat: float,
    lon: float,
    timezone: str,
    days: int,
) -> Tuple[pd.DataFrame, bool]:
    if weather_df is None or weather_df.empty:
        return weather_df, False

    try:
        meteoblue_df = fetch_meteoblue_forecast(
            lat=lat,
            lon=lon,
            timezone=timezone,
            days=days,
        )
    except Exception as e:
        print(f"Meteoblue enrichment failed. Continuing without Meteoblue. Error: {e}")
        return weather_df, False

    if meteoblue_df is None or meteoblue_df.empty:
        return weather_df, False

    out = weather_df.copy()
    out["utcForecastHour"] = pd.to_datetime(
        out["utcForecastHour"],
        utc=True,
        errors="coerce",
    ).dt.floor("h")
    mb = meteoblue_df.copy()
    mb["utcForecastHour"] = pd.to_datetime(
        mb["utcForecastHour"],
        utc=True,
        errors="coerce",
    ).dt.floor("h")
    out = out.merge(mb, on="utcForecastHour", how="left")

    if "meteoblue_cloud_value" in out.columns:
        primary_cloud = pd.to_numeric(out["cloud_value"], errors="coerce")
        mb_cloud = pd.to_numeric(out["meteoblue_cloud_value"], errors="coerce")
        out["cloud_model_delta"] = (primary_cloud - mb_cloud).abs()
        out["cloud_value"] = (0.60 * primary_cloud + 0.40 * mb_cloud).combine_first(primary_cloud)

    if "meteoblue_transparency_value" in out.columns:
        primary_trans = pd.to_numeric(out["transparency_value"], errors="coerce")
        mb_trans = pd.to_numeric(out["meteoblue_transparency_value"], errors="coerce")
        out["transparency_model_delta"] = (primary_trans - mb_trans).abs()
        out["transparency_value"] = (
            0.65 * primary_trans + 0.35 * mb_trans
        ).combine_first(primary_trans)

    if "meteoblue_seeing_proxy_value" in out.columns:
        out["seeing_value"] = pd.to_numeric(
            out["seeing_value"],
            errors="coerce",
        ).fillna(out["meteoblue_seeing_proxy_value"])

    for target, source in [
        ("temperature_value", "meteoblue_temperature_value"),
        ("dewPoint_value", "meteoblue_dewpoint_value"),
        ("wind_value", "meteoblue_wind_value"),
    ]:
        if source in out.columns:
            out[target] = pd.to_numeric(out[target], errors="coerce").fillna(out[source])

    out["meteoblue_validation_available"] = out["meteoblue_cloud_value"].notna()
    return out, True


def enrich_weather_with_air_quality(
    weather_df: pd.DataFrame,
    lat: float,
    lon: float,
    timezone: str,
    days: int,
) -> Tuple[pd.DataFrame, bool]:
    if weather_df is None or weather_df.empty:
        return weather_df, False

    try:
        aq_df = fetch_open_meteo_air_quality(
            lat=lat,
            lon=lon,
            timezone=timezone,
            days=days,
        )
    except Exception as e:
        print(f"Open-Meteo air quality enrichment failed. Error: {e}")
        return weather_df, False

    if aq_df is None or aq_df.empty:
        return weather_df, False

    out = weather_df.copy()
    out["utcForecastHour"] = pd.to_datetime(
        out["utcForecastHour"],
        utc=True,
        errors="coerce",
    )
    out = out.merge(aq_df, on="utcForecastHour", how="left")
    return out, True


def fetch_weather_forecast_with_fallback(
    lat: float,
    lon: float,
    timezone: str = "UTC",
    days: int = 4,
) -> Tuple[pd.DataFrame, str]:
    source = "Open-Meteo fallback"

    try:
        weather_df = fetch_astrospheric_forecast(lat, lon)

        if weather_df is not None and not weather_df.empty:
            source = "Astrospheric"
            weather_df, has_meteoblue = enrich_weather_with_meteoblue(
                weather_df=weather_df,
                lat=lat,
                lon=lon,
                timezone=timezone,
                days=days,
            )
            if has_meteoblue:
                source = f"{source} + Meteoblue Validation"
            enriched_df, has_air_quality = enrich_weather_with_air_quality(
                weather_df=weather_df,
                lat=lat,
                lon=lon,
                timezone=timezone,
                days=days,
            )
            if has_air_quality:
                source = f"{source} + Open-Meteo Air Quality"
            return enriched_df, source

        print("Astrospheric returned empty data. Falling back to Open-Meteo.")

    except Exception as e:
        print(f"Astrospheric failed. Falling back to Open-Meteo. Error: {e}")

    try:
        meteoblue_df = fetch_meteoblue_forecast(
            lat=lat,
            lon=lon,
            timezone=timezone,
            days=days,
        )
        if meteoblue_df is not None and not meteoblue_df.empty:
            weather_df = build_weather_from_meteoblue(meteoblue_df)
            source = "Meteoblue realtime"
            weather_df, has_air_quality = enrich_weather_with_air_quality(
                weather_df=weather_df,
                lat=lat,
                lon=lon,
                timezone=timezone,
                days=days,
            )
            if has_air_quality:
                source = f"{source} + Open-Meteo Air Quality"
            return weather_df, source
    except Exception as e:
        print(f"Meteoblue failed. Falling back to Open-Meteo. Error: {e}")

    weather_df = fetch_open_meteo_forecast(
        lat=lat,
        lon=lon,
        timezone=timezone,
        days=days,
    )

    weather_df, has_meteoblue = enrich_weather_with_meteoblue(
        weather_df=weather_df,
        lat=lat,
        lon=lon,
        timezone=timezone,
        days=days,
    )
    if has_meteoblue:
        source = f"{source} + Meteoblue Validation"

    weather_df, has_air_quality = enrich_weather_with_air_quality(
        weather_df=weather_df,
        lat=lat,
        lon=lon,
        timezone=timezone,
        days=days,
    )
    if has_air_quality:
        source = f"{source} + Air Quality"

    return weather_df, source


# ============================================================
# IPGEOLOCATION ASTRONOMY
# ============================================================

def fetch_ipgeo_astronomy(
    lat: float,
    lon: float,
    start_date: Optional[datetime] = None,
    days: int = 4,
) -> pd.DataFrame:
    if IPGEOLOC_API_KEY is None:
        raise ValueError("Missing IPGEOLOC_API_KEY in .env")

    if start_date is None:
        start_date = datetime.now()

    end_date = start_date + timedelta(days=days - 1)

    url = "https://api.ipgeolocation.io/v3/astronomy/timeSeries"

    def request_timeseries(start_dt: datetime, end_dt: datetime) -> list:
        params = {
            "apiKey": IPGEOLOC_API_KEY,
            "location": f"{lat},{lon}",
            "dateStart": start_dt.strftime("%Y-%m-%d"),
            "dateEnd": end_dt.strftime("%Y-%m-%d"),
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("astronomy", [])

    try:
        astronomy_data = request_timeseries(start_date, end_date)
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else None

        # Some plans/locations reject multi-day timeSeries ranges with HTTP 400.
        # Retry as one-day windows to maximize successful coverage.
        if status == 400 and days > 1:
            astronomy_data = []
            for i in range(days):
                one_day = start_date + timedelta(days=i)
                try:
                    astronomy_data.extend(request_timeseries(one_day, one_day))
                except requests.HTTPError as day_err:
                    day_status = (
                        day_err.response.status_code
                        if day_err.response is not None
                        else None
                    )
                    if day_status != 400:
                        raise
            if not astronomy_data:
                raise ValueError(
                    "IPGeolocation returned no astronomy data for requested dates."
                ) from e
        else:
            raise

    if not astronomy_data:
        raise ValueError("IPGeolocation returned no astronomy data.")

    ip_geo_df = pd.json_normalize(astronomy_data)

    if "date" not in ip_geo_df.columns:
        raise ValueError("IPGeolocation astronomy data has no date column.")

    ip_geo_df["date"] = pd.to_datetime(ip_geo_df["date"]).dt.date

    return ip_geo_df.sort_values("date").reset_index(drop=True)


def build_fallback_astronomy_df(
    start_date: Optional[datetime] = None,
    days: int = 4,
) -> pd.DataFrame:
    """
    Fallback astronomy.

    This keeps the app alive if IPGeolocation fails.
    It is intentionally marked as fallback and should not be treated as precise.
    """

    if start_date is None:
        start_date = datetime.now()

    rows = []

    for i in range(days):
        d = (start_date + timedelta(days=i)).date()

        rows.append(
            {
                "date": d,
                "moon_phase": "Unknown",
                "morning_astro_begin_dt": pd.to_datetime(f"{d} 05:00:00"),
                "evening_astro_end_dt": pd.to_datetime(f"{d} 20:00:00"),
                "moonrise_dt": pd.to_datetime(f"{d} 18:00:00"),
                "moonset_dt": pd.to_datetime(f"{d} 06:00:00"),
            }
        )

    return pd.DataFrame(rows)


def fetch_astronomy_with_fallback(
    lat: float,
    lon: float,
    start_date: Optional[datetime] = None,
    days: int = 4,
) -> Tuple[pd.DataFrame, str]:
    try:
        ip_geo_df = fetch_ipgeo_astronomy(
            lat=lat,
            lon=lon,
            start_date=start_date,
            days=days,
        )

        if ip_geo_df is not None and not ip_geo_df.empty:
            return ip_geo_df, "IPGeolocation"

    except Exception as e:
        print(f"IPGeolocation astronomy fallback active: {e}")

    fallback_df = build_fallback_astronomy_df(
        start_date=start_date,
        days=days,
    )

    return fallback_df, "Simple astronomy fallback"


def _combine_local_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        date_series.astype(str) + " " + time_series.astype(str),
        errors="coerce",
    )


# ============================================================
# TIMEANDDATE OPTIONAL EVENTS / POSITIONS
# ============================================================

def _load_libtad():
    from libtad import AstronomyService, AstrodataService
    from libtad.datatypes.places import Coordinates, LocationId
    from libtad.datatypes.time import TADDateTime
    from libtad.datatypes.astro import (
        AstronomyEventClass,
        AstronomyObjectType,
    )

    return {
        "AstronomyService": AstronomyService,
        "AstrodataService": AstrodataService,
        "Coordinates": Coordinates,
        "LocationId": LocationId,
        "TADDateTime": TADDateTime,
        "AstronomyEventClass": AstronomyEventClass,
        "AstronomyObjectType": AstronomyObjectType,
    }


def fetch_tad_events(
    lat: float,
    lon: float,
    timezone: str,
    start_date: Optional[datetime] = None,
    days: int = 4,
) -> pd.DataFrame:
    """
    Optional detailed Sun/Moon event data from Timeanddate.

    Important fix:
    UTC times are converted to the selected local timezone before becoming naive.
    """

    if not TAD_ACCESS_KEY or not TAD_SECRET_KEY:
        raise ValueError("Missing ASTRO_ACCESS_KEY or ASTRO_SECRET_KEY in .env")

    lib = _load_libtad()

    AstronomyService = lib["AstronomyService"]
    Coordinates = lib["Coordinates"]
    LocationId = lib["LocationId"]
    TADDateTime = lib["TADDateTime"]
    AstronomyEventClass = lib["AstronomyEventClass"]
    AstronomyObjectType = lib["AstronomyObjectType"]

    if start_date is None:
        start_date = datetime.now()

    coordinates = Coordinates(lat, lon)
    place = LocationId(coordinates)

    tad_start = TADDateTime(start_date.year, start_date.month, start_date.day)
    calculated_end = start_date + timedelta(days=days - 1)
    tad_end = TADDateTime(
        calculated_end.year,
        calculated_end.month,
        calculated_end.day,
    )

    service = AstronomyService(TAD_ACCESS_KEY, TAD_SECRET_KEY)
    service.include_utctime = True
    service.include_coordinates = True

    service.types = (
        AstronomyEventClass.Meridian
        | AstronomyEventClass.Phase
        | AstronomyEventClass.NauticalTwilight
        | AstronomyEventClass.CivilTwilight
    )

    def fetch_one(object_type, label: str) -> pd.DataFrame:
        locations = service.get_astronomical_info(
            object_type,
            place,
            tad_start,
            tad_end,
        )

        rows = []

        for location in locations:
            geo = location.geography

            for obj in location.objects:
                for day in obj.days:
                    date_str = (
                        f"{day.date.year}-{day.date.month:02d}-{day.date.day:02d}"
                    )

                    moon_phase = None

                    if label == "Moon" and day.moonphase:
                        moon_phase = day.moonphase.name

                    for event in day.events:
                        event_type = event.type.name if event.type else ""

                        utc = event.utctime
                        dt = utc.date_time if utc else None

                        utc_str = (
                            f"{dt.year}-{dt.month:02d}-{dt.day:02d} "
                            f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"
                            if dt
                            else None
                        )

                        row = {
                            "Object": label,
                            "Date": date_str,
                            "Event": event_type,
                            "UTC Time": utc_str,
                            "Location": (
                                f"{geo.name}, {geo.country.name}" if geo else None
                            ),
                        }

                        if event.altitude is not None:
                            row["Altitude (°)"] = round(float(event.altitude), 2)

                        if event.distance is not None:
                            row["Distance (km)"] = int(float(event.distance))

                        if event.illuminated is not None:
                            row["Illuminated (%)"] = round(float(event.illuminated), 1)

                        if event.posangle is not None:
                            row["Posangle"] = round(float(event.posangle), 1)

                        if moon_phase:
                            row["Moon Phase"] = moon_phase

                        rows.append(row)

        return pd.DataFrame(rows)

    df_sun = fetch_one(AstronomyObjectType.Sun, "Sun")
    df_moon = fetch_one(AstronomyObjectType.Moon, "Moon")

    event_df = pd.concat([df_sun, df_moon], ignore_index=True)

    if event_df.empty:
        return event_df

    event_df["date"] = pd.to_datetime(event_df["Date"]).dt.date
    event_df["utc_dt"] = pd.to_datetime(event_df["UTC Time"], utc=True, errors="coerce")

    event_df["local_dt_naive"] = (
        event_df["utc_dt"]
        .dt.tz_convert(timezone)
        .dt.tz_localize(None)
    )

    return event_df


def fetch_tad_positions(
    lat: float,
    lon: float,
    start_date: Optional[datetime] = None,
    timezone: str = "UTC",
    days: int = 2,
) -> pd.DataFrame:
    def _estimated_position_fallback() -> pd.DataFrame:
        tz = ZoneInfo(timezone) if timezone else ZoneInfo("UTC")
        now_local = datetime.now(tz)
        anchor = start_date or now_local
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=tz)
        else:
            anchor = anchor.astimezone(tz)

        lat_factor = max(0.35, math.cos(math.radians(abs(lat))))
        rows = []
        for day_offset in range(max(int(days), 1)):
            d = (anchor + timedelta(days=day_offset)).date()
            for hour in range(24):
                dt_local = datetime(d.year, d.month, d.day, hour, 0, 0, tzinfo=tz)
                dt_utc = dt_local.astimezone(ZoneInfo("UTC"))

                sun_phase = 2 * math.pi * ((hour - 12) / 24.0)
                sun_alt = 65 * math.sin(sun_phase) * lat_factor
                sun_az = (hour / 24.0 * 360.0 + 180.0) % 360.0

                moon_hour = (hour + 6) % 24
                moon_phase = 2 * math.pi * ((moon_hour - 12) / 24.0)
                moon_alt = 55 * math.sin(moon_phase) * lat_factor
                moon_az = ((hour + 6) / 24.0 * 360.0 + 120.0) % 360.0

                rows.append(
                    {
                        "Object": "Sun",
                        "UTC Time": dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
                        "Hour": hour,
                        "Altitude (°)": round(float(sun_alt), 2),
                        "Azimuth (°)": round(float(sun_az), 2),
                        "Data Source": "Estimated fallback",
                    }
                )
                rows.append(
                    {
                        "Object": "Moon",
                        "UTC Time": dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
                        "Hour": hour,
                        "Altitude (°)": round(float(moon_alt), 2),
                        "Azimuth (°)": round(float(moon_az), 2),
                        "Moon Phase": "Estimated",
                        "Data Source": "Estimated fallback",
                    }
                )

        return pd.DataFrame(rows)

    if not TAD_ACCESS_KEY or not TAD_SECRET_KEY:
        return _estimated_position_fallback()

    lib = _load_libtad()

    AstrodataService = lib["AstrodataService"]
    Coordinates = lib["Coordinates"]
    LocationId = lib["LocationId"]
    TADDateTime = lib["TADDateTime"]
    AstronomyObjectType = lib["AstronomyObjectType"]

    if start_date is None:
        start_date = datetime.now()

    coordinates = Coordinates(lat, lon)
    place = LocationId(coordinates)

    service = AstrodataService(TAD_ACCESS_KEY, TAD_SECRET_KEY)
    service.include_utctime = True
    service.include_isotime = False
    service.is_localtime = False

    intervals = []
    for day_offset in range(max(int(days), 1)):
        day_ref = start_date + timedelta(days=day_offset)
        intervals.extend(
            [
                TADDateTime(day_ref.year, day_ref.month, day_ref.day, hour, 0, 0)
                for hour in range(24)
            ]
        )

    def fetch_one(object_type, label: str) -> list:
        locations = service.get_astrodata(object_type, place, intervals)

        rows = []

        for location in locations:
            for obj in location.objects:
                for current in obj.result:
                    utc = current.utctime
                    dt = utc.date_time if utc else None

                    row = {
                        "Object": label,
                        "UTC Time": (
                            f"{dt.year}-{dt.month:02d}-{dt.day:02d} "
                            f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"
                            if dt
                            else None
                        ),
                        "Hour": dt.hour if dt else None,
                    }

                    if current.altitude is not None:
                        row["Altitude (°)"] = round(float(current.altitude), 2)

                    if current.azimuth is not None:
                        row["Azimuth (°)"] = round(float(current.azimuth), 2)

                    if current.distance is not None:
                        row["Distance (km)"] = int(float(current.distance))

                    if current.illuminated is not None:
                        row["Illuminated (%)"] = round(float(current.illuminated), 1)

                    if current.posangle is not None:
                        row["Posangle"] = round(float(current.posangle), 1)

                    if current.moonphase is not None:
                        row["Moon Phase"] = current.moonphase.name

                    rows.append(row)

        return rows

    sun_rows = fetch_one(AstronomyObjectType.Sun, "Sun")
    moon_rows = fetch_one(AstronomyObjectType.Moon, "Moon")

    out = pd.DataFrame(sun_rows + moon_rows)

    if out.empty:
        return _estimated_position_fallback()

    # Keep only rows that actually contain usable sky-path coordinates.
    for c in ["Altitude (°)", "Azimuth (°)"]:
        if c not in out.columns:
            out[c] = np.nan

    out = out.dropna(subset=["Altitude (°)", "Azimuth (°)"], how="all")

    out = out.reset_index(drop=True)
    if out.empty:
        return _estimated_position_fallback()
    return out


# ============================================================
# BUILD LIVE MASTER DF
# ============================================================

def _prepare_astronomy_df(ip_geo_df: pd.DataFrame) -> pd.DataFrame:
    df = ip_geo_df.copy()

    if "date" not in df.columns:
        raise ValueError("Astronomy dataframe has no date column.")

    df["date"] = pd.to_datetime(df["date"]).dt.date

    ready_cols = [
        "morning_astro_begin_dt",
        "evening_astro_end_dt",
        "moonrise_dt",
        "moonset_dt",
    ]

    if all(c in df.columns for c in ready_cols):
        for c in ready_cols:
            df[c] = pd.to_datetime(df[c], errors="coerce")

        if "moon_phase" not in df.columns:
            df["moon_phase"] = "Unknown"

        return df[
            [
                "date",
                "moon_phase",
                "morning_astro_begin_dt",
                "evening_astro_end_dt",
                "moonrise_dt",
                "moonset_dt",
            ]
        ]

    df["morning_astro_begin_dt"] = _combine_local_datetime(
        df["date"],
        df.get("morning.astronomical_twilight_begin"),
    )

    df["evening_astro_end_dt"] = _combine_local_datetime(
        df["date"],
        df.get("evening.astronomical_twilight_end"),
    )

    df["moonrise_dt"] = _combine_local_datetime(
        df["date"],
        df.get("moonrise"),
    )

    df["moonset_dt"] = _combine_local_datetime(
        df["date"],
        df.get("moonset"),
    )

    if "moon_phase" not in df.columns:
        df["moon_phase"] = "Unknown"

    return df[
        [
            "date",
            "moon_phase",
            "morning_astro_begin_dt",
            "evening_astro_end_dt",
            "moonrise_dt",
            "moonset_dt",
        ]
    ]


def build_master_df(
    weather_df: pd.DataFrame,
    ip_geo_df: pd.DataFrame,
    event_df: Optional[pd.DataFrame],
    timezone: str,
) -> pd.DataFrame:
    if weather_df is None or weather_df.empty:
        return pd.DataFrame()

    weather_df = weather_df.copy()

    weather_df["utcForecastHour"] = pd.to_datetime(
        weather_df["utcForecastHour"],
        utc=True,
        errors="coerce",
    )

    weather_df = weather_df.dropna(subset=["utcForecastHour"])

    if weather_df.empty:
        return pd.DataFrame()

    weather_df["local_dt"] = weather_df["utcForecastHour"].dt.tz_convert(timezone)
    weather_df["date"] = weather_df["local_dt"].dt.date

    astronomy_df = _prepare_astronomy_df(ip_geo_df)

    master_df = weather_df.merge(
        astronomy_df,
        on="date",
        how="left",
    )

    master_df["local_dt_naive"] = master_df["local_dt"].dt.tz_localize(None)

    master_df["is_after_astronomical_twilight"] = (
        master_df["local_dt_naive"] >= master_df["evening_astro_end_dt"]
    )

    master_df["is_dark_enough"] = (
        (master_df["local_dt_naive"] >= master_df["evening_astro_end_dt"])
        | (master_df["local_dt_naive"] <= master_df["morning_astro_begin_dt"])
    )

    def moon_up(row):
        t = row["local_dt_naive"]
        rise = row["moonrise_dt"]
        set_ = row["moonset_dt"]

        if pd.isna(rise) or pd.isna(set_):
            return False

        if rise <= set_:
            return (t >= rise) and (t <= set_)

        return (t >= rise) or (t <= set_)

    master_df["is_moon_up"] = master_df.apply(moon_up, axis=1)

    # Notebook-style logical observing flag
    master_df["is_good_observing_window"] = (
        master_df["is_dark_enough"].fillna(False)
        & (master_df["is_moon_up"] == False)
        & (pd.to_numeric(master_df["cloud_value"], errors="coerce") <= 30)
    )

    # Timeanddate merge
    if event_df is not None and not event_df.empty:
        event_df = event_df.copy()

        if "Object" in event_df.columns and "Event" in event_df.columns:
            event_keep = event_df[
                (
                    (event_df["Object"].str.lower() == "sun")
                    & (
                        event_df["Event"].isin(
                            [
                                "CivilTwilightEnds",
                                "NauticalTwilightEnds",
                                "Meridian",
                            ]
                        )
                    )
                )
                | (
                    (event_df["Object"].str.lower() == "moon")
                    & (event_df["Event"].isin(["Meridian"]))
                )
            ].copy()

            civil_end_df = event_keep[
                (event_keep["Object"].str.lower() == "sun")
                & (event_keep["Event"] == "CivilTwilightEnds")
            ][["date", "local_dt_naive"]].rename(
                columns={"local_dt_naive": "civil_twilight_end_dt"}
            )

            nautical_end_df = event_keep[
                (event_keep["Object"].str.lower() == "sun")
                & (event_keep["Event"] == "NauticalTwilightEnds")
            ][["date", "local_dt_naive"]].rename(
                columns={"local_dt_naive": "nautical_twilight_end_dt"}
            )

            sun_meridian_df = event_keep[
                (event_keep["Object"].str.lower() == "sun")
                & (event_keep["Event"] == "Meridian")
            ][["date", "local_dt_naive"]].rename(
                columns={"local_dt_naive": "sun_meridian_dt"}
            )

            moon_cols = ["date", "local_dt_naive"]
            optional_cols = ["Illuminated (%)", "Altitude (°)", "Moon Phase"]
            moon_cols += [c for c in optional_cols if c in event_keep.columns]

            moon_meridian_df = event_keep[
                (event_keep["Object"].str.lower() == "moon")
                & (event_keep["Event"] == "Meridian")
            ][moon_cols].rename(
                columns={
                    "local_dt_naive": "moon_meridian_dt",
                    "Illuminated (%)": "moon_illuminated_pct",
                    "Altitude (°)": "moon_meridian_altitude",
                    "Moon Phase": "moon_phase_event",
                }
            )

            master_df = master_df.merge(civil_end_df, on="date", how="left")
            master_df = master_df.merge(nautical_end_df, on="date", how="left")
            master_df = master_df.merge(sun_meridian_df, on="date", how="left")
            master_df = master_df.merge(moon_meridian_df, on="date", how="left")

    defaults = {
        "civil_twilight_end_dt": pd.NaT,
        "nautical_twilight_end_dt": pd.NaT,
        "sun_meridian_dt": pd.NaT,
        "moon_meridian_dt": pd.NaT,
        "moon_illuminated_pct": np.nan,
        "moon_meridian_altitude": np.nan,
        "moon_phase_event": np.nan,
    }

    for col, default in defaults.items():
        if col not in master_df.columns:
            master_df[col] = default

    master_df["is_after_civil_twilight_end"] = (
        master_df["local_dt_naive"] >= master_df["civil_twilight_end_dt"]
    )

    master_df["is_after_nautical_twilight_end"] = (
        master_df["local_dt_naive"] >= master_df["nautical_twilight_end_dt"]
    )

    master_df["is_after_sun_meridian"] = (
        master_df["local_dt_naive"] >= master_df["sun_meridian_dt"]
    )

    master_df["is_after_moon_meridian"] = (
        master_df["local_dt_naive"] >= master_df["moon_meridian_dt"]
    )

    master_df["hours_since_civil_twilight_end"] = (
        (master_df["local_dt_naive"] - master_df["civil_twilight_end_dt"])
        .dt.total_seconds()
        / 3600
    ).clip(-24, 24)

    master_df["hours_since_nautical_twilight_end"] = (
        (master_df["local_dt_naive"] - master_df["nautical_twilight_end_dt"])
        .dt.total_seconds()
        / 3600
    ).clip(-24, 24)

    master_df["hours_since_sun_meridian"] = (
        (master_df["local_dt_naive"] - master_df["sun_meridian_dt"])
        .dt.total_seconds()
        / 3600
    ).clip(-24, 24)

    master_df["hours_since_moon_meridian"] = (
        (master_df["local_dt_naive"] - master_df["moon_meridian_dt"])
        .dt.total_seconds()
        / 3600
    ).clip(-24, 24)

    return master_df


def apply_position_features_to_master(
    master_df: pd.DataFrame,
    position_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds nearest hourly Moon altitude/illumination from Timeanddate positions.

    This is more defensible than using moon meridian altitude for every hour:
    moonlight impact depends strongly on where the Moon actually is at that
    forecast hour.
    """
    if master_df is None or master_df.empty or position_df is None or position_df.empty:
        return master_df

    if "Object" not in position_df.columns or "UTC Time" not in position_df.columns:
        return master_df

    moon_df = position_df[
        position_df["Object"].astype(str).str.lower() == "moon"
    ].copy()
    if moon_df.empty:
        return master_df

    moon_df["utcForecastHour"] = pd.to_datetime(
        moon_df["UTC Time"],
        utc=True,
        errors="coerce",
    ).dt.floor("h")
    moon_df = moon_df.dropna(subset=["utcForecastHour"])
    if moon_df.empty:
        return master_df

    rename_cols = {}
    if "Altitude (°)" in moon_df.columns:
        rename_cols["Altitude (°)"] = "moon_hourly_altitude"
    if "Illuminated (%)" in moon_df.columns:
        rename_cols["Illuminated (%)"] = "moon_hourly_illuminated_pct"

    moon_keep_cols = ["utcForecastHour"] + list(rename_cols.values())
    moon_df = moon_df.rename(columns=rename_cols)
    moon_df = moon_df[[c for c in moon_keep_cols if c in moon_df.columns]]
    if len(moon_df.columns) <= 1:
        return master_df

    moon_df = moon_df.groupby("utcForecastHour", as_index=False).mean(numeric_only=True)

    out = master_df.copy()
    out["utcForecastHour"] = pd.to_datetime(
        out["utcForecastHour"],
        utc=True,
        errors="coerce",
    ).dt.floor("h")

    out = out.merge(moon_df, on="utcForecastHour", how="left")

    if "moon_hourly_altitude" in out.columns:
        out["moon_meridian_altitude"] = out["moon_hourly_altitude"].combine_first(
            pd.to_numeric(out.get("moon_meridian_altitude"), errors="coerce")
        )

    if "moon_hourly_illuminated_pct" in out.columns:
        out["moon_illuminated_pct"] = out["moon_hourly_illuminated_pct"].combine_first(
            pd.to_numeric(out.get("moon_illuminated_pct"), errors="coerce")
        )

    return out


# ============================================================
# SCORING MODEL — VALIDATED SYNTHESIS VERSION
# ============================================================

def classify_window(score: float) -> str:
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Marginal"
    elif score >= 25:
        return "Poor"
    return "No-Go"


def score_stargazing_windows(
    master_df: pd.DataFrame,
    bortle_index: float = 5,
) -> pd.DataFrame:
    """
    Deterministic synthesis model:

    1. Observability:
       - smooth cloud transmission
       - astronomical darkness gate
       - baseline darkness/contrast

    2. View quality once observable:
       - transparency_norm
       - seeing_norm
       - humidity_quality
       - optional aerosol/PM haze quality

    3. Darkness / contrast:
       - moon illumination
       - hourly moon altitude when available, otherwise daily meridian fallback
       - Bortle light pollution

    4. Final:
       stargazing_score = geometric blend of observability and view quality
    """

    if master_df is None or master_df.empty:
        return pd.DataFrame()

    score_df = master_df.copy()

    numeric_cols = [
        "cloud_value",
        "transparency_value",
        "seeing_value",
        "dewPoint_value",
        "temperature_value",
        "moon_illuminated_pct",
        "moon_meridian_altitude",
        "moon_hourly_altitude",
        "moon_hourly_illuminated_pct",
        "aerosol_optical_depth",
        "pm2_5",
        "dust",
    ]

    for col in numeric_cols:
        if col not in score_df.columns:
            score_df[col] = np.nan

        score_df[col] = pd.to_numeric(score_df[col], errors="coerce")

    if "date" not in score_df.columns:
        score_df["date"] = pd.to_datetime(score_df["local_dt"]).dt.date

    score_df["is_moon_up"] = (
        score_df.get("is_moon_up", False)
        .astype("boolean")
        .fillna(False)
    )

    score_df["is_dark_enough"] = (
        score_df.get("is_dark_enough", False)
        .astype("boolean")
        .fillna(False)
    )

    # Fill key weather features conservatively
    score_df["cloud_value"] = score_df["cloud_value"].fillna(100)
    score_df["transparency_value"] = score_df["transparency_value"].fillna(5)
    score_df["seeing_value"] = score_df["seeing_value"].fillna(3)
    score_df["temperature_value"] = score_df["temperature_value"].fillna(10)
    score_df["dewPoint_value"] = score_df["dewPoint_value"].fillna(5)

    # Notebook logic: propagate daily moon illumination
    score_df["moon_illuminated_pct"] = (
        score_df.groupby("date")["moon_illuminated_pct"]
        .transform(lambda x: x.ffill().bfill())
    )

    # Notebook fallback
    score_df["moon_illuminated_pct"] = score_df["moon_illuminated_pct"].fillna(50)

    # If no moon altitude, use 0 so moon brightness penalty does not overstate precision.
    score_df["moon_meridian_altitude"] = score_df["moon_meridian_altitude"].fillna(0)

    # ========================================================
    # Stage 1: Observability Gate
    # ========================================================

    cloud_fraction = (score_df["cloud_value"].clip(0, 100) / 100.0)

    # Smooth extinction curve: avoids discontinuities at 40/60/80% cloud.
    score_df["cloud_transmission"] = np.exp(
        -2.8 * np.power(cloud_fraction, 1.6)
    ).clip(0.03, 1.0)

    score_df["darkness_gate"] = np.where(score_df["is_dark_enough"], 1.0, 0.08)

    # Backward-compatible field name used by existing charts.
    score_df["visibility_penalty"] = (
        score_df["cloud_transmission"] * score_df["darkness_gate"]
    ).clip(0, 1)

    # ========================================================
    # Stage 2: Astrospheric Quality — notebook updated version
    # ========================================================

    ABS_TRANS_GOOD = 1.0
    ABS_TRANS_BAD = 5.0

    score_df["transparency_norm"] = 1 - (
        (
            score_df["transparency_value"]
            .clip(ABS_TRANS_GOOD, ABS_TRANS_BAD)
            - ABS_TRANS_GOOD
        )
        / (ABS_TRANS_BAD - ABS_TRANS_GOOD)
    )

    ABS_SEEING_GOOD = 1.0
    ABS_SEEING_BAD = 5.0

    score_df["seeing_norm"] = 1 - (
        (
            score_df["seeing_value"]
            .clip(ABS_SEEING_GOOD, ABS_SEEING_BAD)
            - ABS_SEEING_GOOD
        )
        / (ABS_SEEING_BAD - ABS_SEEING_GOOD)
    )

    score_df["dew_spread"] = (
        score_df["temperature_value"] - score_df["dewPoint_value"]
    )

    ABS_DEW_GOOD = 5.0
    ABS_DEW_BAD = 0.0

    score_df["humidity_quality"] = (
        (
            score_df["dew_spread"]
            .clip(ABS_DEW_BAD, ABS_DEW_GOOD)
            - ABS_DEW_BAD
        )
        / (ABS_DEW_GOOD - ABS_DEW_BAD)
    )

    aod = score_df["aerosol_optical_depth"]
    pm25 = score_df["pm2_5"]

    aod_quality = 1 - ((aod.clip(0.02, 0.50) - 0.02) / (0.50 - 0.02))
    pm25_quality = 1 - ((pm25.clip(0, 35) - 0) / 35)
    score_df["haze_quality"] = pd.concat(
        [aod_quality, pm25_quality],
        axis=1,
    ).mean(axis=1, skipna=True)
    score_df["haze_quality"] = score_df["haze_quality"].fillna(0.65).clip(0, 1)

    score_df["atmospheric_score"] = (
        0.40 * score_df["transparency_norm"]
        + 0.30 * score_df["seeing_norm"]
        + 0.15 * score_df["humidity_quality"]
        + 0.15 * score_df["haze_quality"]
    )

    # ========================================================
    # Stage 3: Darkness / Contrast Model
    # ========================================================

    alt_rad = np.radians(score_df["moon_meridian_altitude"].clip(0, 90))
    altitude_factor = np.sin(alt_rad)

    score_df["moon_brightness_penalty"] = np.where(
        score_df["is_moon_up"],
        (score_df["moon_illuminated_pct"] / 100.0) * altitude_factor,
        0,
    )

    score_df["light_pollution_factor"] = float(bortle_index) / 9.0

    base_darkness = 1.0 - (score_df["light_pollution_factor"] * 0.8)

    score_df["effective_darkness"] = (
        base_darkness
        - (
            score_df["moon_brightness_penalty"]
            * (1 - score_df["light_pollution_factor"])
        )
    ).clip(0, 1)

    # ========================================================
    # Final Composite Score — validated synthesis version
    # ========================================================

    score_df["observability_score"] = (
        100
        * score_df["darkness_gate"]
        * score_df["cloud_transmission"]
        * (0.70 + 0.30 * score_df["effective_darkness"])
    ).clip(0, 100)

    score_df["view_quality_score"] = (
        100
        * (
            0.52 * score_df["atmospheric_score"]
            + 0.48 * score_df["effective_darkness"]
        )
    ).clip(0, 100)

    # Geometric blending keeps either a bad gate or bad quality from being
    # hidden by the other component, while preserving useful spread at night.
    score_df["stargazing_score"] = (
        100
        * np.power(score_df["observability_score"] / 100.0, 0.62)
        * np.power(score_df["view_quality_score"] / 100.0, 0.38)
    )

    # Keep the notebook-era product for diagnostics and migration comparison.
    score_df["legacy_stargazing_score"] = (
        100
        * (
            score_df["visibility_penalty"]
            * (
                0.65 * score_df["atmospheric_score"]
                + 0.35 * score_df["effective_darkness"]
            )
        )
    ).clip(0, 100)

    score_df["stargazing_score"] = score_df["stargazing_score"].clip(0, 100)

    score_df["recommendation"] = score_df["stargazing_score"].apply(
        classify_window
    )

    return score_df


# ============================================================
# SUMMARY FUNCTIONS
# ============================================================

def get_top_windows(score_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if score_df is None or score_df.empty:
        return pd.DataFrame()

    score_df = score_df.copy()

    if "stargazing_score" not in score_df.columns:
        return pd.DataFrame()

    score_df["stargazing_score"] = pd.to_numeric(
        score_df["stargazing_score"],
        errors="coerce",
    )

    score_df = score_df.dropna(subset=["stargazing_score"])

    if score_df.empty:
        return pd.DataFrame()

    # Product logic:
    # Prefer dark enough windows.
    # If none exist, show best available but still label honestly.
    if "is_dark_enough" in score_df.columns:
        candidate_df = score_df[score_df["is_dark_enough"] == True].copy()

        if candidate_df.empty:
            candidate_df = score_df.copy()
    else:
        candidate_df = score_df.copy()

    keep_cols = [
        "local_dt",
        "date",
        "stargazing_score",
        "recommendation",
        "cluster_label",
        "cluster_description",
        "cloud_value",
        "transparency_value",
        "seeing_value",
        "moon_illuminated_pct",
        "is_dark_enough",
        "is_moon_up",
        "visibility_penalty",
        "transparency_norm",
        "seeing_norm",
        "humidity_quality",
        "moon_brightness_penalty",
        "effective_darkness",
        "atmospheric_score",
    ]

    keep_cols = [c for c in keep_cols if c in candidate_df.columns]

    out = (
        candidate_df[keep_cols]
        .sort_values("stargazing_score", ascending=False)
        .head(n)
        .copy()
    )

    out["time_label"] = pd.to_datetime(out["local_dt"]).dt.strftime(
        "%b %d, %I:%M %p"
    )

    return out


def build_daily_summary(score_df: pd.DataFrame) -> pd.DataFrame:
    if score_df is None or score_df.empty:
        return pd.DataFrame()

    if "date" not in score_df.columns or "stargazing_score" not in score_df.columns:
        return pd.DataFrame()

    daily = (
        score_df.groupby("date")
        .agg(
            avg_score=("stargazing_score", "mean"),
            peak_score=("stargazing_score", "max"),
            viable_hours=(
                "recommendation",
                lambda x: x.isin(["Excellent", "Good"]).sum(),
            ),
        )
        .reset_index()
    )

    best_rows = (
        score_df.sort_values("stargazing_score", ascending=False)
        .drop_duplicates("date")
        [["date", "local_dt"]]
        .copy()
    )

    best_rows["best_time"] = pd.to_datetime(best_rows["local_dt"]).dt.strftime(
        "%I:%M %p"
    )

    daily = daily.merge(
        best_rows[["date", "best_time"]],
        on="date",
        how="left",
    )

    return daily


def cluster_windows(score_df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    K-means clustering on sky-condition features.
    Adds cluster_label and cluster_description columns.
    Does not modify stargazing_score.
    """
    FEATURE_COLS = [
        "cloud_value",
        "transparency_norm",
        "seeing_norm",
        "effective_darkness",
        "moon_brightness_penalty",
    ]

    out = score_df.copy()

    missing = [c for c in FEATURE_COLS if c not in out.columns]
    if missing or len(out) < k:
        out["cluster_label"] = "Unknown"
        out["cluster_description"] = "Insufficient data to cluster."
        return out

    X = out[FEATURE_COLS].fillna(out[FEATURE_COLS].median()).values.astype(float)

    try:
        # Avoid scipy whiten warnings when one feature has zero variance.
        std = X.std(axis=0)
        safe_std = np.where(std == 0, 1.0, std)
        X_w = X / safe_std

        unique_count = len(np.unique(np.round(X_w, 6), axis=0))
        k_eff = min(k, unique_count, len(out))

        if k_eff < 2:
            out["cluster_label"] = "Unknown"
            out["cluster_description"] = "Insufficient feature diversity to cluster."
            return out

        unique_points = np.unique(np.round(X_w, 6), axis=0)
        rng = np.random.default_rng(42)
        center_idx = rng.choice(len(unique_points), size=k_eff, replace=False)
        centers = unique_points[center_idx].astype(float)

        labels = np.zeros(len(X_w), dtype=int)
        for _ in range(100):
            distances = np.linalg.norm(X_w[:, None, :] - centers[None, :, :], axis=2)
            new_labels = distances.argmin(axis=1)

            new_centers = centers.copy()
            for cid in range(k_eff):
                members = X_w[new_labels == cid]
                if len(members):
                    new_centers[cid] = members.mean(axis=0)
                else:
                    farthest_idx = distances.min(axis=1).argmax()
                    new_centers[cid] = X_w[farthest_idx]
                    new_labels[farthest_idx] = cid

            if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
                labels = new_labels
                break

            labels = new_labels
            centers = new_centers
    except Exception:
        out["cluster_label"] = "Unknown"
        out["cluster_description"] = "Clustering unavailable for this forecast."
        return out

    out["_cluster_id"] = labels

    # Compute centroid profiles in original (unwhitened) feature space
    centroids_orig = out.groupby("_cluster_id")[FEATURE_COLS].mean()

    def _name(row):
        cloud = row["cloud_value"]
        dark  = row["effective_darkness"]
        moon  = row["moon_brightness_penalty"]
        trans = row["transparency_norm"]
        see   = row["seeing_norm"]

        if cloud > 70:
            return "Overcast", (
                "Heavy cloud cover is the main obstacle — "
                "most deep-sky objects will be hidden."
            )
        if cloud > 45:
            return "Cloudy but Dark", (
                "The sky is dark enough, but cloud cover is the main limiting factor."
            )
        if dark < 0.28:
            return "Twilight", (
                "It's not astronomically dark yet — "
                "the sky background is too bright for faint objects."
            )
        if moon > 0.38:
            return "Moonlit", (
                "The moon is bright and above the horizon — "
                "it will wash out nebulae and faint galaxies."
            )
        if trans < 0.35:
            return "Hazy", (
                "Atmospheric haze is reducing sky transparency; "
                "only bright objects will be clearly visible."
            )
        if see < 0.35:
            return "Turbulent", (
                "Atmospheric turbulence makes star images unsteady — "
                "poor conditions for planetary or lunar detail."
            )
        return "Clear & Dark", (
            "Sky conditions are favorable — "
            "dark background with solid atmospheric clarity."
        )

    cluster_map = {
        cid: _name(row) for cid, row in centroids_orig.iterrows()
    }

    out["cluster_label"]       = out["_cluster_id"].map(lambda i: cluster_map[i][0])
    out["cluster_description"] = out["_cluster_id"].map(lambda i: cluster_map[i][1])
    out = out.drop(columns=["_cluster_id"])
    return out


def build_llm_context(
    city_name: str,
    lat: float,
    lon: float,
    score_df: pd.DataFrame,
    daily_summary: pd.DataFrame,
    top_windows: pd.DataFrame,
    weather_source: str,
    astronomy_source: str,
) -> Dict:
    return {
        "city": city_name,
        "lat": lat,
        "lon": lon,
        "weather_source": weather_source,
        "astronomy_source": astronomy_source,
        "daily_summary": (
            daily_summary.to_dict(orient="records")
            if daily_summary is not None and not daily_summary.empty
            else []
        ),
        "top_windows": (
            top_windows.to_dict(orient="records")
            if top_windows is not None and not top_windows.empty
            else []
        ),
    }


def generate_llm_recommendation(context: Dict) -> str:
    """
    LLM only explains model outputs.
    It should never recompute or modify score.
    """

    if not OPENAI_API_KEY:
        return "LLM explanation unavailable because OPENAI_API_KEY is not set."

    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = f"""
You are a stargazing recommendation assistant.

Use only the model output below.
Do not invent forecast values.
Do not change or recalculate scores.

Data:
{json.dumps(context, default=str, indent=2)}

Write a concise recommendation with:
1. Overall Outlook
2. Best Viewing Times
3. Main Limiting Factors
4. Practical Suggestion

Mention clearly if fallback data sources were used.
"""

        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            input=prompt,
        )

        return response.output_text

    except Exception as e:
        return f"LLM explanation failed: {e}"


def _dedupe_sources(retrieved_items) -> list:
    """
    Extract unique source file names from retrieved RAG chunks.
    """
    sources = []

    for item in retrieved_items or []:
        source = item.get("source") if isinstance(item, dict) else None
        if source and source not in sources:
            sources.append(source)

    return sources


def _format_sources(sources: list) -> str:
    if not sources:
        return "No retrieved source metadata available."

    return "\n".join([f"- {source}" for source in sources])


def _retrieve_stargazing_knowledge(query: str, top_k: int = 5) -> Tuple[str, str, str]:
    """
    Retrieve knowledge for AI/RAG.

    Priority:
    1. FAISS vector retrieval from vector_store/
    2. Keyword fallback retrieval from data/knowledge/

    Returns:
    - retrieved_context: formatted text block for LLM
    - rag_mode: retrieval mode description
    - retrieved_sources_text: bullet list of source names
    """

    try:
        if retrieve_vector_context is None or format_vector_context is None:
            raise RuntimeError("FAISS vector retrieval unavailable.")
        retrieved = retrieve_vector_context(query, top_k=top_k)
        retrieved_context = format_vector_context(retrieved)
        rag_mode = "FAISS vector retrieval"

    except Exception as retrieval_error:
        retrieved = retrieve_context(query, top_k=min(top_k, 4))
        retrieved_context = format_retrieved_context(retrieved)
        rag_mode = (
            "Keyword fallback retrieval. "
            f"Vector retrieval failed because: {retrieval_error}"
        )

    sources = _dedupe_sources(retrieved)
    retrieved_sources_text = _format_sources(sources)

    return retrieved_context, rag_mode, retrieved_sources_text


def _compact_context_for_ai(context: Dict) -> Dict:
    """
    Keep AI context compact and structured.

    Do not send raw full dataframes to the LLM. The pipeline already turns
    score_df into top_windows and daily_summary. This function keeps the
    explanation grounded while avoiding unnecessary tokens.
    """

    if not isinstance(context, dict):
        return {}

    compact = {
        "city": context.get("city"),
        "lat": context.get("lat"),
        "lon": context.get("lon"),
        "timezone": context.get("timezone"),
        "weather_source": context.get("weather_source"),
        "astronomy_source": context.get("astronomy_source"),
        "top_windows": context.get("top_windows", [])[:5]
        if isinstance(context.get("top_windows", []), list)
        else [],
        "daily_summary": context.get("daily_summary", [])
        if isinstance(context.get("daily_summary", []), list)
        else [],
    }

    return compact


def generate_forecast_ai_insight(context: Dict) -> str:
    """
    Generate an automatic forecast-based AI insight.

    This is different from user Q&A:
    - No user question is required.
    - It explains the current forecast, top windows, limiting factors,
      and what users should observe.
    - It uses vector RAG only as an explanation layer.
    - It does not modify or recalculate the deterministic score.
    """

    if not OPENAI_API_KEY:
        return "Forecast AI insight unavailable because OPENAI_API_KEY is not set."

    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)

        compact_context = _compact_context_for_ai(context)

        city = compact_context.get("city", "the selected location")
        top_windows = compact_context.get("top_windows", [])
        daily_summary = compact_context.get("daily_summary", [])
        weather_source = compact_context.get("weather_source", "Unknown")
        astronomy_source = compact_context.get("astronomy_source", "Unknown")

        retrieval_query = f"""
Generate a stargazing forecast explanation for {city}.

Current forecast facts:
Weather source: {weather_source}
Astronomy source: {astronomy_source}
Top observing windows: {json.dumps(top_windows, default=str)}
Daily summary: {json.dumps(daily_summary, default=str)}

Relevant concepts:
light pollution, Bortle scale, city lights, sky darkness, moon illumination,
moon phase, cloud cover, transparency, seeing, dark adaptation, urban stargazing,
what to observe under poor, hazy, moonlit, cloudy, or dark conditions.
"""

        retrieved_context, rag_mode, retrieved_sources_text = _retrieve_stargazing_knowledge(
            retrieval_query,
            top_k=5,
        )

        prompt = f"""
You are a stargazing forecast assistant.

Task:
Generate an automatic AI insight report for the current forecast.

Use:
1. Fixed deterministic model output from the app.
2. Retrieved stargazing knowledge from the RAG knowledge base.

Core rules:
- Do not recalculate the score.
- Do not change recommendation labels.
- Do not invent weather values.
- Do not add units unless those units are explicitly present in the fixed model output or retrieved context.
- Do not say a data source does not provide a variable unless this is explicitly stated in the fixed model output or retrieved context.
- If a value is missing from the AI context, say the value was not included in the provided context.
- Do not list clouds, haze, transparency, seeing, city lights, or light pollution as observing targets.
- Observing targets should only be sky objects, such as the Moon, bright planets, bright stars, star clusters, galaxies, nebulae, or the Milky Way.
- Do not claim that a specific star, planet, constellation, or deep-sky object is visible tonight unless it appears in the fixed model output or retrieved context.
- If moon illumination appears as a fallback/default value, describe it cautiously.
- If fallback data sources were used, mention that the result is less precise.
- AI/RAG only explains the score; it does not modify it.
- Use clear markdown with headings and bullet points.

Fixed model output:
{json.dumps(compact_context, default=str, indent=2)}

RAG retrieval mode:
{rag_mode}

Retrieved source names:
{retrieved_sources_text}

Retrieved knowledge context:
{retrieved_context}

Write the answer in this markdown format:

## Overall Outlook
Briefly summarize the forecast quality using the fixed model output.

## Best Viewing Windows
List the best windows from the model output. Include score and recommendation label when available.

## Main Limiting Factors
Explain the main limiting factors: cloud cover, transparency, seeing, moon illumination, darkness, and city lights. Be careful with fallback/default values.

## What to Observe
Give general sky-object suggestions only.

## Practical Advice
Give user-friendly planning advice. Recommend using the app's score chart and heatmap when choosing a time.

## Data Limitations
Mention fallback or approximation limits if relevant. Also state that AI/RAG explains the score but does not modify it.

## Retrieved Sources
{retrieved_sources_text}
"""

        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            input=prompt,
        )

        return response.output_text

    except Exception as e:
        return f"Forecast AI insight failed: {e}"


# ============================================================
# AI TRAVEL PLAN / NEARBY LOCATION SEARCH
# ============================================================

def _context_best_score(context: Dict) -> Optional[float]:
    top_windows = context.get("top_windows", []) if isinstance(context, dict) else []
    if not top_windows:
        return None

    try:
        return float(top_windows[0].get("stargazing_score"))
    except Exception:
        return None


def _destination_point(lat: float, lon: float, distance_km: float, bearing_deg: float) -> Tuple[float, float]:
    earth_radius_km = 6371.0
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    angular_distance = distance_km / earth_radius_km

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )

    return math.degrees(lat2), ((math.degrees(lon2) + 540) % 360) - 180


def _nearby_candidate_points(
    lat: float,
    lon: float,
    radius_km: float,
    max_candidates: int,
) -> list:
    if max_candidates <= 0 or radius_km <= 0:
        return []

    points = []
    rings = [
        (radius_km * 0.33, 4),
        (radius_km * 0.66, 8),
        (radius_km, 12),
    ]

    for distance_km, bearing_count in rings:
        for idx in range(bearing_count):
            if len(points) >= max_candidates:
                return points
            bearing = (360.0 / bearing_count) * idx
            cand_lat, cand_lon = _destination_point(lat, lon, distance_km, bearing)
            points.append(
                {
                    "lat": cand_lat,
                    "lon": cand_lon,
                    "distance_km": float(distance_km),
                    "bearing_deg": float(bearing),
                }
            )

    return points


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lon / 2) ** 2
    )
    return earth_radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _destination_suitability_score(tags: Dict, distance_km: float) -> float:
    tags = tags or {}
    score = max(0.0, 25.0 - distance_km)

    if tags.get("boundary") == "protected_area":
        score += 14
    if tags.get("leisure") in {"nature_reserve", "park", "recreation_ground"}:
        score += 12
    if tags.get("tourism") in {"viewpoint", "camp_site", "picnic_site"}:
        score += 11
    if tags.get("natural") in {"beach", "wood", "grassland", "heath"}:
        score += 8
    if tags.get("landuse") in {"forest", "meadow", "grass", "recreation_ground"}:
        score += 7
    if tags.get("access") in {"private", "no"}:
        score -= 25
    if tags.get("landuse") in {"industrial", "commercial", "retail"}:
        score -= 15

    return score


def _osm_element_lat_lon(element: Dict) -> Tuple[Optional[float], Optional[float]]:
    lat = element.get("lat")
    lon = element.get("lon")

    if lat is not None and lon is not None:
        return float(lat), float(lon)

    center = element.get("center") or {}
    if center.get("lat") is not None and center.get("lon") is not None:
        return float(center["lat"]), float(center["lon"])

    return None, None


def resolve_nearby_public_destinations(
    lat: float,
    lon: float,
    radius_km: float = 10,
) -> list:
    """
    Resolve useful public-ish stargazing destinations near a coordinate.

    Uses OpenStreetMap's public Overpass API. Failures return [] so travel
    planning can fall back to coordinate-based recommendations.
    """
    cache_key = (round(float(lat), 3), round(float(lon), 3), round(float(radius_km), 1))
    if cache_key in _DESTINATION_CACHE:
        return _DESTINATION_CACHE[cache_key]

    radius_m = int(max(1000, min(float(radius_km), 25.0) * 1000))
    query = f"""
    [out:json][timeout:12];
    (
      node(around:{radius_m},{lat},{lon})["leisure"~"^(park|nature_reserve|recreation_ground)$"];
      way(around:{radius_m},{lat},{lon})["leisure"~"^(park|nature_reserve|recreation_ground)$"];
      relation(around:{radius_m},{lat},{lon})["leisure"~"^(park|nature_reserve|recreation_ground)$"];
      node(around:{radius_m},{lat},{lon})["tourism"~"^(viewpoint|camp_site|picnic_site)$"];
      way(around:{radius_m},{lat},{lon})["tourism"~"^(viewpoint|camp_site|picnic_site)$"];
      relation(around:{radius_m},{lat},{lon})["tourism"~"^(viewpoint|camp_site|picnic_site)$"];
      node(around:{radius_m},{lat},{lon})["natural"~"^(beach|wood|grassland|heath)$"];
      way(around:{radius_m},{lat},{lon})["natural"~"^(beach|wood|grassland|heath)$"];
      relation(around:{radius_m},{lat},{lon})["natural"~"^(beach|wood|grassland|heath)$"];
      node(around:{radius_m},{lat},{lon})["boundary"="protected_area"];
      way(around:{radius_m},{lat},{lon})["boundary"="protected_area"];
      relation(around:{radius_m},{lat},{lon})["boundary"="protected_area"];
      node(around:{radius_m},{lat},{lon})["information"="trailhead"];
    );
    out center tags 50;
    """

    try:
        resp = requests.post(
            OVERPASS_API_URL,
            data={"data": query},
            timeout=15,
            headers={"User-Agent": "stargazing-assistant-final-project/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Overpass destination lookup failed: {e}")
        _DESTINATION_CACHE[cache_key] = []
        return []

    destinations = []
    seen = set()
    for element in data.get("elements", []):
        tags = element.get("tags") or {}
        elem_lat, elem_lon = _osm_element_lat_lon(element)
        if elem_lat is None or elem_lon is None:
            continue

        if tags.get("access") in {"private", "no"}:
            continue

        osm_key = (element.get("type"), element.get("id"))
        if osm_key in seen:
            continue
        seen.add(osm_key)

        distance_from_candidate_km = _haversine_km(lat, lon, elem_lat, elem_lon)
        name = (
            tags.get("name")
            or tags.get("official_name")
            or tags.get("operator")
            or "Unnamed public outdoor area"
        )

        destinations.append(
            {
                "name": name,
                "lat": elem_lat,
                "lon": elem_lon,
                "osm_type": element.get("type"),
                "osm_id": element.get("id"),
                "tags": tags,
                "distance_from_candidate_km": distance_from_candidate_km,
                "suitability_score": _destination_suitability_score(
                    tags,
                    distance_from_candidate_km,
                ),
            }
        )

    destinations = sorted(
        destinations,
        key=lambda item: (item["suitability_score"], -item["distance_from_candidate_km"]),
        reverse=True,
    )
    _DESTINATION_CACHE[cache_key] = destinations
    return destinations


def _osm_darkness_adjustment(destination: Optional[Dict]) -> float:
    if not destination:
        return 0.0

    tags = destination.get("tags") or {}
    adjustment = 0.0

    if tags.get("boundary") == "protected_area":
        adjustment += 0.5
    if tags.get("leisure") == "nature_reserve":
        adjustment += 0.45
    elif tags.get("leisure") in {"park", "recreation_ground"}:
        adjustment += 0.2
    if tags.get("tourism") in {"viewpoint", "camp_site"}:
        adjustment += 0.35
    if tags.get("natural") in {"wood", "grassland", "heath", "beach"}:
        adjustment += 0.25
    if tags.get("landuse") in {"industrial", "commercial", "retail"}:
        adjustment -= 0.5
    if tags.get("landuse") == "residential":
        adjustment -= 0.25

    return adjustment


def _estimate_travel_bortle(
    base_bortle_index: float,
    distance_km: float,
    destination: Optional[Dict] = None,
) -> float:
    """
    Estimate nearby sky brightness from baseline Bortle, distance, and OSM
    public/outdoor land-use signals. This is still an estimate, not a light map.
    """
    base = float(base_bortle_index)
    improvement = (distance_km / 40.0) + _osm_darkness_adjustment(destination)
    return float(np.clip(base - improvement, 1.0, 9.0))


def _score_candidate_forecast(
    point: Dict,
    timezone: str,
    days: int,
    start_date: datetime,
    candidate_bortle: float,
) -> Optional[Dict]:
    cache_key = (
        round(float(point["lat"]), 3),
        round(float(point["lon"]), 3),
        timezone,
        int(days),
        round(float(candidate_bortle), 2),
    )
    if cache_key in _TRAVEL_SCORE_CACHE:
        return _TRAVEL_SCORE_CACHE[cache_key]

    weather_df, weather_source = fetch_weather_forecast_with_fallback(
        lat=point["lat"],
        lon=point["lon"],
        timezone=timezone,
        days=days,
    )
    astronomy_df = build_fallback_astronomy_df(
        start_date=start_date,
        days=days,
    )
    master_df = build_master_df(
        weather_df=weather_df,
        ip_geo_df=astronomy_df,
        event_df=pd.DataFrame(),
        timezone=timezone,
    )
    score_df = score_stargazing_windows(
        master_df=master_df,
        bortle_index=candidate_bortle,
    )
    top_window = get_top_windows(score_df, n=1)

    if top_window is None or top_window.empty:
        return None

    row = top_window.iloc[0]
    scored = {
        "best_score": float(row.get("stargazing_score", 0)),
        "recommendation": row.get("recommendation", "Unknown"),
        "time_label": row.get("time_label", "Unknown"),
        "cloud_value": _safe_float(row.get("cloud_value")),
        "transparency_value": _safe_float(row.get("transparency_value")),
        "seeing_value": _safe_float(row.get("seeing_value")),
        "moon_illuminated_pct": _safe_float(row.get("moon_illuminated_pct")),
        "weather_source": weather_source,
        "astronomy_source": "Simple astronomy fallback",
    }
    _TRAVEL_SCORE_CACHE[cache_key] = scored
    return scored


def search_nearby_stargazing_locations(
    context: Dict,
    bortle_index: float,
    days: int = 4,
    radius_km: float = 75.0,
    max_candidates: int = 12,
    score_threshold: float = TRAVEL_PLAN_SCORE_THRESHOLD,
) -> Dict:
    compact_context = _compact_context_for_ai(context)
    current_score = _context_best_score(compact_context)

    if current_score is None:
        return {
            "status": "not_eligible",
            "reason": "No current stargazing score is available yet.",
            "current_score": None,
            "score_threshold": score_threshold,
            "candidates": [],
            "best_candidate": None,
        }

    if current_score < score_threshold:
        return {
            "status": "not_eligible",
            "reason": (
                f"Current best score is {current_score:.1f}/100, below the "
                f"{score_threshold:.0f}/100 travel-plan threshold."
            ),
            "current_score": current_score,
            "score_threshold": score_threshold,
            "candidates": [],
            "best_candidate": None,
        }

    lat = compact_context.get("lat")
    lon = compact_context.get("lon")
    timezone = compact_context.get("timezone") or "UTC"

    if lat is None or lon is None:
        return {
            "status": "not_eligible",
            "reason": "No current latitude/longitude is available for nearby search.",
            "current_score": current_score,
            "score_threshold": score_threshold,
            "candidates": [],
            "best_candidate": None,
        }

    start_date = datetime.now()
    candidates = []
    candidate_errors = []

    for point in _nearby_candidate_points(
        lat=float(lat),
        lon=float(lon),
        radius_km=float(radius_km),
        max_candidates=int(max_candidates),
    ):
        try:
            destinations = resolve_nearby_public_destinations(
                lat=point["lat"],
                lon=point["lon"],
                radius_km=min(10.0, max(3.0, float(radius_km) / 8.0)),
            )
            destination = destinations[0] if destinations else None
            candidate_bortle = _estimate_travel_bortle(
                bortle_index,
                point["distance_km"],
                destination=destination,
            )
            scored = _score_candidate_forecast(
                point=point,
                timezone=timezone,
                days=days,
                start_date=start_date,
                candidate_bortle=candidate_bortle,
            )

            if scored is None:
                continue

            candidates.append(
                {
                    "lat": point["lat"],
                    "lon": point["lon"],
                    "distance_km": point["distance_km"],
                    "bearing_deg": point["bearing_deg"],
                    "estimated_bortle_index": candidate_bortle,
                    "destination": destination,
                    "destination_name": destination.get("name") if destination else None,
                    "destination_lat": destination.get("lat") if destination else None,
                    "destination_lon": destination.get("lon") if destination else None,
                    "destination_distance_km": (
                        destination.get("distance_from_candidate_km")
                        if destination
                        else None
                    ),
                    **scored,
                }
            )
        except Exception as e:
            print(f"Nearby stargazing candidate failed: {e}")
            candidate_errors.append(str(e))
            continue

    if not candidates:
        return {
            "status": "no_candidates",
            "reason": "Nearby search did not return any scorable candidate locations.",
            "current_score": current_score,
            "score_threshold": score_threshold,
            "radius_km": radius_km,
            "candidate_errors": candidate_errors,
            "candidates": [],
            "best_candidate": None,
            "destination": None,
        }

    candidates = sorted(candidates, key=lambda item: item["best_score"], reverse=True)
    best_candidate = candidates[0]

    return {
        "status": "eligible",
        "reason": "Current score is high enough for nearby travel planning.",
        "current_score": current_score,
        "score_threshold": score_threshold,
        "radius_km": radius_km,
        "candidate_count": len(candidates),
        "candidate_errors": candidate_errors,
        "candidates": candidates,
        "best_candidate": best_candidate,
        "destination": best_candidate.get("destination"),
    }


def _safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def generate_stargazing_travel_plan(
    context: Dict,
    search_result: Dict,
) -> str:
    if search_result.get("status") != "eligible":
        return (
            "Travel plan not generated. "
            f"{search_result.get('reason', 'The current score is not eligible.')}"
        )

    best_candidate = search_result.get("best_candidate") or {}
    destination = search_result.get("destination") or best_candidate.get("destination")
    compact_context = _compact_context_for_ai(context)

    if not OPENAI_API_KEY:
        return _build_deterministic_travel_plan(compact_context, search_result)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)

        retrieval_query = f"""
Stargazing travel planning advice.
Current score: {search_result.get("current_score")}
Best nearby score: {best_candidate.get("best_score")}
Nearby location: {best_candidate.get("lat")}, {best_candidate.get("lon")}
Resolved destination: {json.dumps(destination, default=str)}
Concepts: dark sky travel, Bortle scale, local stargazing trip, moonlight,
cloud cover, transparency, seeing, safety planning, choosing observing sites.
"""
        retrieved_context, rag_mode, retrieved_sources_text = _retrieve_stargazing_knowledge(
            retrieval_query,
            top_k=5,
        )

        prompt = f"""
You are a stargazing travel-planning assistant.

Task:
Create a practical travel plan from the user's current forecast location to the
best nearby stargazing candidate found by the app.

Rules:
- Do not recalculate or change any score.
- Use only the score and candidate facts provided below.
- Do not claim the candidate is an official park, observatory, or named dark-sky site.
- If a destination object is provided, refer to its OSM name and coordinates.
- If no destination object is provided, refer to the destination by coordinates.
- Explain that nearby light pollution/Bortle is estimated, not measured from a map.
- Keep driving and safety advice general; do not invent roads, closures, fees, or exact travel times.
- Focus on where to go, why it is better, when to go, what to bring, and how to decide if the trip is worth it.
- Use markdown with clear headings and concise bullets.

Current forecast context:
{json.dumps(compact_context, default=str, indent=2)}

Nearby search result:
{json.dumps(search_result, default=str, indent=2)}

RAG retrieval mode:
{rag_mode}

Retrieved stargazing knowledge:
{retrieved_context}

Write the answer in this markdown structure:

## Travel Recommendation
State whether the user should consider traveling and summarize the best nearby coordinate.

## Why This Spot Is Better
Compare current best score and nearby best score. Explain the forecast factors.

## When To Go
Use the best candidate time window and recommendation label.

## Trip Plan
Give practical planning steps.

## What To Bring
Give concise gear and comfort recommendations.

## Caveats
Mention estimated Bortle/light pollution, fallback astronomy data, and that AI explains the deterministic search.

## Retrieved Sources
{retrieved_sources_text}
"""

        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            input=prompt,
        )

        return response.output_text

    except Exception as e:
        fallback = _build_deterministic_travel_plan(compact_context, search_result)
        return f"{fallback}\n\n_AI travel-plan generation failed: {e}_"


def _build_deterministic_travel_plan(context: Dict, search_result: Dict) -> str:
    best = search_result.get("best_candidate") or {}
    destination = search_result.get("destination") or best.get("destination") or {}
    current_score = search_result.get("current_score")
    current_label = "current location"
    if context.get("city"):
        current_label = str(context["city"])
    destination_name = destination.get("name") or "the nearby candidate area"
    destination_lat = (
        destination.get("lat")
        if destination.get("lat") is not None
        else best.get("lat", 0)
    )
    destination_lon = (
        destination.get("lon")
        if destination.get("lon") is not None
        else best.get("lon", 0)
    )
    destination_distance = destination.get("distance_from_candidate_km")
    destination_note = ""
    if destination_distance is not None:
        destination_note = (
            f"- The selected public outdoor destination is about "
            f"`{destination_distance:.1f} km` from the highest-scoring forecast point.\n"
        )

    return f"""
## Travel Recommendation
- Consider traveling from {current_label} toward **{destination_name}** at `{destination_lat:.4f}, {destination_lon:.4f}`.
- Current best score: `{current_score:.1f}/100`.
- Nearby best score: `{best.get("best_score", 0):.1f}/100`.
{destination_note}

## Why This Spot Is Better
- It is about `{best.get("distance_km", 0):.1f} km` away inside the search radius.
- Estimated Bortle index improves to `{best.get("estimated_bortle_index", 0):.1f}` using distance plus OpenStreetMap outdoor-place signals when available.
- Forecast factors: cloud `{best.get("cloud_value")}`, transparency `{best.get("transparency_value")}`, seeing `{best.get("seeing_value")}`.

## When To Go
- Best nearby window: `{best.get("time_label", "Unknown")}`.
- Recommendation label: `{best.get("recommendation", "Unknown")}`.

## Trip Plan
- Check **{destination_name}** on a map and confirm it is safe and legally accessible at night.
- Re-run the forecast shortly before leaving.
- Arrive before the best window so your eyes can dark-adapt.
- Keep a backup plan if clouds move in.

## What To Bring
- Warm layers, water, a red flashlight, a charged phone, and a simple sky map.
- Binoculars or a small telescope are optional; the naked eye is enough for basic skywatching.

## Caveats
- Nearby Bortle/light-pollution is estimated, not measured from a live map.
- Nearby astronomy values use the app's fallback approximation because moon/twilight timing changes very little across this radius.
- This plan explains deterministic score output; it does not change the score.
"""


def generate_travel_plan_for_current_forecast(
    context: Dict,
    bortle_index: float,
    days: int = 4,
    radius_km: float = 75.0,
    max_candidates: int = 12,
    score_threshold: float = TRAVEL_PLAN_SCORE_THRESHOLD,
    use_ai_text: bool = True,
) -> Dict:
    search_result = search_nearby_stargazing_locations(
        context=context,
        bortle_index=bortle_index,
        days=days,
        radius_km=radius_km,
        max_candidates=max_candidates,
        score_threshold=score_threshold,
    )

    travel_plan = None
    if search_result.get("status") == "eligible":
        if use_ai_text:
            travel_plan = generate_stargazing_travel_plan(
                context=context,
                search_result=search_result,
            )
        else:
            travel_plan = _build_deterministic_travel_plan(
                _compact_context_for_ai(context),
                search_result,
            )

    return {
        "search": search_result,
        "travel_plan": travel_plan,
    }


def answer_semantic_knowledge_question(
    user_question: str,
    context: Optional[Dict] = None,
    use_forecast_context: bool = True,
) -> str:
    """
    Semantic knowledge search / Q&A.

    User types a question. The system performs semantic search over the
    FAISS vector knowledge base and generates a grounded answer.

    If use_forecast_context is True, the current forecast context is added
    so the answer can connect general knowledge to the current location and score.
    """

    if not OPENAI_API_KEY:
        return "Semantic knowledge answer unavailable because OPENAI_API_KEY is not set."

    if not user_question or not user_question.strip():
        return "Please enter a stargazing question."

    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)

        user_question_clean = user_question.strip()
        compact_context = _compact_context_for_ai(context or {})

        retrieval_query = user_question_clean

        if use_forecast_context and compact_context:
            retrieval_query = f"""
User question:
{user_question_clean}

Current forecast context:
{json.dumps(compact_context, default=str)}

Relevant concepts:
stargazing, city lights, light pollution, Bortle scale, moon illumination,
moon phase, cloud cover, transparency, seeing, dark sky, observing tips.
"""

        retrieved_context, rag_mode, retrieved_sources_text = _retrieve_stargazing_knowledge(
            retrieval_query,
            top_k=5,
        )

        forecast_block = ""
        if use_forecast_context and compact_context:
            forecast_block = f"""
Current forecast context:
{json.dumps(compact_context, default=str, indent=2)}
"""

        prompt = f"""
You are a stargazing knowledge assistant.

Answer the user's question using retrieved knowledge.
If forecast context is provided, connect the answer to the current forecast.
If the retrieved context does not answer the question, say so.

Rules:
- Do not recalculate the app's score.
- Do not change recommendation labels.
- Do not invent forecast values.
- Do not add units unless they are present in the provided context.
- Do not list clouds, haze, transparency, seeing, city lights, or light pollution as observing targets.
- Observing targets should only be sky objects.
- Do not claim exact visibility of specific objects unless supported by the context.
- Keep the answer practical and user-friendly.
- Use markdown.
- Include retrieved source names at the end.

User question:
{user_question_clean}

{forecast_block}

RAG retrieval mode:
{rag_mode}

Retrieved knowledge context:
{retrieved_context}

Write the answer in markdown.

End with:

## Retrieved Sources
{retrieved_sources_text}
"""

        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            input=prompt,
        )

        return response.output_text

    except Exception as e:
        return f"Semantic knowledge answer failed: {e}"


def generate_rag_recommendation(context: Dict, user_question: str = "") -> str:
    """
    Backward-compatible wrapper for older app.py versions.

    The new app.py uses:
    - generate_forecast_ai_insight()
    - answer_semantic_knowledge_question()

    This wrapper keeps old UI code from breaking.
    """

    if user_question and user_question.strip():
        return answer_semantic_knowledge_question(
            user_question=user_question,
            context=context,
            use_forecast_context=True,
        )

    return generate_forecast_ai_insight(context)


# ============================================================
# TELEMETRY HELPERS
# ============================================================

def _telemetry_log(logs: list, message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs.append(f"[{timestamp}] {message}")


def _safe_preview_df(df: pd.DataFrame, max_rows: int = 5) -> list:
    if df is None or df.empty:
        return []
    return df.head(max_rows).to_dict(orient="records")


def _build_telemetry_report(
    logs: list,
    weather_df: pd.DataFrame,
    ip_geo_df: pd.DataFrame,
    event_df: pd.DataFrame,
    position_df: pd.DataFrame,
    master_df: pd.DataFrame,
    score_df: pd.DataFrame,
    top_windows: pd.DataFrame,
    weather_source: str,
    astronomy_source: str,
    bortle_index: float,
) -> Dict:
    recommendation_counts = {}
    cluster_counts = {}

    if score_df is not None and not score_df.empty:
        if "recommendation" in score_df.columns:
            recommendation_counts = (
                score_df["recommendation"].value_counts().to_dict()
            )
        if "cluster_label" in score_df.columns:
            cluster_counts = score_df["cluster_label"].value_counts().to_dict()

    return {
        "logs": logs,
        "pipeline": {
            "weather_source": weather_source,
            "astronomy_source": astronomy_source,
            "bortle_index": float(bortle_index),
            "rows": {
                "weather_df": int(len(weather_df)) if weather_df is not None else 0,
                "ip_geo_df": int(len(ip_geo_df)) if ip_geo_df is not None else 0,
                "event_df": int(len(event_df)) if event_df is not None else 0,
                "position_df": int(len(position_df)) if position_df is not None else 0,
                "master_df": int(len(master_df)) if master_df is not None else 0,
                "score_df": int(len(score_df)) if score_df is not None else 0,
                "top_windows": int(len(top_windows)) if top_windows is not None else 0,
            },
            "recommendation_counts": recommendation_counts,
            "cluster_counts": cluster_counts,
        },
        "scoring_model": {
            "observability": {
                "cloud_transmission": "exp(-2.8 * (cloud_cover_fraction ** 1.6))",
                "darkness_gate": {
                    "dark_enough": 1.0,
                    "daylight_or_twilight": 0.08,
                },
                "formula": "100 * darkness_gate * cloud_transmission * (0.70 + 0.30 * effective_darkness)",
            },
            "atmospheric_weights": {
                "transparency_norm": 0.40,
                "seeing_norm": 0.30,
                "humidity_quality": 0.15,
                "haze_quality": 0.15,
            },
            "view_quality": {
                "atmospheric_score": 0.52,
                "effective_darkness": 0.48,
                "formula": "100 * (0.52 * atmospheric_score + 0.48 * effective_darkness)",
            },
            "final_synthesis": {
                "formula": "100 * (observability_score / 100)^0.62 * (view_quality_score / 100)^0.38",
                "legacy_score_column": "legacy_stargazing_score",
            },
            "recommendation_thresholds": {
                "excellent": ">= 85",
                "good": ">= 70",
                "marginal": ">= 50",
                "poor": ">= 25",
                "no_go": "< 25",
            },
        },
        "api_preview": {
            "weather_preview": _safe_preview_df(weather_df, max_rows=5),
            "astronomy_preview": _safe_preview_df(ip_geo_df, max_rows=5),
            "event_preview": _safe_preview_df(event_df, max_rows=5),
            "position_preview": _safe_preview_df(position_df, max_rows=5),
            "top_windows_preview": _safe_preview_df(top_windows, max_rows=5),
        },
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    city_name: str = "New York City",
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    timezone: Optional[str] = None,
    days: int = 4,
    bortle_index: float = 5,
    include_tad: bool = True,
    include_positions: bool = False,
    include_llm: bool = False,
) -> Dict:
    telemetry_logs = []
    _telemetry_log(telemetry_logs, "Pipeline started.")

    if lat is None or lon is None:
        lat, lon, inferred_tz = get_city_coordinates(city_name)
        timezone = timezone or inferred_tz
        _telemetry_log(
            telemetry_logs,
            f"Resolved coordinates from city preset: lat={lat:.4f}, lon={lon:.4f}, tz={timezone}.",
        )
    else:
        timezone = timezone or "UTC"
        _telemetry_log(
            telemetry_logs,
            f"Using custom coordinates: lat={lat:.4f}, lon={lon:.4f}, tz={timezone}.",
        )

    start_date = datetime.now()

    _telemetry_log(telemetry_logs, "Fetching weather forecast.")
    weather_df, weather_source = fetch_weather_forecast_with_fallback(
        lat=lat,
        lon=lon,
        timezone=timezone,
        days=days,
    )
    _telemetry_log(
        telemetry_logs,
        f"Weather fetch completed via {weather_source}. rows={len(weather_df)}.",
    )

    _telemetry_log(telemetry_logs, "Fetching astronomy forecast.")
    ip_geo_df, astronomy_source = fetch_astronomy_with_fallback(
        lat=lat,
        lon=lon,
        start_date=start_date,
        days=days,
    )
    _telemetry_log(
        telemetry_logs,
        f"Astronomy fetch completed via {astronomy_source}. rows={len(ip_geo_df)}.",
    )

    event_df = pd.DataFrame()
    position_df = pd.DataFrame()

    if include_tad:
        _telemetry_log(telemetry_logs, "Fetching Timeanddate event dataset.")
        try:
            event_df = fetch_tad_events(
                lat=lat,
                lon=lon,
                timezone=timezone,
                start_date=start_date,
                days=days,
            )
            _telemetry_log(
                telemetry_logs,
                f"Timeanddate event fetch completed. rows={len(event_df)}.",
            )
        except Exception as e:
            print(f"Timeanddate events failed. Continuing without events. Error: {e}")
            _telemetry_log(
                telemetry_logs,
                f"Timeanddate event fetch failed. Falling back to empty dataset. error={e}",
            )
            event_df = pd.DataFrame()

    if include_positions:
        _telemetry_log(telemetry_logs, "Fetching Timeanddate position dataset.")
        try:
            position_df = fetch_tad_positions(
                lat=lat,
                lon=lon,
                start_date=start_date,
            )
            _telemetry_log(
                telemetry_logs,
                f"Timeanddate position fetch completed. rows={len(position_df)}.",
            )
        except Exception as e:
            print(f"Timeanddate positions failed. Continuing without positions. Error: {e}")
            _telemetry_log(
                telemetry_logs,
                f"Timeanddate position fetch failed. Falling back to empty dataset. error={e}",
            )
            position_df = pd.DataFrame()

    _telemetry_log(telemetry_logs, "Building master dataframe.")
    master_df = build_master_df(
        weather_df=weather_df,
        ip_geo_df=ip_geo_df,
        event_df=event_df,
        timezone=timezone,
    )

    if include_positions and position_df is not None and not position_df.empty:
        master_df = apply_position_features_to_master(
            master_df=master_df,
            position_df=position_df,
        )
        _telemetry_log(
            telemetry_logs,
            "Applied hourly Sun/Moon position features to scoring dataframe.",
        )

    _telemetry_log(telemetry_logs, f"Master dataframe ready. rows={len(master_df)}.")

    _telemetry_log(telemetry_logs, "Running deterministic scoring model.")
    score_df = score_stargazing_windows(
        master_df=master_df,
        bortle_index=bortle_index,
    )
    _telemetry_log(telemetry_logs, f"Scoring completed. rows={len(score_df)}.")

    _telemetry_log(telemetry_logs, "Running K-means clustering labels.")
    score_df = cluster_windows(score_df)
    _telemetry_log(telemetry_logs, "Clustering labels computed.")

    top_windows = get_top_windows(score_df, n=10)
    daily_summary = build_daily_summary(score_df)
    _telemetry_log(
        telemetry_logs,
        f"Summary objects ready. top_windows={len(top_windows)}, daily_summary={len(daily_summary)}.",
    )

    llm_context = build_llm_context(
        city_name=city_name,
        lat=lat,
        lon=lon,
        score_df=score_df,
        daily_summary=daily_summary,
        top_windows=top_windows,
        weather_source=weather_source,
        astronomy_source=astronomy_source,
    )

    llm_text = None

    if include_llm:
        llm_text = generate_llm_recommendation(llm_context)
        _telemetry_log(telemetry_logs, "LLM recommendation generated.")

    telemetry = _build_telemetry_report(
        logs=telemetry_logs,
        weather_df=weather_df,
        ip_geo_df=ip_geo_df,
        event_df=event_df,
        position_df=position_df,
        master_df=master_df,
        score_df=score_df,
        top_windows=top_windows,
        weather_source=weather_source,
        astronomy_source=astronomy_source,
        bortle_index=bortle_index,
    )
    _telemetry_log(telemetry_logs, "Pipeline completed.")

    return {
        "city_name": city_name,
        "lat": lat,
        "lon": lon,
        "timezone": timezone,
        "weather_source": weather_source,
        "astronomy_source": astronomy_source,
        "weather_df": weather_df,
        "ip_geo_df": ip_geo_df,
        "event_df": event_df,
        "position_df": position_df,
        "master_df": master_df,
        "score_df": score_df,
        "daily_summary": daily_summary,
        "top_windows": top_windows,
        "llm_context": llm_context,
        "llm_text": llm_text,
        "telemetry": telemetry,
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":
    result = run_pipeline(
        city_name="New York City",
        days=4,
        bortle_index=5,
        include_tad=True,
        include_positions=False,
        include_llm=False,
    )

    print("Pipeline finished.")
    print("City:", result["city_name"])
    print("Location:", result["lat"], result["lon"])
    print("Timezone:", result["timezone"])
    print("Weather source:", result["weather_source"])
    print("Astronomy source:", result["astronomy_source"])
    print("Weather df:", result["weather_df"].shape)
    print("Master df:", result["master_df"].shape)
    print("Score df:", result["score_df"].shape)
    print("Top windows:", result["top_windows"].shape)
    print(result["top_windows"].head())
