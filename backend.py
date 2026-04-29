import os
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

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


def fetch_weather_forecast_with_fallback(
    lat: float,
    lon: float,
    timezone: str = "UTC",
    days: int = 4,
) -> Tuple[pd.DataFrame, str]:
    try:
        weather_df = fetch_astrospheric_forecast(lat, lon)

        if weather_df is not None and not weather_df.empty:
            return weather_df, "Astrospheric"

        print("Astrospheric returned empty data. Falling back to Open-Meteo.")

    except Exception as e:
        print(f"Astrospheric failed. Falling back to Open-Meteo. Error: {e}")

    weather_df = fetch_open_meteo_forecast(
        lat=lat,
        lon=lon,
        timezone=timezone,
        days=days,
    )

    return weather_df, "Open-Meteo fallback"


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

    params = {
        "apiKey": IPGEOLOC_API_KEY,
        "location": f"{lat},{lon}",
        "dateStart": start_date.strftime("%Y-%m-%d"),
        "dateEnd": end_date.strftime("%Y-%m-%d"),
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    astronomy_data = data.get("astronomy", [])

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
        print(f"IPGeolocation astronomy failed. Using fallback astronomy. Error: {e}")

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
) -> pd.DataFrame:
    if not TAD_ACCESS_KEY or not TAD_SECRET_KEY:
        raise ValueError("Missing ASTRO_ACCESS_KEY or ASTRO_SECRET_KEY in .env")

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

    intervals = [
        TADDateTime(start_date.year, start_date.month, start_date.day, hour, 0, 0)
        for hour in range(24)
    ]

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

    return pd.DataFrame(sun_rows + moon_rows)


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


# ============================================================
# SCORING MODEL — NOTEBOOK UPDATED VERSION
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
    This follows the notebook's updated model:

    1. Visibility hard constraints:
       - cloud penalty
       - daylight penalty

    2. Atmospheric quality:
       - transparency_norm
       - seeing_norm
       - humidity_quality

    3. Darkness / contrast:
       - moon illumination
       - moon altitude factor
       - Bortle light pollution

    4. Final:
       stargazing_score = 100 * visibility_penalty * base_stargazing_quality
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
    # Stage 1: Visibility Hard Constraints
    # ========================================================

    def visibility_penalty(row):
        penalty = 1.0

        if row["cloud_value"] >= 80:
            penalty = 0.05
        elif row["cloud_value"] >= 60:
            penalty = 0.25
        elif row["cloud_value"] >= 40:
            penalty = 0.55

        if not bool(row["is_dark_enough"]):
            penalty *= 0.05

        return penalty

    score_df["visibility_penalty"] = score_df.apply(
        visibility_penalty,
        axis=1,
    )

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

    score_df["atmospheric_score"] = (
        0.45 * score_df["transparency_norm"]
        + 0.35 * score_df["seeing_norm"]
        + 0.20 * score_df["humidity_quality"]
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
    # Final Composite Score — notebook updated version
    # ========================================================

    base_stargazing_quality = (
        0.65 * score_df["atmospheric_score"]
        + 0.35 * score_df["effective_darkness"]
    )

    score_df["stargazing_score"] = (
        100
        * (
            score_df["visibility_penalty"]
            * base_stargazing_quality
        )
    )

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
    from scipy.cluster.vq import kmeans2, whiten

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
        # whiten = divide each feature column by its std (equivalent to StandardScaler)
        X_w = whiten(X)
        _, labels = kmeans2(X_w, k, seed=42, minit="points")
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
    if lat is None or lon is None:
        lat, lon, inferred_tz = get_city_coordinates(city_name)
        timezone = timezone or inferred_tz
    else:
        timezone = timezone or "UTC"

    start_date = datetime.now()

    weather_df, weather_source = fetch_weather_forecast_with_fallback(
        lat=lat,
        lon=lon,
        timezone=timezone,
        days=days,
    )

    ip_geo_df, astronomy_source = fetch_astronomy_with_fallback(
        lat=lat,
        lon=lon,
        start_date=start_date,
        days=days,
    )

    event_df = pd.DataFrame()
    position_df = pd.DataFrame()

    if include_tad:
        try:
            event_df = fetch_tad_events(
                lat=lat,
                lon=lon,
                timezone=timezone,
                start_date=start_date,
                days=days,
            )
        except Exception as e:
            print(f"Timeanddate events failed. Continuing without events. Error: {e}")
            event_df = pd.DataFrame()

    if include_positions:
        try:
            position_df = fetch_tad_positions(
                lat=lat,
                lon=lon,
                start_date=start_date,
            )
        except Exception as e:
            print(f"Timeanddate positions failed. Continuing without positions. Error: {e}")
            position_df = pd.DataFrame()

    master_df = build_master_df(
        weather_df=weather_df,
        ip_geo_df=ip_geo_df,
        event_df=event_df,
        timezone=timezone,
    )

    score_df = score_stargazing_windows(
        master_df=master_df,
        bortle_index=bortle_index,
    )

    score_df = cluster_windows(score_df)

    top_windows = get_top_windows(score_df, n=10)
    daily_summary = build_daily_summary(score_df)

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
