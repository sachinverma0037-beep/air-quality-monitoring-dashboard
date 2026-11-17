"""
backend.py - India Air Quality Monitoring and Environmental Analysis System
===========================================================================

All data logic lives here. frontend.py only imports this module and calls
its functions; it contains no data-processing code of its own.

Sections
--------
1.  Constants (pollutants, cities, CPCB breakpoints, categories)
2.  Data collection  (file loading, dataset download, sample dataset)
3.  Live data        (OpenAQ API v3 requests)
4.  Data cleaning    (missing values, duplicates, outliers, inconsistent data)
5.  Statistics       (descriptive statistics, correlation)
6.  Visualization data preparation
7.  Environmental classification (CPCB AQI, alerts, recommendations)
8.  Trend analysis   (regression, moving average, forecast)
9.  Export and reporting
"""

from __future__ import annotations

import html
import io
import os
import re
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------------
# 1. CONSTANTS
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
SAMPLE_PATH = DATA_DIR / "sample_air_quality.csv"

# Load the API key from the .env file if python-dotenv is installed.
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:  # the app still works; the key can be typed in the UI
    pass

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

POLLUTANT_NAMES = {
    "PM2.5": "Particulate Matter (PM2.5)",
    "PM10": "Particulate Matter (PM10)",
    "NO2": "Nitrogen Dioxide (NO2)",
    "SO2": "Sulfur Dioxide (SO2)",
    "CO": "Carbon Monoxide (CO)",
    "O3": "Ozone (O3)",
}

# Internal standard units used everywhere in this project (CPCB convention).
POLLUTANT_UNITS = {
    "PM2.5": "µg/m³",
    "PM10": "µg/m³",
    "NO2": "µg/m³",
    "SO2": "µg/m³",
    "CO": "mg/m³",
    "O3": "µg/m³",
}

# Values outside 0..max are treated as sensor errors / sentinel values.
PLAUSIBLE_MAX = {"PM2.5": 1500, "PM10": 3000, "NO2": 2000, "SO2": 2000, "CO": 200, "O3": 1000}

# Approximate centre of each city (latitude, longitude) used for the OpenAQ search.
INDIAN_CITIES = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Ahmedabad": (23.0225, 72.5714),
    "Pune": (18.5204, 73.8567),
    "Lucknow": (26.8467, 80.9462),
    "Jaipur": (26.9124, 75.7873),
    "Patna": (25.5941, 85.1376),
    "Kanpur": (26.4499, 80.3319),
}

CITY_ALIASES = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "bombay": "Mumbai",
    "madras": "Chennai",
    "calcutta": "Kolkata",
    "new delhi": "Delhi",
    "gurgaon": "Gurugram",
    "delhi ncr": "Delhi",
}

# ---- CPCB AQI -------------------------------------------------------------
# Each pollutant: concentration breakpoints (x) mapped to AQI values (y).
# The AQI sub-index is found by linear interpolation between breakpoints,
# which is the standard CPCB method:  Ip = (IHi-ILo)/(BPHi-BPLo) * (Cp-BPLo) + ILo
# Averaging period: PM2.5, PM10, NO2, SO2 -> 24-hour average
#                   CO, O3                -> 8-hour (max) average
# CO is in mg/m3, all others in ug/m3.
AQI_VALUES = [0, 50, 100, 200, 300, 400, 500]
AQI_BREAKPOINTS = {
    "PM2.5": [0, 30, 60, 90, 120, 250, 380],
    "PM10": [0, 50, 100, 250, 350, 430, 510],
    "NO2": [0, 40, 80, 180, 280, 400, 520],
    "SO2": [0, 40, 80, 380, 800, 1600, 2400],
    "CO": [0, 1, 2, 10, 17, 34, 50],
    "O3": [0, 50, 100, 168, 208, 748, 1000],
}
AVERAGING_PERIOD = {"PM2.5": "24-hour", "PM10": "24-hour", "NO2": "24-hour",
                    "SO2": "24-hour", "CO": "8-hour", "O3": "8-hour"}

CATEGORY_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
CATEGORY_COLORS = {
    "Good": "#2E9E4F",
    "Satisfactory": "#8BC34A",
    "Moderate": "#F2C500",
    "Poor": "#FF8C00",
    "Very Poor": "#E53935",
    "Severe": "#7E0023",
    "No Data": "#9E9E9E",
}
CATEGORY_RANGES = {
    "Good": "0-50", "Satisfactory": "51-100", "Moderate": "101-200",
    "Poor": "201-300", "Very Poor": "301-400", "Severe": "401-500",
}
CATEGORY_HEALTH = {
    "Good": "Minimal impact.",
    "Satisfactory": "Minor breathing discomfort to sensitive people.",
    "Moderate": "Breathing discomfort to people with lung disease, asthma and heart disease.",
    "Poor": "Breathing discomfort to most people on prolonged exposure.",
    "Very Poor": "Respiratory illness on prolonged exposure.",
    "Severe": "Affects healthy people and seriously impacts those with existing diseases.",
    "No Data": "Not enough pollutant data to calculate a CPCB AQI.",
}

# National Ambient Air Quality Standards (CPCB, 2009) used for exceedance alerts.
NAAQS_LIMITS = {"PM2.5": 60, "PM10": 100, "NO2": 80, "SO2": 80, "CO": 2.0, "O3": 100}

FREQ_MAP = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}

OPENAQ_BASE_URL = "https://api.openaq.org/v3"
OPENAQ_PARAM_MAP = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO2", "so2": "SO2", "co": "CO", "o3": "O3"}

# Conversion factors ppm -> ug/m3 (25 C, 1 atm); CO is converted to mg/m3.
PPM_TO_UGM3 = {"NO2": 1880.0, "SO2": 2620.0, "O3": 1960.0}
CO_PPM_TO_MGM3 = 1.145


class OpenAQError(Exception):
    """Raised for any problem while talking to the OpenAQ API."""


# --------------------------------------------------------------------------
# 2. DATA COLLECTION: loading files, downloading, sample dataset
# --------------------------------------------------------------------------
def _norm_key(text) -> str:
    """'PM2.5 (ug/m3)' -> 'pm25'. Used to recognise column names."""
    text = re.sub(r"\(.*?\)", "", str(text))
    return re.sub(r"[^a-z0-9]", "", text.lower())


COLUMN_ALIASES = {
    "pm25": "PM2.5", "pm2": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2", "nitrogendioxide": "NO2",
    "so2": "SO2", "sulphurdioxide": "SO2", "sulfurdioxide": "SO2",
    "co": "CO", "carbonmonoxide": "CO",
    "o3": "O3", "ozone": "O3",
    "city": "City", "cityname": "City",
    "date": "Date", "datetime": "Date", "fromdate": "Date", "timestamp": "Date",
    "measurementdate": "Date", "day": "Date",
    "state": "State",
    "aqi": "AQI_Reported", "aqibucket": "AQI_Bucket_Reported",
    # long format (one row per measurement)
    "parameter": "_param", "pollutant": "_param", "value": "_value",
}


def get_api_key() -> str:
    """Return the OpenAQ API key from the environment (.env file) or ''."""
    return os.getenv("OPENAQ_API_KEY", "").strip()


def download_dataset(url: str, dest_name: Optional[str] = None, timeout: int = 60) -> Path:
    """Download a CSV/Excel file from a direct URL into the data/ folder."""
    if not url or not url.lower().startswith(("http://", "https://")):
        raise ValueError("Please enter a valid http(s) URL.")
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise ValueError(f"Could not download the file: {exc}") from exc
    if not response.content:
        raise ValueError("The downloaded file is empty.")
    name = dest_name or Path(url.split("?")[0]).name or "downloaded_dataset.csv"
    if Path(name).suffix.lower() not in (".csv", ".xlsx", ".xls", ".txt"):
        name += ".csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    path.write_bytes(response.content)
    return path


def load_dataset(source, filename: Optional[str] = None) -> pd.DataFrame:
    """Read a CSV or Excel file (path or Streamlit upload object) into a DataFrame."""
    name = filename or getattr(source, "name", str(source))
    ext = Path(str(name)).suffix.lower()
    try:
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(source)
        elif ext in (".csv", ".txt"):
            try:
                df = pd.read_csv(source)
            except UnicodeDecodeError:
                if hasattr(source, "seek"):
                    source.seek(0)
                df = pd.read_csv(source, encoding="latin-1")
        else:
            raise ValueError("Unsupported file type. Please use .csv, .xlsx or .xls files.")
    except ValueError:
        raise
    except Exception as exc:  # corrupt file, missing openpyxl, etc.
        raise ValueError(f"Could not read the file: {exc}") from exc
    if df.empty:
        raise ValueError("The file contains no rows.")
    return df


def generate_sample_dataset(start: str = "2023-01-01", end: str = "2024-12-31", seed: int = 42,
                            save_path: Optional[Path] = SAMPLE_PATH) -> pd.DataFrame:
    """
    Build the SAMPLE dataset (synthetic demonstration data - NOT real measurements).

    It imitates typical Indian patterns (winter smog peaks, monsoon dips) and
    intentionally contains a few problems (missing values, duplicate rows,
    spikes, inconsistent city names) so that the cleaning module has
    something to show. The 'Source' column labels every row as SAMPLE.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    doy = dates.dayofyear.values
    winter = np.cos(2 * np.pi * (doy - 15) / 365.25)          # +1 mid January
    monsoon = np.exp(-((doy - 215) ** 2) / (2 * 35.0 ** 2))   # dip around early August
    trend = 1 - 0.06 * np.linspace(0, 1, n)                     # slow improvement

    # city: (PM2.5, PM10, NO2, SO2, CO mg/m3, O3, winter amplitude)
    base = {
        "Delhi": (95, 190, 52, 14, 1.5, 38, 0.55),
        "Mumbai": (42, 92, 38, 12, 0.9, 36, 0.25),
        "Kolkata": (65, 125, 44, 10, 1.1, 33, 0.45),
        "Chennai": (32, 72, 24, 9, 0.7, 40, 0.15),
        "Bengaluru": (28, 62, 28, 7, 0.8, 37, 0.10),
        "Hyderabad": (40, 88, 27, 8, 0.8, 41, 0.20),
    }
    frames = []
    for city, (p25, p10, no2, so2, co, o3, amp) in base.items():
        season = 1 + amp * winter - 0.30 * monsoon
        common = rng.lognormal(0, 0.22, n)
        frames.append(pd.DataFrame({
            "City": city,
            "Date": dates,
            "PM2.5": p25 * season * trend * common * rng.lognormal(0, 0.08, n),
            "PM10": p10 * season * trend * common * rng.lognormal(0, 0.10, n),
            "NO2": no2 * (1 + 0.6 * amp * winter) * common ** 0.6 * rng.lognormal(0, 0.12, n),
            "SO2": so2 * (1 + 0.2 * winter) * rng.lognormal(0, 0.20, n),
            "CO": co * (1 + 0.5 * amp * winter) * common ** 0.5 * rng.lognormal(0, 0.12, n),
            "O3": o3 * (1 - 0.25 * winter) * rng.lognormal(0, 0.18, n),
        }))
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    for p in POLLUTANTS:
        df[p] = df[p].round(2)

    # --- deliberate imperfections for the cleaning demo ---
    for p in POLLUTANTS:                                   # ~3% missing values
        df.loc[rng.random(len(df)) < 0.03, p] = np.nan
    spikes = rng.choice(len(df), 6, replace=False)         # sensor spikes
    df.loc[spikes, "PM2.5"] = df.loc[spikes, "PM2.5"].fillna(40) * 8
    df.loc[df.sample(12, random_state=1).index, "City"] = "delhi "      # inconsistent names
    df.loc[df.index[df["City"] == "Bengaluru"][:10], "City"] = "Bangalore"
    df["Source"] = "SAMPLE (synthetic demo data - not real measurements)"
    df = pd.concat([df, df.sample(25, random_state=7)], ignore_index=True)  # duplicate rows

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
    return df


def load_sample_dataset() -> pd.DataFrame:
    """Load data/sample_air_quality.csv (it is re-created if the file is missing)."""
    if not SAMPLE_PATH.exists():
        generate_sample_dataset()
    return load_dataset(SAMPLE_PATH)


# --------------------------------------------------------------------------
# 3. LIVE DATA: OpenAQ API v3
# --------------------------------------------------------------------------
def _retry_wait(response: requests.Response, default: float = 20.0) -> float:
    for header in ("Retry-After", "x-ratelimit-reset"):
        value = response.headers.get(header)
        if value:
            try:
                return float(value)
            except ValueError:
                pass
    return default


def _openaq_get(path: str, api_key: str, params: Optional[dict] = None,
                retries: int = 3, timeout: int = 30) -> dict:
    """GET request to OpenAQ with error handling, retries and rate-limit waiting."""
    if not api_key:
        raise OpenAQError("No OpenAQ API key provided. Add it to the .env file or type it in the app.")
    headers = {"X-API-Key": api_key, "Accept": "application/json"}
    url = OPENAQ_BASE_URL + path
    last_error = "unknown error"
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            last_error = f"network problem ({exc.__class__.__name__})"
            time.sleep(1.5 * (attempt + 1))
            continue
        code = response.status_code
        if code == 200:
            try:
                return response.json()
            except ValueError as exc:
                raise OpenAQError("OpenAQ returned an unreadable response.") from exc
        if code in (401, 403):
            raise OpenAQError("OpenAQ rejected the API key (HTTP %d). Check that it is correct." % code)
        if code in (404, 422):
            raise OpenAQError(f"OpenAQ could not process the request (HTTP {code}) for {path}.")
        if code == 429:
            last_error = "rate limit reached"
            time.sleep(min(_retry_wait(response), 65))
            continue
        if code >= 500:
            last_error = f"OpenAQ server error (HTTP {code})"
            time.sleep(2 * (attempt + 1))
            continue
        raise OpenAQError(f"Unexpected response from OpenAQ (HTTP {code}).")
    raise OpenAQError(f"OpenAQ request failed after {retries} attempts: {last_error}.")


def _timestamp(value) -> Optional[pd.Timestamp]:
    """Parse the 'utc' string of an OpenAQ datetime object."""
    try:
        return pd.to_datetime(value, utc=True)
    except (ValueError, TypeError):
        return None


def find_locations(city: str, api_key: str, radius_m: int = 25000, limit: int = 20) -> List[dict]:
    """Find monitoring stations around a city centre, best-equipped stations first."""
    if city not in INDIAN_CITIES:
        raise OpenAQError(f"Unknown city: {city}")
    lat, lon = INDIAN_CITIES[city]
    data = _openaq_get("/locations", api_key,
                       {"coordinates": f"{lat},{lon}", "radius": radius_m, "limit": limit})
    locations = data.get("results", []) or []
    ranked = []
    for loc in locations:
        sensors = [s for s in loc.get("sensors", []) if _sensor_parameter(s) in POLLUTANTS]
        if not sensors:
            continue
        last = _timestamp(((loc.get("datetimeLast") or {}).get("utc")))
        loc["_supported_sensors"] = sensors
        ranked.append((len(sensors), last if last is not None else pd.Timestamp(0, tz="UTC"), loc))
    ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [t[2] for t in ranked]


def _sensor_parameter(sensor: dict) -> Optional[str]:
    """Return our pollutant name ('PM2.5', ...) for an OpenAQ sensor dict, or None."""
    param = (sensor.get("parameter") or {}).get("name") or str(sensor.get("name", "")).split(" ")[0]
    return OPENAQ_PARAM_MAP.get(str(param).lower())


def normalize_value(parameter: str, value, unit) -> Optional[float]:
    """Convert a reading to the project's standard units (ug/m3; CO in mg/m3). None if unknown unit."""
    if value is None or pd.isna(value):
        return None
    u = str(unit or "").lower().replace("µ", "u").replace("μ", "u").replace("³", "3").replace(" ", "")
    v = float(value)
    if parameter == "CO":
        if u in ("mg/m3",):
            return v
        if u in ("ug/m3",):
            return v / 1000.0
        if u == "ppm":
            return v * CO_PPM_TO_MGM3
        if u == "ppb":
            return v / 1000.0 * CO_PPM_TO_MGM3
        return None
    if u in ("ug/m3",):
        return v
    if parameter in PPM_TO_UGM3:
        if u == "ppm":
            return v * PPM_TO_UGM3[parameter]
        if u == "ppb":
            return v * PPM_TO_UGM3[parameter] / 1000.0
    return None


def fetch_latest(cities: List[str], api_key: str, max_locations: int = 3, max_age_days: int = 7,
                 progress: Optional[Callable[[float, str], None]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fetch the most recent reading of every pollutant at stations around each city.
    Returns (long_dataframe, messages). Readings older than max_age_days are skipped.
    """
    rows: List[dict] = []
    messages: List[str] = []
    now = pd.Timestamp.now(tz="UTC")
    for i, city in enumerate(cities):
        if progress:
            progress(i / max(len(cities), 1), f"Fetching {city} ...")
        try:
            locations = find_locations(city, api_key)[:max_locations]
            if not locations:
                messages.append(f"{city}: no monitoring stations with supported pollutants were found.")
                continue
            for loc in locations:
                sensor_map = {s["id"]: s for s in loc["_supported_sensors"]}
                data = _openaq_get(f"/locations/{loc['id']}/latest", api_key, {"limit": 100})
                for rec in data.get("results", []) or []:
                    sensor = sensor_map.get(rec.get("sensorsId"))
                    if sensor is None:
                        continue
                    ts = _timestamp((rec.get("datetime") or {}).get("utc"))
                    if ts is None or (now - ts).days > max_age_days:
                        continue
                    parameter = _sensor_parameter(sensor)
                    unit = (sensor.get("parameter") or {}).get("units")
                    value = normalize_value(parameter, rec.get("value"), unit)
                    if value is None:
                        continue
                    rows.append({"City": city, "Location": loc.get("name", "Unknown"),
                                 "Parameter": parameter, "Value": value, "Unit": POLLUTANT_UNITS[parameter],
                                 "Raw_Value": rec.get("value"), "Raw_Unit": unit, "Timestamp_UTC": ts})
        except OpenAQError as exc:
            messages.append(f"{city}: {exc}")
    if progress:
        progress(1.0, "Done")
    columns = ["City", "Location", "Parameter", "Value", "Unit", "Raw_Value", "Raw_Unit", "Timestamp_UTC"]
    return pd.DataFrame(rows, columns=columns), messages


def latest_snapshot(long_df: pd.DataFrame) -> pd.DataFrame:
    """Turn the long 'latest' table into one row per city (wide format, mean of stations)."""
    if long_df is None or long_df.empty:
        return pd.DataFrame()
    wide = long_df.pivot_table(index="City", columns="Parameter", values="Value", aggfunc="mean")
    wide.columns.name = None
    wide = wide.reset_index()
    info = long_df.groupby("City").agg(Latest_Reading=("Timestamp_UTC", "max"),
                                       Stations=("Location", "nunique")).reset_index()
    out = wide.merge(info, on="City")
    out["Latest_Reading"] = out["Latest_Reading"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    out["Date"] = out["Latest_Reading"].dt.normalize()
    out["Source"] = "OpenAQ API (live)"
    first = ["City", "Date", "Latest_Reading", "Stations"]
    pollutants = [p for p in POLLUTANTS if p in out.columns]
    return out[first + pollutants + ["Source"]]


def fetch_history(cities: List[str], api_key: str, date_from, date_to, max_locations: int = 2,
                  progress: Optional[Callable[[float, str], None]] = None
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Fetch DAILY averages from OpenAQ for the chosen date range.
    Returns (wide_dataframe, long_dataframe, messages). Nothing is invented:
    days without data stay empty.
    """
    rows: List[dict] = []
    messages: List[str] = []
    params = {"datetime_from": f"{pd.Timestamp(date_from).date()}T00:00:00Z",
              "datetime_to": f"{pd.Timestamp(date_to).date()}T23:59:59Z", "limit": 1000}
    for i, city in enumerate(cities):
        try:
            locations = find_locations(city, api_key)[:max_locations]
            if not locations:
                messages.append(f"{city}: no monitoring stations with supported pollutants were found.")
                continue
            total = sum(len(l["_supported_sensors"]) for l in locations)
            done = 0
            for loc in locations:
                for sensor in loc["_supported_sensors"]:
                    done += 1
                    if progress:
                        progress((i + done / total) / len(cities), f"{city}: {loc.get('name', '')} ({done}/{total})")
                    parameter = _sensor_parameter(sensor)
                    unit = (sensor.get("parameter") or {}).get("units")
                    try:
                        try:
                            data = _openaq_get(f"/sensors/{sensor['id']}/days", api_key, params)
                        except OpenAQError as first_error:
                            if "HTTP 404" in str(first_error):   # alternative path name
                                data = _openaq_get(f"/sensors/{sensor['id']}/measurements/daily", api_key, params)
                            else:
                                raise
                    except OpenAQError as exc:
                        messages.append(f"{city} / {loc.get('name', '')} / {parameter}: {exc}")
                        if "rate limit" in str(exc) or "rejected" in str(exc):
                            raise
                        continue
                    for rec in data.get("results", []) or []:
                        period = (rec.get("period") or {}).get("datetimeFrom") or {}
                        stamp = period.get("local") or period.get("utc")
                        value = normalize_value(parameter, rec.get("value"), unit)
                        if not stamp or value is None:
                            continue
                        rows.append({"City": city, "Location": loc.get("name", "Unknown"),
                                     "Date": pd.to_datetime(str(stamp)[:10]), "Parameter": parameter,
                                     "Value": value, "Unit": POLLUTANT_UNITS[parameter]})
        except OpenAQError as exc:
            messages.append(f"{city}: stopped - {exc}")
    if progress:
        progress(1.0, "Done")
    long_df = pd.DataFrame(rows, columns=["City", "Location", "Date", "Parameter", "Value", "Unit"])
    if long_df.empty:
        return pd.DataFrame(), long_df, messages
    wide = long_df.groupby(["City", "Date", "Parameter"])["Value"].mean().unstack("Parameter").reset_index()
    wide.columns.name = None
    wide["Source"] = "OpenAQ API (live)"
    return wide, long_df, messages


# --------------------------------------------------------------------------
# 4. DATA CLEANING
# --------------------------------------------------------------------------
def available_pollutants(df: pd.DataFrame) -> List[str]:
    """Pollutant columns that exist in df and contain at least one value."""
    return [p for p in POLLUTANTS if p in df.columns and df[p].notna().any()]


def _long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        raise ValueError("No date column found. Please include a column named Date or Datetime.")
    if "City" not in df.columns:
        df["City"] = "Unknown"
    df["_param"] = df["_param"].map(lambda p: COLUMN_ALIASES.get(_norm_key(p)))
    df = df[df["_param"].isin(POLLUTANTS)].copy()
    if df.empty:
        raise ValueError("The 'parameter' column does not contain any supported pollutant.")
    df["_value"] = pd.to_numeric(df["_value"], errors="coerce")
    wide = df.pivot_table(index=["City", "Date"], columns="_param", values="_value", aggfunc="mean")
    wide.columns.name = None
    return wide.reset_index()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common column spellings to the standard names (City, Date, PM2.5, ...)."""
    df = df.copy()
    rename, used = {}, set()
    for col in df.columns:
        target = COLUMN_ALIASES.get(_norm_key(col))
        if target and target not in used:
            rename[col] = target
            used.add(target)
    df = df.rename(columns=rename)
    if "_param" in df.columns and "_value" in df.columns:
        df = _long_to_wide(df)
    if "Date" not in df.columns:
        raise ValueError("No date column found. Please include a column named Date or Datetime.")
    if not any(p in df.columns for p in POLLUTANTS):
        raise ValueError("No pollutant columns found. Expected at least one of: " + ", ".join(POLLUTANTS))
    if "City" not in df.columns:
        df["City"] = "Unknown"
    return df


def _parse_dates(series: pd.Series) -> pd.Series:
    """Convert a column to datetime (handles ISO and day-first dates; time zones -> IST)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(series, errors="coerce")
        except (ValueError, TypeError):
            parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().sum() / max(len(parsed), 1) > 0.3:
        alt = pd.to_datetime(series, errors="coerce", dayfirst=True)
        if pd.api.types.is_datetime64_any_dtype(alt) and alt.isna().sum() < parsed.isna().sum():
            parsed = alt
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return parsed


def _clean_city(name) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)) or str(name).strip() == "":
        return "Unknown"
    text = re.sub(r"\s+", " ", str(name)).strip()
    return CITY_ALIASES.get(text.lower(), text.title())


def _iqr_bounds(df: pd.DataFrame, pollutant: str, k: float) -> Tuple[pd.Series, pd.Series]:
    """Per-city IQR fences (so Delhi's normal winter is not compared with Chennai)."""
    grouped = df.groupby("City")[pollutant]
    q1 = grouped.transform(lambda s: s.quantile(0.25))
    q3 = grouped.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    return (q1 - k * iqr).clip(lower=0), q3 + k * iqr


def detect_outliers(df: pd.DataFrame, pollutants: Optional[List[str]] = None, k: float = 1.5) -> pd.DataFrame:
    """Boolean DataFrame: True where a value lies outside the per-city IQR fences."""
    pollutants = pollutants or available_pollutants(df)
    mask = pd.DataFrame(False, index=df.index, columns=pollutants)
    for p in pollutants:
        lower, upper = _iqr_bounds(df, p, k)
        mask[p] = ((df[p] < lower) | (df[p] > upper)).fillna(False)
    return mask


def outlier_summary(df: pd.DataFrame, pollutants: Optional[List[str]] = None, k: float = 1.5) -> pd.DataFrame:
    """Number and percentage of IQR outliers per pollutant."""
    pollutants = pollutants or available_pollutants(df)
    mask = detect_outliers(df, pollutants, k)
    out = pd.DataFrame({"Outliers": mask.sum(), "Observed values": df[pollutants].notna().sum()})
    out["Outliers (%)"] = (out["Outliers"] / out["Observed values"].replace(0, np.nan) * 100).round(2)
    return out


def handle_missing_values(df: pd.DataFrame, pollutants: List[str], method: str = "interpolate") -> pd.DataFrame:
    """
    Fill missing pollutant values city by city.
      interpolate : linear interpolation over time (gaps up to 14 rows), the rest -> city median
      mean/median : city mean / city median
      none        : leave as it is
    A pollutant that was never measured in a city stays empty (nothing is invented).
    """
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)
    if method == "none":
        return df
    if method not in ("interpolate", "mean", "median"):
        raise ValueError("missing_method must be interpolate, mean, median or none.")
    for p in pollutants:
        if method == "interpolate":
            df[p] = df.groupby("City")[p].transform(
                lambda s: s.interpolate(method="linear", limit=14, limit_direction="both"))
            df[p] = df[p].fillna(df.groupby("City")[p].transform("median"))
        else:
            df[p] = df[p].fillna(df.groupby("City")[p].transform(method))
    return df


def clean_data(df: pd.DataFrame, missing_method: str = "interpolate", outlier_action: str = "flag",
               outlier_k: float = 1.5, co_unit: str = "mg/m3") -> Tuple[pd.DataFrame, dict]:
    """
    Full cleaning pipeline. Returns (clean_dataframe, report_dictionary).

    Steps: rename columns -> numeric/datetime conversion -> tidy city names ->
    remove duplicates -> invalid values to NaN -> outlier handling -> fill missing.
    outlier_action: 'flag' (count only), 'cap' (clip to IQR fences), 'remove' (set to NaN, then fill).
    """
    if outlier_action not in ("flag", "cap", "remove"):
        raise ValueError("outlier_action must be flag, cap or remove.")
    df = standardize_columns(df)
    report: dict = {"rows_initial": int(len(df)), "missing_method": missing_method,
                    "outlier_action": outlier_action, "outlier_k": outlier_k}
    pollutants = [p for p in POLLUTANTS if p in df.columns]

    for p in pollutants:                                   # text -> numbers ('NA', '-' become NaN)
        df[p] = pd.to_numeric(df[p], errors="coerce").astype(float)
    if "CO" in pollutants and co_unit == "ug/m3":
        df["CO"] = df["CO"] / 1000.0
        report["co_converted_to_mg_m3"] = True
    report["missing_before"] = {p: int(df[p].isna().sum()) for p in pollutants}

    df["Date"] = _parse_dates(df["Date"])
    report["invalid_dates_removed"] = int(df["Date"].isna().sum())
    df = df.dropna(subset=["Date"]).copy()
    df["City"] = df["City"].map(_clean_city)

    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = int(before - len(df))

    aggregation = {c: ("mean" if c in pollutants else "first") for c in df.columns if c not in ("City", "Date")}
    before = len(df)
    df = df.groupby(["City", "Date"], as_index=False, sort=True).agg(aggregation)
    report["same_day_rows_merged"] = int(before - len(df))

    invalid = 0
    for p in pollutants:                                   # negatives / impossible values
        bad = (df[p] < 0) | (df[p] > PLAUSIBLE_MAX[p])
        invalid += int(bad.sum())
        df.loc[bad, p] = np.nan
    report["invalid_values_set_missing"] = invalid

    mask = detect_outliers(df, pollutants, outlier_k)
    report["outliers_detected"] = {p: int(mask[p].sum()) for p in pollutants}
    for p in pollutants:
        if outlier_action == "cap":
            lower, upper = _iqr_bounds(df, p, outlier_k)
            df[p] = df[p].clip(lower=lower, upper=upper)
        elif outlier_action == "remove":
            df.loc[mask[p], p] = np.nan

    df = handle_missing_values(df, pollutants, missing_method)
    df = df.dropna(subset=pollutants, how="all")
    report["missing_after"] = {p: int(df[p].isna().sum()) for p in pollutants}

    others = [c for c in df.columns if c not in ["City", "Date"] + pollutants]
    df = df[["City", "Date"] + pollutants + others].sort_values(["City", "Date"]).reset_index(drop=True)
    report["rows_after"] = int(len(df))
    return df, report


# --------------------------------------------------------------------------
# 5. STATISTICS
# --------------------------------------------------------------------------
def descriptive_statistics(df: pd.DataFrame, pollutants: Optional[List[str]] = None) -> pd.DataFrame:
    """Mean, median, mode, min, max, std, variance and quartiles for every pollutant."""
    pollutants = pollutants or available_pollutants(df)
    rows = {}
    for p in pollutants:
        s = df[p].dropna()
        if s.empty:
            continue
        mode = s.round(1).mode()          # continuous data: mode of values rounded to 0.1
        rows[p] = {
            "Count": int(s.count()), "Mean": s.mean(), "Median": s.median(),
            "Mode": mode.iloc[0] if not mode.empty else np.nan,
            "Min": s.min(), "Max": s.max(), "Std Dev": s.std(), "Variance": s.var(),
            "Q1 (25%)": s.quantile(0.25), "Q2 (50%)": s.quantile(0.50), "Q3 (75%)": s.quantile(0.75),
            "IQR": s.quantile(0.75) - s.quantile(0.25),
        }
    return pd.DataFrame(rows).T.round(2)


def city_statistics(df: pd.DataFrame, pollutant: str) -> pd.DataFrame:
    """Summary statistics of one pollutant for every city."""
    out = df.groupby("City")[pollutant].agg(["count", "mean", "median", "min", "max", "std"])
    out.columns = ["Count", "Mean", "Median", "Min", "Max", "Std Dev"]
    return out.round(2).sort_values("Mean", ascending=False).reset_index()


def correlation_matrix(df: pd.DataFrame, pollutants: Optional[List[str]] = None,
                       method: str = "pearson") -> pd.DataFrame:
    """Correlation coefficients between pollutants (pearson, spearman or kendall)."""
    pollutants = pollutants or available_pollutants(df)
    return df[pollutants].corr(method=method).round(3)


def strongest_correlations(corr: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Pollutant pairs sorted by absolute correlation."""
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r):
                pairs.append({"Pollutant A": cols[i], "Pollutant B": cols[j],
                              "Correlation (r)": round(float(r), 3), "Strength": describe_correlation(r)})
    out = pd.DataFrame(pairs, columns=["Pollutant A", "Pollutant B", "Correlation (r)", "Strength"])
    if out.empty:
        return out
    out["abs"] = out["Correlation (r)"].abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").head(top_n).reset_index(drop=True)


def describe_correlation(r: float) -> str:
    a = abs(r)
    word = ("very strong" if a >= 0.8 else "strong" if a >= 0.6 else "moderate" if a >= 0.4
            else "weak" if a >= 0.2 else "negligible")
    if word == "negligible":
        return "negligible"
    return f"{word} {'positive' if r > 0 else 'negative'}"


# --------------------------------------------------------------------------
# 6. VISUALIZATION DATA PREPARATION
# --------------------------------------------------------------------------
def prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    """One row per city and calendar day (mean of all readings of that day)."""
    pollutants = available_pollutants(df)
    d = df[["City", "Date"] + pollutants].copy()
    d["Date"] = pd.to_datetime(d["Date"]).dt.normalize()
    return d.groupby(["City", "Date"], as_index=False)[pollutants].mean()


def city_comparison(df: pd.DataFrame, pollutants: Optional[List[str]] = None, stat: str = "mean") -> pd.DataFrame:
    """City x pollutant table of mean / median / max / min."""
    pollutants = pollutants or available_pollutants(df)
    return df.groupby("City")[pollutants].agg(stat).round(2)


def normalize_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Min-max scale each column to 0-1 so different units can share one heatmap."""
    span = (table.max() - table.min()).replace(0, np.nan)
    return ((table - table.min()) / span).fillna(0)


def time_series(df: pd.DataFrame, pollutant: str, freq_label: str = "Daily",
                cities: Optional[List[str]] = None) -> pd.DataFrame:
    """Long table (City, Date, <pollutant>) aggregated to Daily / Weekly / Monthly means."""
    d = df if not cities else df[df["City"].isin(cities)]
    d = d[["City", "Date", pollutant]].dropna().copy()
    d["Date"] = pd.to_datetime(d["Date"])
    out = d.groupby(["City", pd.Grouper(key="Date", freq=FREQ_MAP[freq_label])])[pollutant].mean()
    return out.dropna().reset_index()


def city_series(df: pd.DataFrame, pollutant: str, city: str = "All Cities",
                freq_label: str = "Daily") -> pd.Series:
    """Time series (DatetimeIndex) for one city, or the average of all cities."""
    d = df if city == "All Cities" else df[df["City"] == city]
    s = d.groupby(pd.to_datetime(d["Date"]).dt.normalize())[pollutant].mean().dropna()
    if FREQ_MAP[freq_label] != "D" and not s.empty:
        s = s.resample(FREQ_MAP[freq_label]).mean().dropna()
    s.name = pollutant
    return s.sort_index()


def long_format(df: pd.DataFrame, pollutants: Optional[List[str]] = None) -> pd.DataFrame:
    pollutants = pollutants or available_pollutants(df)
    return df.melt(id_vars=["City", "Date"], value_vars=pollutants,
                   var_name="Pollutant", value_name="Concentration").dropna()


# --------------------------------------------------------------------------
# 7. ENVIRONMENTAL CLASSIFICATION (CPCB AQI)
# --------------------------------------------------------------------------
def sub_index(values: pd.Series, pollutant: str) -> pd.Series:
    """CPCB sub-index (0-500) of a pollutant from its concentration (piecewise linear)."""
    arr = pd.to_numeric(values, errors="coerce").astype(float).values
    result = np.interp(arr, AQI_BREAKPOINTS[pollutant], AQI_VALUES)
    return pd.Series(result, index=values.index)


def aqi_category(aqi) -> str:
    if aqi is None or pd.isna(aqi):
        return "No Data"
    for limit, name in zip([50, 100, 200, 300, 400], CATEGORY_ORDER[:5]):
        if aqi <= limit:
            return name
    return "Severe"


def breakpoints_table() -> pd.DataFrame:
    """Concentration ranges of each CPCB category per pollutant (for the methodology section)."""
    rows = []
    for i, cat in enumerate(CATEGORY_ORDER):
        row = {"Category": cat, "AQI": CATEGORY_RANGES[cat]}
        for p in POLLUTANTS:
            bp = AQI_BREAKPOINTS[p]
            row[f"{p} ({POLLUTANT_UNITS[p]}, {AVERAGING_PERIOD[p]})"] = (
                f"> {bp[5]:g}" if i == 5 else f"{bp[i]:g} - {bp[i + 1]:g}")
        rows.append(row)
    return pd.DataFrame(rows)


def compute_aqi(daily_df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """
    Add CPCB sub-indices, AQI, category and dominant pollutant to DAILY data.

    The input must hold daily averages (see prepare_daily); PM2.5, PM10, NO2 and SO2 are
    24-hour averages. CO and O3 should be 8-hour maxima - with daily data the daily mean is used as
    an approximation (documented limitation).
    strict=True applies the CPCB rule: AQI is reported only if at least three pollutants
    are available and one of them is PM2.5 or PM10.
    """
    df = daily_df.copy()
    pollutants = [p for p in POLLUTANTS if p in df.columns]
    if not pollutants:
        raise ValueError("No pollutant columns available for AQI calculation.")
    si = pd.DataFrame({p: sub_index(df[p], p) for p in pollutants}, index=df.index)
    count = si.notna().sum(axis=1)
    pm_cols = [c for c in ("PM2.5", "PM10") if c in si.columns]
    has_pm = si[pm_cols].notna().any(axis=1) if pm_cols else pd.Series(False, index=df.index)
    valid = (count >= 3) & has_pm
    aqi = si.max(axis=1, skipna=True)
    dominant = si.fillna(-1).idxmax(axis=1).where(count > 0)
    if strict:
        aqi = aqi.where(valid)
        dominant = dominant.where(valid)
    for p in pollutants:
        df[f"{p}_SubIndex"] = si[p].round(0)
    df["AQI"] = aqi.round(0)
    df["AQI_Category"] = aqi.map(aqi_category)
    df["Dominant_Pollutant"] = dominant
    df["CPCB_Rule_Met"] = valid
    return df


def city_classification(daily_df: pd.DataFrame, strict: bool = True, as_of=None) -> pd.DataFrame:
    """
    AQI per city: the latest day with a valid AQI, or the given calendar date (as_of).
    """
    aq = compute_aqi(daily_df, strict)
    aq = aq.sort_values("Date")
    if as_of is not None:
        out = aq[aq["Date"] == pd.Timestamp(as_of).normalize()]
    else:
        latest_valid = aq.dropna(subset=["AQI"]).groupby("City").tail(1)
        latest_any = aq.groupby("City").tail(1)
        missing = latest_any[~latest_any["City"].isin(latest_valid["City"])]
        out = pd.concat([latest_valid, missing])
    return out.sort_values("AQI", ascending=False, na_position="last").reset_index(drop=True)


def generate_alerts(row: pd.Series) -> List[dict]:
    """Pollution alerts for one classification row: [{'level': ..., 'message': ...}, ...]."""
    alerts: List[dict] = []
    category = row.get("AQI_Category", "No Data")
    aqi = row.get("AQI")
    dominant = row.get("Dominant_Pollutant")
    if category in ("Very Poor", "Severe"):
        alerts.append({"level": "error", "message": f"{category.upper()} air quality (AQI {aqi:.0f}). "
                       f"Avoid outdoor activity; {dominant} is the main pollutant."})
    elif category == "Poor":
        alerts.append({"level": "warning", "message": f"POOR air quality (AQI {aqi:.0f}). "
                       "Prolonged outdoor exposure can cause breathing discomfort."})
    elif category == "Moderate":
        alerts.append({"level": "info", "message": f"Moderate air quality (AQI {aqi:.0f}). "
                       "Sensitive groups should limit heavy outdoor exercise."})
    elif category == "No Data":
        alerts.append({"level": "info", "message": "AQI could not be calculated (CPCB needs at least 3 pollutants "
                       "including PM2.5 or PM10)."})
    else:
        alerts.append({"level": "success", "message": f"{category} air quality (AQI {aqi:.0f}). No alert."})
    for p, limit in NAAQS_LIMITS.items():
        value = row.get(p)
        if value is not None and pd.notna(value) and value > limit:
            alerts.append({"level": "warning", "message":
                           f"{p} = {value:.2f} {POLLUTANT_UNITS[p]} exceeds the NAAQS limit of "
                           f"{limit:g} {POLLUTANT_UNITS[p]} ({AVERAGING_PERIOD[p]})."})
    return alerts


def get_recommendations(category: str, dominant: Optional[str]) -> Dict[str, List[str]]:
    """Health advice for the public plus measures that reduce the dominant pollutant."""
    health = {
        "Good": ["Air quality is healthy - enjoy outdoor activities."],
        "Satisfactory": ["Unusually sensitive people should consider reducing prolonged outdoor exertion."],
        "Moderate": ["People with asthma, heart or lung disease should reduce prolonged outdoor exertion.",
                     "Keep quick-relief medicines handy."],
        "Poor": ["Avoid long outdoor exercise, especially near traffic.", "Wear an N95 mask outdoors.",
                 "Keep windows closed during peak pollution hours; use an air purifier if available."],
        "Very Poor": ["Stay indoors as much as possible; avoid outdoor exercise.",
                      "Wear an N95 mask if you must go out.", "Children, elderly and patients should take extra care."],
        "Severe": ["Stay indoors and keep windows closed.", "Avoid all outdoor physical activity.",
                   "Seek medical advice if you have breathing difficulty, chest pain or persistent cough."],
    }.get(category, ["Collect data for at least three pollutants (including PM2.5 or PM10) to get advice."])
    sources = {
        "PM2.5": ["Avoid burning waste, crop residue and biomass.", "Enforce vehicle emission checks and clean fuels.",
                  "Use covered transport of construction material and water sprinkling."],
        "PM10": ["Control road and construction dust (sprinkling, covering, paved shoulders).",
                 "Increase green cover and mechanical road sweeping."],
        "NO2": ["Reduce traffic congestion; promote public transport, EVs and carpooling.",
                "Regulate diesel generators and industrial combustion."],
        "SO2": ["Use low-sulphur fuels and flue-gas desulphurisation in power plants and industry.",
                "Monitor emissions of coal-fired units."],
        "CO": ["Improve vehicle maintenance and traffic flow; reduce incomplete combustion.",
               "Avoid indoor burning of coal or wood without ventilation."],
        "O3": ["Reduce NOx and VOC emissions (traffic, solvents, industry).",
               "Limit outdoor exercise in the afternoon when ozone peaks."],
    }.get(str(dominant), ["Identify and reduce the main local emission sources."])
    return {"health": health, "action": sources}


# --------------------------------------------------------------------------
# 8. TREND ANALYSIS
# --------------------------------------------------------------------------
def compute_trend(series: pd.Series) -> dict:
    """Linear regression of concentration against time (scipy linregress)."""
    s = series.dropna()
    if len(s) < 5 or s.index.nunique() < 5:
        return {"status": "insufficient", "direction": "Not enough data (need at least 5 points)"}
    x = np.asarray((s.index - s.index[0]).days, dtype=float)
    res = stats.linregress(x, s.values.astype(float))
    mean = float(s.mean())
    per30 = res.slope * 30
    if res.pvalue < 0.05:
        direction = "Increasing" if res.slope > 0 else "Decreasing"
    else:
        direction = "Stable (no significant trend)"
    return {"status": "ok", "direction": direction, "slope_per_day": float(res.slope),
            "slope_per_30d": float(per30), "pct_per_30d": float(per30 / mean * 100) if mean else np.nan,
            "r2": float(res.rvalue ** 2), "p_value": float(res.pvalue), "n_points": int(len(s)),
            "intercept": float(res.intercept)}


def trend_line(series: pd.Series) -> pd.Series:
    """Fitted regression line values for plotting."""
    s = series.dropna()
    info = compute_trend(s)
    if info["status"] != "ok":
        return pd.Series(dtype=float)
    x = np.asarray((s.index - s.index[0]).days, dtype=float)
    return pd.Series(info["intercept"] + info["slope_per_day"] * x, index=s.index, name="Trend line")


def moving_average_frame(series: pd.Series, window: int = 7) -> pd.DataFrame:
    """Value, moving average and percentage change from the previous period."""
    out = pd.DataFrame({"Value": series})
    out["Moving_Avg"] = series.rolling(window=max(int(window), 1), min_periods=1).mean()
    out["Pct_Change"] = (series.pct_change() * 100).replace([np.inf, -np.inf], np.nan)
    return out


def most_polluted_periods(series: pd.Series, top_n: int = 5) -> pd.DataFrame:
    top = series.dropna().nlargest(top_n)
    return pd.DataFrame({"Period": top.index.strftime("%Y-%m-%d"), "Concentration": top.values.round(2)})


def historical_vs_current(df: pd.DataFrame, pollutant: str, city: str = "All Cities",
                          recent_days: int = 30) -> Optional[dict]:
    """Compare the last N days with all earlier data (and with the same period a year earlier)."""
    s = city_series(df, pollutant, city, "Daily")
    if s.empty:
        return None
    last = s.index.max()
    cutoff = last - pd.Timedelta(days=recent_days - 1)
    current, history = s[s.index >= cutoff], s[s.index < cutoff]
    if current.empty or history.empty:
        return None
    result = {"current_mean": float(current.mean()), "historical_mean": float(history.mean()),
              "current_start": cutoff.date(), "current_end": last.date()}
    result["pct_diff"] = (result["current_mean"] - result["historical_mean"]) / result["historical_mean"] * 100
    year_ago = s[(s.index >= cutoff - pd.Timedelta(days=365)) & (s.index <= last - pd.Timedelta(days=365))]
    if not year_ago.empty:
        result["last_year_mean"] = float(year_ago.mean())
        result["pct_vs_last_year"] = (result["current_mean"] - result["last_year_mean"]) / result["last_year_mean"] * 100
    return result


def _time_features(dates: pd.DatetimeIndex, origin: pd.Timestamp, seasonal: bool) -> np.ndarray:
    t = np.asarray((dates - origin).days, dtype=float)
    cols = [t]
    if seasonal:
        doy = np.asarray(dates.dayofyear, dtype=float)
        cols += [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
    return np.column_stack(cols)


def forecast_trend(series: pd.Series, horizon_days: int = 30) -> Tuple[pd.DataFrame, float]:
    """
    Simple machine-learning projection (scikit-learn LinearRegression).
    Features: time, plus a yearly sine/cosine cycle when the data spans at least one year.
    Returns (forecast_dataframe, r2_on_training_data). An indicative projection, not a prediction guarantee.
    """
    s = series.dropna()
    daily = s.resample("D").mean().interpolate(limit_direction="both") if len(s) > 1 else s
    if len(daily) < 30:
        return pd.DataFrame(columns=["Date", "Forecast"]), float("nan")
    seasonal = (daily.index.max() - daily.index.min()).days >= 365
    origin = daily.index[0]
    model = LinearRegression().fit(_time_features(daily.index, origin, seasonal), daily.values)
    r2 = float(model.score(_time_features(daily.index, origin, seasonal), daily.values))
    future = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    pred = np.clip(model.predict(_time_features(future, origin, seasonal)), 0, None)
    return pd.DataFrame({"Date": future, "Forecast": pred.round(2)}), r2


def monthly_profile(df: pd.DataFrame, pollutant: str, city: str = "All Cities") -> pd.DataFrame:
    """Average concentration for each calendar month (shows seasonality)."""
    s = city_series(df, pollutant, city, "Daily")
    if s.empty:
        return pd.DataFrame(columns=["Month", "Average"])
    prof = s.groupby(s.index.month).mean()
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return pd.DataFrame({"Month": [names[m - 1] for m in prof.index], "Average": prof.values.round(2)})


def trend_summary_table(df: pd.DataFrame, city: str = "All Cities") -> pd.DataFrame:
    """Trend direction of every pollutant (daily data, regression)."""
    rows = []
    for p in available_pollutants(df):
        info = compute_trend(city_series(df, p, city, "Daily"))
        if info["status"] != "ok":
            continue
        rows.append({"Pollutant": p, "Trend": info["direction"],
                     "Change per 30 days": round(info["slope_per_30d"], 3),
                     "Change per 30 days (%)": round(info["pct_per_30d"], 2),
                     "R²": round(info["r2"], 3), "p-value": round(info["p_value"], 4)})
    return pd.DataFrame(rows, columns=["Pollutant", "Trend", "Change per 30 days",
                                       "Change per 30 days (%)", "R²", "p-value"])


# --------------------------------------------------------------------------
# 9. EXPORT AND REPORTING
# --------------------------------------------------------------------------
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Cleaned Data") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


def save_dataframe(df: pd.DataFrame, filename: str = "cleaned_air_quality.csv") -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    df.to_csv(path, index=False)
    return path


def _md_table(df: pd.DataFrame, index: bool = True) -> str:
    table = df.reset_index() if index else df
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    divider = "| " + " | ".join("---" for _ in table.columns) + " |"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in table.itertuples(index=False)]
    return "\n".join([header, divider] + body)


def _report_sections(source_label, df, cleaning_report, stats_df, corr_df, city_class, trend_df, strict):
    """Shared content for the Markdown and HTML reports: list of (title, text, dataframe_or_None, index_flag)."""
    pollutants = available_pollutants(df)
    sections = []
    overview = (f"Data source: {source_label}\n\nRecords: {len(df)} | Cities: {df['City'].nunique()} | "
                f"Period: {pd.to_datetime(df['Date']).min().date()} to {pd.to_datetime(df['Date']).max().date()} | "
                f"Pollutants: {', '.join(pollutants)}")
    sections.append(("1. Dataset overview", overview, None, False))
    if cleaning_report:
        miss = pd.DataFrame({"Missing before": cleaning_report.get("missing_before", {}),
                             "Missing after": cleaning_report.get("missing_after", {})})
        text = (f"Rows before/after: {cleaning_report['rows_initial']} / {cleaning_report['rows_after']}. "
                f"Duplicates removed: {cleaning_report['duplicates_removed']}. "
                f"Same-day rows merged: {cleaning_report['same_day_rows_merged']}. "
                f"Invalid values set to missing: {cleaning_report['invalid_values_set_missing']}. "
                f"Missing-value method: {cleaning_report['missing_method']}. "
                f"Outlier action: {cleaning_report['outlier_action']} (IQR k={cleaning_report['outlier_k']}). "
                f"Outliers detected: {cleaning_report['outliers_detected']}")
        sections.append(("2. Data cleaning", text, miss, True))
    sections.append(("3. Descriptive statistics", "", stats_df, True))
    if corr_df is not None and not corr_df.empty:
        sections.append(("4. Correlation matrix (Pearson)", "", corr_df, True))
    if city_class is not None and not city_class.empty:
        cols = [c for c in ["City", "Date", "AQI", "AQI_Category", "Dominant_Pollutant"] if c in city_class.columns]
        view = city_class[cols].copy()
        view["Date"] = pd.to_datetime(view["Date"]).dt.strftime("%Y-%m-%d")
        method = ("CPCB AQI: sub-index per pollutant by linear interpolation between CPCB breakpoints; AQI is the "
                  "maximum sub-index; PM2.5/PM10/NO2/SO2 use 24-hour averages, CO/O3 use 8-hour values (daily mean "
                  "used as approximation for daily data). "
                  + ("Strict CPCB rule applied (at least 3 pollutants incl. PM)." if strict else "Relaxed rule used."))
        sections.append(("5. Environmental classification", method, view, False))
        valid = city_class.dropna(subset=["AQI"])
        if not valid.empty:
            worst = valid.iloc[0]
            recs = get_recommendations(worst["AQI_Category"], worst["Dominant_Pollutant"])
            text = (f"Most polluted city (latest): {worst['City']} - AQI {worst['AQI']:.0f} "
                    f"({worst['AQI_Category']}), dominant pollutant {worst['Dominant_Pollutant']}.\n\n"
                    "Recommendations: " + " ".join(recs["health"] + recs["action"]))
            sections.append(("6. Alerts and recommendations", text, None, False))
    if trend_df is not None and not trend_df.empty:
        sections.append(("7. Trend analysis (linear regression, all selected cities)", "", trend_df, False))
    return sections


def build_report_markdown(source_label, df, cleaning_report, stats_df, corr_df, city_class, trend_df,
                          strict: bool = True) -> str:
    lines = ["# India Air Quality Monitoring and Environmental Analysis Report", "",
             f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ""]
    for title, text, table, index in _report_sections(source_label, df, cleaning_report, stats_df, corr_df,
                                                       city_class, trend_df, strict):
        lines += [f"## {title}", ""]
        if text:
            lines += [text, ""]
        if table is not None:
            lines += [_md_table(table, index), ""]
    return "\n".join(lines)


def build_report_html(source_label, df, cleaning_report, stats_df, corr_df, city_class, trend_df,
                      strict: bool = True) -> str:
    parts = ["<html><head><meta charset='utf-8'><title>Air Quality Report</title><style>"
             "body{font-family:Segoe UI,Arial,sans-serif;max-width:1000px;margin:30px auto;padding:0 16px;color:#222}"
             "h1{color:#0b5394}h2{border-bottom:2px solid #0b5394;padding-bottom:4px;margin-top:32px}"
             "table{border-collapse:collapse;margin:10px 0;font-size:14px}"
             "th,td{border:1px solid #ccc;padding:6px 10px;text-align:right}"
             "th{background:#0b5394;color:#fff}</style></head><body>",
             "<h1>India Air Quality Monitoring and Environmental Analysis Report</h1>",
             f"<p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>"]
    for title, text, table, index in _report_sections(source_label, df, cleaning_report, stats_df, corr_df,
                                                       city_class, trend_df, strict):
        parts.append(f"<h2>{html.escape(title)}</h2>")
        if text:
            parts.append("".join(f"<p>{html.escape(p)}</p>" for p in text.split("\n\n")))
        if table is not None:
            parts.append(table.to_html(index=index, na_rep="", border=0))
    parts.append("</body></html>")
    return "\n".join(parts)


def save_report(content: str, filename: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename
    path.write_text(content, encoding="utf-8")
    return path
