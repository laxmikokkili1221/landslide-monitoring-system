# ============================================================
# AI-BASED LANDSLIDE & FLOOD RISK MONITORING API
# Global / Multi-Region Prototype  —  OFFLINE-CAPABLE VERSION
# ============================================================
#
# WHAT WAS BROKEN BEFORE:
#   1. calculate_terrain() had NO cache at all. Any time the
#      internet / Open-Meteo elevation API was down, it raised
#      an HTTPException(503) and killed the whole request —
#      even /complete-risk, which is supposed to be the
#      "works offline" endpoint.
#   2. The weather cache only ever stored ONE location (a
#      single flat JSON file gets overwritten every call), so
#      if you tested two different spots, the second location
#      silently got the first location's cached weather.
#   3. get_flood_data() had no cache either — offline it just
#      falls back to zeroed placeholder data instead of the
#      last real reading.
#   4. The ML feature vector was hardcoded as
#      [rainfall_mm, slope_angle] instead of being built from
#      live_feature_names.json, so if the model's feature
#      order/names ever changed, predictions would silently
#      be wrong.
#
# WHAT THIS VERSION DOES:
#   - Every external data source (weather, terrain, flood) now
#     has its own location-keyed cache file. Cache key = lat/lon
#     rounded to 3 decimals (~110m precision), so multiple
#     locations can each have their own last-known-good reading.
#   - Nothing raises an HTTPException just because the internet
#     is down. Every function degrades gracefully:
#       LIVE -> CACHED_OFFLINE -> UNAVAILABLE (zeroed defaults)
#   - The landslide model input vector is built dynamically
#     from model_features, so it stays correct even if you
#     retrain with different/more features.
#   - Added GET /cache-status so you can show judges, live,
#     which locations have offline data saved.
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import requests
import joblib
import json
import numpy as np
import math
import os

from datetime import datetime


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Landslide Risk Monitoring API",
    description="AI-based landslide and flood risk monitoring system (offline-capable)",
    version="2.3",
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_FILE = "landslide_live_model.pkl"
FEATURE_FILE = "live_feature_names.json"

WEATHER_CACHE_FILE = "weather_cache.json"
TERRAIN_CACHE_FILE = "terrain_cache.json"
FLOOD_CACHE_FILE = "flood_cache.json"


# ============================================================
# LOAD ML MODEL
# ============================================================

try:
    model = joblib.load(MODEL_FILE)
    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        model_features = json.load(f)
    print("ML model loaded successfully. Features:", model_features)
except Exception as e:
    print("Model loading failed:", e)
    model = None
    model_features = ["Rainfall_mm", "Slope_Angle"]


# ============================================================
# GENERIC LOCATION-KEYED CACHE HELPERS
# ============================================================
# Each cache file on disk looks like:
# {
#   "17.686,83.219": { ...data..., "cached_at": "2026-09-05T10:00:00" },
#   "12.97,77.594":  { ...data..., "cached_at": "..." }
# }

def location_key(latitude, longitude):
    return f"{round(latitude, 3)},{round(longitude, 3)}"


def load_cache_file(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Cache read failed for {path}:", e)
        return {}


def save_cache_entry(path, key, data):
    """
    Save/update a single location's entry inside a cache file,
    without wiping out other locations already saved there.
    """
    try:
        cache = load_cache_file(path)
        entry = dict(data)
        entry["cached_at"] = datetime.now().isoformat()
        cache[key] = entry
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        return entry
    except Exception as e:
        print(f"Cache write failed for {path}:", e)
        return data


def load_cache_entry(path, key):
    cache = load_cache_file(path)
    return cache.get(key)


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def classify_risk(score):
    score = max(0.0, min(100.0, float(score)))
    if score < 25:
        level = "LOW"
    elif score < 50:
        level = "MEDIUM"
    elif score < 75:
        level = "HIGH"
    else:
        level = "VERY HIGH"
    return round(score, 2), level


# ============================================================
# TERRAIN CALCULATION (now offline-capable)
# ============================================================

def _empty_terrain():
    return {
        "elevation_m": 0,
        "slope_angle": 0,
        "aspect": 0,
        "terrain_source": "Unavailable",
        "terrain_status": "UNAVAILABLE",
        "cached_at": None,
    }


def calculate_terrain(latitude, longitude):
    """
    Get elevation and calculate approximate slope/aspect using a
    3x3 elevation grid from Open-Meteo.

    Falls back to the last cached reading for this exact location
    if the internet/API is unavailable, and to safe zero defaults
    if there is no cache either. Never raises an exception.
    """
    key = location_key(latitude, longitude)

    try:
        grid_size = 0.01
        latitudes = [latitude - grid_size, latitude, latitude + grid_size]
        longitudes = [longitude - grid_size, longitude, longitude + grid_size]

        all_lat, all_lon = [], []
        for lat in latitudes:
            for lon in longitudes:
                all_lat.append(lat)
                all_lon.append(lon)

        lat_string = ",".join(map(str, all_lat))
        lon_string = ",".join(map(str, all_lon))

        url = (
            "https://api.open-meteo.com/v1/elevation"
            f"?latitude={lat_string}&longitude={lon_string}"
        )

        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        elevations = data.get("elevation", [])
        if len(elevations) < 9:
            raise ValueError("Insufficient elevation data")

        elevations = [safe_float(x, 0) for x in elevations]
        z = np.array(elevations).reshape(3, 3)
        center_elevation = z[1, 1]

        lat_distance = 111320 * grid_size
        lon_distance = 111320 * math.cos(math.radians(latitude)) * grid_size
        if lon_distance <= 0:
            lon_distance = lat_distance

        dz_dx = (z[1, 2] - z[1, 0]) / (2 * lon_distance)
        dz_dy = (z[2, 1] - z[0, 1]) / (2 * lat_distance)

        slope_radians = math.atan(math.sqrt(dz_dx ** 2 + dz_dy ** 2))
        slope_angle = math.degrees(slope_radians)

        aspect_radians = math.atan2(dz_dy, -dz_dx)
        aspect = math.degrees(aspect_radians)
        if aspect < 0:
            aspect += 360

        terrain_data = {
            "elevation_m": round(center_elevation, 2),
            "slope_angle": round(slope_angle, 2),
            "aspect": round(aspect, 2),
            "terrain_source": "Open-Meteo Elevation / Copernicus DEM",
            "terrain_status": "LIVE",
        }

        cached_entry = save_cache_entry(TERRAIN_CACHE_FILE, key, terrain_data)
        terrain_data["cached_at"] = cached_entry.get("cached_at")
        return terrain_data

    except Exception as e:
        print("Terrain API error:", e)

        cached = load_cache_entry(TERRAIN_CACHE_FILE, key)
        if cached is not None:
            cached = dict(cached)
            cached["terrain_status"] = "CACHED_OFFLINE"
            return cached

        print("No terrain cache for this location — using safe defaults.")
        return _empty_terrain()


# ============================================================
# WEATHER (offline-capable, per location)
# ============================================================

def _empty_weather():
    return {
        "temperature_c": 0,
        "humidity_percent": 0,
        "rainfall_mm": 0,
        "rainfall_3day_mm": 0,
        "rainfall_7day_mm": 0,
        "soil_temperature_c": 0,
        "soil_moisture": 0,
        "weather_source": "Unavailable",
        "data_status": "UNAVAILABLE",
        "cached_at": None,
    }


def get_weather(latitude, longitude):
    """
    Get current weather and accumulated rainfall.
    Internet available -> live data, saved to this location's cache.
    Internet unavailable -> last cached reading for this location.
    Neither -> safe zeroed defaults.
    """
    key = location_key(latitude, longitude)

    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}&longitude={longitude}"
            "&current=temperature_2m,relative_humidity_2m,rain,"
            "soil_temperature_0cm,soil_moisture_0_to_1cm"
            "&hourly=rain&past_days=7&forecast_days=1&timezone=auto"
        )

        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})
        hourly = data.get("hourly", {})
        rain_values = [safe_float(x, 0) for x in hourly.get("rain", [])]

        current_rainfall = safe_float(current.get("rain"), 0)
        rainfall_3day = sum(rain_values[-72:])
        rainfall_7day = sum(rain_values[-168:])

        weather_data = {
            "temperature_c": round(safe_float(current.get("temperature_2m"), 0), 2),
            "humidity_percent": round(safe_float(current.get("relative_humidity_2m"), 0), 2),
            "rainfall_mm": round(current_rainfall, 2),
            "rainfall_3day_mm": round(rainfall_3day, 2),
            "rainfall_7day_mm": round(rainfall_7day, 2),
            "soil_temperature_c": round(safe_float(current.get("soil_temperature_0cm"), 0), 2),
            "soil_moisture": round(safe_float(current.get("soil_moisture_0_to_1cm"), 0), 4),
            "weather_source": "Open-Meteo",
            "data_status": "LIVE",
        }

        cached_entry = save_cache_entry(WEATHER_CACHE_FILE, key, weather_data)
        weather_data["cached_at"] = cached_entry.get("cached_at")
        return weather_data

    except Exception as e:
        print("Live weather unavailable. Reason:", e)

        cached = load_cache_entry(WEATHER_CACHE_FILE, key)
        if cached is not None:
            cached = dict(cached)
            cached["data_status"] = "CACHED_OFFLINE"
            return cached

        print("No weather cache for this location — using safe defaults.")
        return _empty_weather()


# ============================================================
# FLOOD / RIVER DATA (offline-capable, per location)
# ============================================================

def _empty_flood(status="UNAVAILABLE", condition="Flood data unavailable"):
    return {
        "current_discharge": 0,
        "forecast_max_discharge": 0,
        "historical_mean_discharge": 0,
        "historical_p75_discharge": 0,
        "historical_p90_discharge": 0,
        "historical_p95_discharge": 0,
        "historical_max_discharge": 0,
        "flood_score": 0,
        "flood_level": "UNKNOWN",
        "flood_condition": condition,
        "flood_source": "Unavailable",
        "flood_status": status,
        "cached_at": None,
    }


def _score_discharge(value, mean, p75, p90, p95):
    if value <= mean:
        return 10
    elif value <= p75:
        return 25
    elif value <= p90:
        return 50
    elif value <= p95:
        return 75
    else:
        return 95


def get_flood_data(latitude, longitude):
    """
    Estimate river/flood risk using GloFAS river discharge.
    This is a river-flow indicator, NOT an official flood warning.
    Falls back to cached data offline, then to safe defaults.
    """
    key = location_key(latitude, longitude)

    try:
        url = (
            "https://flood-api.open-meteo.com/v1/flood"
            f"?latitude={latitude}&longitude={longitude}"
            "&daily=river_discharge,river_discharge_mean,river_discharge_max,"
            "river_discharge_p25,river_discharge_p75"
            "&past_days=30&forecast_days=7&timezone=auto"
        )

        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()

        daily = data.get("daily", {})
        discharge = [safe_float(x, 0) for x in daily.get("river_discharge", []) if x is not None]

        if len(discharge) < 5:
            result = _empty_flood(status="NO_DATA", condition="Insufficient river data")
            result["flood_source"] = "GloFAS / Open-Meteo"
            return result

        historical = discharge[: min(30, len(discharge))]
        forecast = discharge[-7:]
        current_discharge = discharge[-8] if len(discharge) >= 8 else discharge[-1]
        forecast_max = max(forecast) if forecast else current_discharge

        historical_array = np.array(historical, dtype=float)
        historical_mean = float(np.mean(historical_array))
        historical_p75 = float(np.percentile(historical_array, 75))
        historical_p90 = float(np.percentile(historical_array, 90))
        historical_p95 = float(np.percentile(historical_array, 95))
        historical_max = float(np.max(historical_array))

        current_score = _score_discharge(current_discharge, historical_mean, historical_p75, historical_p90, historical_p95)
        forecast_score = _score_discharge(forecast_max, historical_mean, historical_p75, historical_p90, historical_p95)
        flood_score = 0.60 * current_score + 0.40 * forecast_score

        # Conservative classification (current flow = primary signal)
        if current_discharge <= historical_mean:
            if forecast_max <= historical_p90:
                flood_level, flood_condition = "LOW", (
                    "Current river discharge is below recent average. "
                    "Forecast conditions do not indicate a significant flood signal."
                )
            elif forecast_max <= historical_p95:
                flood_level, flood_condition = "MEDIUM", (
                    "Current river discharge is below recent average, "
                    "but the forecast shows elevated river flow."
                )
            else:
                flood_level, flood_condition = "HIGH", (
                    "Current river discharge is below recent average, "
                    "but a strong increase is expected in the forecast."
                )
        elif current_discharge <= historical_p75:
            if forecast_max <= historical_p90:
                flood_level, flood_condition = "LOW", "River discharge is within a relatively normal range."
            elif forecast_max <= historical_p95:
                flood_level, flood_condition = "MEDIUM", (
                    "River discharge is somewhat elevated and the forecast indicates further increase."
                )
            else:
                flood_level, flood_condition = "HIGH", (
                    "River discharge is elevated and the forecast indicates a strong increase."
                )
        elif current_discharge <= historical_p90:
            flood_level, flood_condition = "MEDIUM", "Current river discharge is elevated relative to recent conditions."
        elif current_discharge <= historical_p95:
            flood_level, flood_condition = "HIGH", (
                "Current river discharge is significantly elevated. Monitor official flood warnings."
            )
        else:
            flood_level, flood_condition = "VERY HIGH", (
                "Current river discharge is exceptionally elevated. Follow official emergency guidance."
            )

        # Keep score consistent with the classified level
        if flood_level == "LOW":
            flood_score = min(flood_score, 24)
        elif flood_level == "MEDIUM":
            flood_score = max(25, min(flood_score, 49))
        elif flood_level == "HIGH":
            flood_score = max(50, min(flood_score, 74))
        else:
            flood_score = max(75, flood_score)

        flood_data = {
            "current_discharge": round(current_discharge, 3),
            "forecast_max_discharge": round(forecast_max, 3),
            "historical_mean_discharge": round(historical_mean, 3),
            "historical_p75_discharge": round(historical_p75, 3),
            "historical_p90_discharge": round(historical_p90, 3),
            "historical_p95_discharge": round(historical_p95, 3),
            "historical_max_discharge": round(historical_max, 3),
            "flood_score": round(flood_score, 2),
            "flood_level": flood_level,
            "flood_condition": flood_condition,
            "flood_source": "GloFAS / Open-Meteo",
            "flood_status": "LIVE",
        }

        cached_entry = save_cache_entry(FLOOD_CACHE_FILE, key, flood_data)
        flood_data["cached_at"] = cached_entry.get("cached_at")
        return flood_data

    except Exception as e:
        print("Flood API error:", e)

        cached = load_cache_entry(FLOOD_CACHE_FILE, key)
        if cached is not None:
            cached = dict(cached)
            cached["flood_status"] = "CACHED_OFFLINE"
            return cached

        print("No flood cache for this location — using safe defaults.")
        return _empty_flood()


# ============================================================
# LANDSLIDE RISK
# ============================================================

def calculate_landslide_risk(rainfall_mm, rainfall_3day, rainfall_7day, slope_angle, flood_level="LOW"):
    """
    Combine an ML model score (when supported) with a rule-based
    condition score built from rainfall + slope. A low-rainfall
    gate prevents false medium/high risk when nothing is happening.
    """
    rainfall_mm = safe_float(rainfall_mm, 0)
    rainfall_3day = safe_float(rainfall_3day, 0)
    rainfall_7day = safe_float(rainfall_7day, 0)
    slope_angle = safe_float(slope_angle, 0)

    # --- ML model (feature vector built dynamically from model_features) ---
    model_score = 0.0
    model_available = model is not None
    model_supported = 5 <= slope_angle <= 80

    if model_available and model_supported:
        try:
            feature_values = {
                "Rainfall_mm": rainfall_mm,
                "Slope_Angle": slope_angle,
                "Rainfall_3Day_mm": rainfall_3day,
                "Rainfall_7Day_mm": rainfall_7day,
            }
            X = np.array([[feature_values.get(name, 0.0) for name in model_features]])
            probability = model.predict_proba(X)[0][1]
            model_score = float(probability) * 100
        except Exception as e:
            print("Model prediction error:", e)
            model_score = 0.0

    # --- Current rainfall score ---
    if rainfall_mm <= 1:
        current_rain_score = 0
    elif rainfall_mm < 25:
        current_rain_score = (rainfall_mm / 25) * 15
    elif rainfall_mm < 50:
        current_rain_score = 15 + ((rainfall_mm - 25) / 25) * 15
    elif rainfall_mm < 100:
        current_rain_score = 30 + ((rainfall_mm - 50) / 50) * 25
    elif rainfall_mm < 150:
        current_rain_score = 55 + ((rainfall_mm - 100) / 50) * 20
    else:
        current_rain_score = min(100, 75 + ((rainfall_mm - 150) / 150) * 25)

    # --- 3-day rainfall score ---
    if rainfall_3day < 20:
        rain_3day_score = 0
    elif rainfall_3day < 100:
        rain_3day_score = ((rainfall_3day - 20) / 80) * 30
    elif rainfall_3day < 200:
        rain_3day_score = 30 + ((rainfall_3day - 100) / 100) * 30
    else:
        rain_3day_score = min(100, 60 + ((rainfall_3day - 200) / 400) * 40)

    # --- 7-day rainfall score ---
    if rainfall_7day < 40:
        rain_7day_score = 0
    elif rainfall_7day < 200:
        rain_7day_score = ((rainfall_7day - 40) / 160) * 30
    elif rainfall_7day < 500:
        rain_7day_score = 30 + ((rainfall_7day - 200) / 300) * 30
    else:
        rain_7day_score = min(100, 60 + ((rainfall_7day - 500) / 500) * 40)

    # --- Slope score ---
    if slope_angle < 5:
        slope_score = 0
    elif slope_angle < 15:
        slope_score = ((slope_angle - 5) / 10) * 15
    elif slope_angle < 30:
        slope_score = 15 + ((slope_angle - 15) / 15) * 25
    elif slope_angle < 45:
        slope_score = 40 + ((slope_angle - 30) / 15) * 30
    else:
        slope_score = min(100, 70 + ((slope_angle - 45) / 35) * 30)

    condition_score = (
        0.45 * current_rain_score
        + 0.30 * rain_3day_score
        + 0.10 * rain_7day_score
        + 0.15 * slope_score
    )

    combined_score = (0.50 * model_score + 0.50 * condition_score) if model_supported else condition_score

    # --- Low rainfall gate ---
    no_significant_rain = rainfall_mm < 1 and rainfall_3day < 20 and rainfall_7day < 40
    no_flood_signal = flood_level in ["LOW", "UNKNOWN", "NO_DATA"]

    if no_significant_rain and no_flood_signal:
        combined_score = min(combined_score, 10)
        condition_status = "No significant rainfall or flood signal"
    else:
        condition_status = "Weather/terrain conditions may contribute to landslide risk"

    # --- Extreme rainfall overrides ---
    if rainfall_mm >= 150:
        combined_score = max(combined_score, 70)
    if rainfall_mm >= 200:
        combined_score = max(combined_score, 80)
    if rainfall_3day >= 250 and slope_angle >= 30:
        combined_score = max(combined_score, 70)

    score, level = classify_risk(combined_score)

    return {
        "risk_score": score,
        "risk_level": level,
        "model_score": round(model_score, 2),
        "condition_score": round(condition_score, 2),
        "model_supported": model_supported,
        "condition_status": condition_status,
    }


# ============================================================
# OVERALL HAZARD
# ============================================================

def calculate_overall_hazard(landslide_score, flood_score, landslide_level, flood_level):
    landslide_score = safe_float(landslide_score, 0)
    flood_score = safe_float(flood_score, 0)

    overall_score = 0.55 * landslide_score + 0.45 * flood_score

    if landslide_level == "VERY HIGH":
        overall_score = max(overall_score, 75)
    if flood_level == "VERY HIGH":
        overall_score = max(overall_score, 75)
    if landslide_level == "HIGH" and flood_level == "HIGH":
        overall_score = max(overall_score, 70)
    if landslide_level == "LOW" and flood_level == "LOW":
        overall_score = min(overall_score, 24)

    return classify_risk(overall_score)


# ============================================================
# ALERTS
# ============================================================

def generate_alerts(landslide, flood, overall_level, weather, terrain_status="LIVE", weather_status="LIVE"):
    alerts = []

    landslide_level = landslide["risk_level"]
    flood_level = flood["flood_level"]
    rainfall = safe_float(weather.get("rainfall_mm", 0))
    rainfall_3day = safe_float(weather.get("rainfall_3day_mm", 0))

    if landslide_level == "VERY HIGH":
        alerts.append("VERY HIGH landslide risk. Avoid unstable slopes and follow official emergency instructions.")
    elif landslide_level == "HIGH":
        alerts.append("HIGH landslide risk detected. Exercise caution near steep or unstable terrain.")
    elif landslide_level == "MEDIUM":
        alerts.append("Moderate landslide conditions detected. Continue monitoring rainfall and terrain.")

    if flood_level == "VERY HIGH":
        alerts.append("VERY HIGH river/flood indicator. Avoid low-lying areas and river channels.")
    elif flood_level == "HIGH":
        alerts.append("HIGH river/flood indicator detected. Monitor official flood warnings.")
    elif flood_level == "MEDIUM":
        alerts.append("River discharge is elevated relative to recent conditions. Monitor official flood information.")

    if rainfall >= 100:
        alerts.append("Heavy current rainfall detected.")
    elif rainfall >= 50:
        alerts.append("Significant current rainfall detected.")

    if rainfall_3day >= 200:
        alerts.append("High accumulated rainfall over the last 3 days.")

    if weather_status != "LIVE" or terrain_status != "LIVE":
        alerts.append(
            f"Running in offline mode — showing last cached data "
            f"(weather: {weather_status}, terrain: {terrain_status})."
        )

    if not alerts:
        alerts.append("No significant landslide or flood hazard detected from available data.")

    return alerts


# ============================================================
# REQUEST MODELS
# ============================================================

class LocationRequest(BaseModel):
    latitude: float
    longitude: float


class PredictionRequest(BaseModel):
    rainfall_mm: float
    slope_angle: float


class CompleteRiskRequest(BaseModel):
    latitude: float
    longitude: float


class SeedLocationsRequest(BaseModel):
    locations: list[LocationRequest]


def _validate_coords(latitude, longitude):
    if not -90 <= latitude <= 90:
        raise HTTPException(status_code=400, detail="Invalid latitude")
    if not -180 <= longitude <= 180:
        raise HTTPException(status_code=400, detail="Invalid longitude")


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "system": "AI-Based Landslide & Flood Risk Monitoring",
        "version": "2.3",
        "status": "running",
        "modules": [
            "GPS / Location",
            "Automatic Terrain Analysis",
            "Live Weather",
            "Offline Caching (weather, terrain, flood)",
            "Landslide ML Prediction",
            "GloFAS Flood Monitoring",
            "Overall Hazard Assessment",
            "Alert Generation",
        ],
        "data_sources": {
            "weather": "Open-Meteo",
            "elevation": "Open-Meteo / Copernicus DEM",
            "flood": "GloFAS via Open-Meteo",
            "ml": "Trained Logistic Regression model",
        },
    }


# ============================================================
# CACHE STATUS (handy for demoing offline mode)
# ============================================================

@app.get("/cache-status")
def cache_status():
    return {
        "weather_locations_cached": list(load_cache_file(WEATHER_CACHE_FILE).keys()),
        "terrain_locations_cached": list(load_cache_file(TERRAIN_CACHE_FILE).keys()),
        "flood_locations_cached": list(load_cache_file(FLOOD_CACHE_FILE).keys()),
    }


# ============================================================
# SEED CACHE — pre-warm offline data before going offline
# ============================================================
#
# Run this ONCE while you still have internet, for every
# location you plan to demo offline. It just calls the same
# live-fetch functions used elsewhere, which automatically
# save results into the location-keyed caches.
#
# Example body:
# {
#   "locations": [
#     {"latitude": 25.5941, "longitude": 85.1376},
#     {"latitude": 11.4102, "longitude": 76.6950}
#   ]
# }

@app.post("/seed-cache")
def seed_cache(request: SeedLocationsRequest):
    results = []

    for loc in request.locations:
        latitude, longitude = loc.latitude, loc.longitude

        try:
            _validate_coords(latitude, longitude)
        except HTTPException as e:
            results.append({
                "latitude": latitude,
                "longitude": longitude,
                "status": "SKIPPED",
                "detail": e.detail,
            })
            continue

        terrain = calculate_terrain(latitude, longitude)
        weather = get_weather(latitude, longitude)
        flood = get_flood_data(latitude, longitude)

        results.append({
            "latitude": latitude,
            "longitude": longitude,
            "terrain_status": terrain.get("terrain_status"),
            "weather_status": weather.get("data_status"),
            "flood_status": flood.get("flood_status"),
        })

    seeded_ok = sum(
        1 for r in results
        if r.get("terrain_status") == "LIVE" and r.get("weather_status") == "LIVE"
    )

    return {
        "requested": len(request.locations),
        "seeded_successfully": seeded_ok,
        "results": results,
    }


# ============================================================
# LOCATION FEATURES
# ============================================================

@app.post("/location-features")
def location_features(request: LocationRequest):
    latitude, longitude = request.latitude, request.longitude
    _validate_coords(latitude, longitude)

    terrain = calculate_terrain(latitude, longitude)
    weather = get_weather(latitude, longitude)
    flood = get_flood_data(latitude, longitude)

    return {
        "latitude": latitude,
        "longitude": longitude,
        "terrain": terrain,
        "weather": weather,
        "flood": flood,
    }


# ============================================================
# SIMPLE LANDSLIDE PREDICTION
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):
    rainfall = safe_float(request.rainfall_mm)
    slope = safe_float(request.slope_angle)

    result = calculate_landslide_risk(
        rainfall_mm=rainfall,
        rainfall_3day=rainfall,
        rainfall_7day=rainfall,
        slope_angle=slope,
        flood_level="LOW",
    )

    return {
        "rainfall_mm": rainfall,
        "slope_angle": slope,
        **result,
    }


# ============================================================
# COMPLETE RISK (fully offline-capable)
# ============================================================

@app.post("/complete-risk")
def complete_risk(request: CompleteRiskRequest):
    latitude, longitude = request.latitude, request.longitude
    _validate_coords(latitude, longitude)

    terrain = calculate_terrain(latitude, longitude)
    weather = get_weather(latitude, longitude)
    flood = get_flood_data(latitude, longitude)

    landslide = calculate_landslide_risk(
        rainfall_mm=weather.get("rainfall_mm", 0),
        rainfall_3day=weather.get("rainfall_3day_mm", 0),
        rainfall_7day=weather.get("rainfall_7day_mm", 0),
        slope_angle=terrain.get("slope_angle", 0),
        flood_level=flood.get("flood_level", "UNKNOWN"),
    )

    overall_score, overall_level = calculate_overall_hazard(
        landslide_score=landslide["risk_score"],
        flood_score=flood["flood_score"],
        landslide_level=landslide["risk_level"],
        flood_level=flood["flood_level"],
    )

    terrain_status = terrain.get("terrain_status", "UNKNOWN")
    weather_status = weather.get("data_status", "UNKNOWN")

    alerts = generate_alerts(
        landslide=landslide,
        flood=flood,
        overall_level=overall_level,
        weather=weather,
        terrain_status=terrain_status,
        weather_status=weather_status,
    )

    return {
        "location": {"latitude": latitude, "longitude": longitude},
        "terrain": {
            "elevation_m": terrain["elevation_m"],
            "slope_angle": terrain["slope_angle"],
            "aspect": terrain["aspect"],
            "source": terrain["terrain_source"],
            "data_status": terrain_status,
            "cached_at": terrain.get("cached_at"),
        },
        "weather": {
            "temperature_c": weather["temperature_c"],
            "humidity_percent": weather["humidity_percent"],
            "rainfall_mm": weather["rainfall_mm"],
            "rainfall_3day_mm": weather["rainfall_3day_mm"],
            "rainfall_7day_mm": weather["rainfall_7day_mm"],
            "soil_temperature_c": weather["soil_temperature_c"],
            "soil_moisture": weather["soil_moisture"],
            "source": weather["weather_source"],
            "data_status": weather_status,
            "cached_at": weather.get("cached_at"),
        },
        "landslide_risk": landslide,
        "flood_risk": {
            "risk_score": flood["flood_score"],
            "risk_level": flood["flood_level"],
            "flood_condition": flood["flood_condition"],
            "current_discharge": flood["current_discharge"],
            "forecast_max_discharge": flood["forecast_max_discharge"],
            "historical_mean_discharge": flood["historical_mean_discharge"],
            "historical_p75_discharge": flood["historical_p75_discharge"],
            "historical_p90_discharge": flood["historical_p90_discharge"],
            "historical_p95_discharge": flood["historical_p95_discharge"],
            "historical_max_discharge": flood["historical_max_discharge"],
            "source": flood["flood_source"],
            "data_status": flood["flood_status"],
            "cached_at": flood.get("cached_at"),
        },
        "overall_hazard": {"risk_score": overall_score, "risk_level": overall_level},
        "alerts": alerts,
        "system": {
            "mode": "Global / Multi-Region",
            "offline_support": True,
            "ml_model": MODEL_FILE,
            "ml_features": model_features,
        },
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)