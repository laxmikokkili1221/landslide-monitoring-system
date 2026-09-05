import streamlit as st
import requests
import pandas as pd
import json
import os

from streamlit_geolocation import streamlit_geolocation


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Disaster Risk Monitoring",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# API CONFIGURATION
# ============================================================

API_URL = "https://landslide-monitoring-system-a8l0.onrender.com"


# ============================================================
# LOCATION-NAME LOOKUP (reverse geocoding, offline-cached)
# ============================================================
# FIX (bug 1): previously "selected_location" was ONLY ever
# set inside the 4 preset city buttons, so any manual/GPS
# coordinate change kept showing whatever label was last set
# (always "Patna, Bihar" by default). This resolves a real
# name for ANY lat/lon, and caches it to a local file so it
# still works offline for locations already looked up once.

GEOCODE_CACHE_FILE = "geocode_cache.json"


def location_key(latitude, longitude):
    return f"{round(latitude, 3)},{round(longitude, 3)}"


def load_geocode_cache():
    if not os.path.exists(GEOCODE_CACHE_FILE):
        return {}
    try:
        with open(GEOCODE_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_geocode_cache(cache):
    try:
        with open(GEOCODE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print("Geocode cache save failed:", e)


def reverse_geocode(latitude, longitude):
    """
    Resolve coordinates into a human-readable place name using
    OpenStreetMap Nominatim. Falls back to a local cache entry
    (if this exact spot was looked up before) when offline, and
    to a generic "Custom Location" label as a last resort.
    """
    key = location_key(latitude, longitude)
    cache = load_geocode_cache()

    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": latitude,
            "lon": longitude,
            "format": "json",
            "zoom": 10,
            "accept-language": "en",  # force English names regardless of region
        }
        headers = {"User-Agent": "LandslideGuard-SIH-Prototype/1.0"}

        response = requests.get(url, params=params, headers=headers, timeout=6)
        response.raise_for_status()
        data = response.json()
        address = data.get("address", {})

        city = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("county")
            or address.get("state_district")
        )
        state = address.get("state")

        if city and state:
            name = f"{city}, {state}"
        elif state:
            name = state
        else:
            name = data.get("display_name", "Unknown location")

        cache[key] = name
        save_geocode_cache(cache)
        return name

    except Exception as e:
        print("Reverse geocoding failed (using cache/fallback):", e)
        cached_name = cache.get(key)
        if cached_name:
            return cached_name
        return f"Custom Location ({latitude:.4f}, {longitude:.4f})"


# ============================================================
# LOCATION SEARCH (forward geocoding: name/address -> lat/lon)
# ============================================================
# NEW: lets the user type any place name or address and jump
# straight to its coordinates, instead of only GPS / manual
# numeric entry / the 4 fixed preset buttons.
#
# Uses the same Nominatim service as reverse_geocode(), so no
# extra API key or dependency is needed. Also fills the same
# geocode cache keyed by rounded lat/lon, so once a place has
# been found it round-trips through reverse_geocode() offline
# too.

def search_locations(query, limit=8):
    """
    Autocomplete-style place search using the Photon geocoder
    (built on OpenStreetMap data, designed for partial/prefix
    text — unlike Nominatim's /search, which only matches whole
    words). Returns a list of dicts (latitude, longitude,
    display_name), best match first, up to `limit` results, so
    typing e.g. "visakha" already surfaces "Visakhapatnam".
    Returns an empty list if nothing was found or the request
    failed (e.g. offline).
    """
    try:
        url = "https://photon.komoot.io/api/"
        params = {
            "q": query,
            "limit": limit,
            "lang": "en",
        }
        headers = {"User-Agent": "LandslideGuard-SIH-Prototype/1.0"}

        response = requests.get(url, params=params, headers=headers, timeout=8)
        response.raise_for_status()
        data = response.json()

        matches = []
        cache = load_geocode_cache()

        for feature in data.get("features", []):
            props = feature.get("properties", {})
            coords = feature.get("geometry", {}).get("coordinates", [None, None])

            if len(coords) < 2 or coords[0] is None or coords[1] is None:
                continue

            found_lon, found_lat = float(coords[0]), float(coords[1])

            name_parts = [
                props.get("name"),
                props.get("state"),
                props.get("country"),
            ]
            found_name = ", ".join(part for part in name_parts if part)
            if not found_name:
                found_name = query

            matches.append({
                "latitude": found_lat,
                "longitude": found_lon,
                "display_name": found_name,
                "type": props.get("osm_value", ""),
            })

            # Warm the reverse-geocode cache too, so each of these
            # spots already has an offline-friendly name if picked.
            cache[location_key(found_lat, found_lon)] = found_name

        save_geocode_cache(cache)
        return matches

    except Exception as e:
        print("Location search failed:", e)
        return []


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f9f7;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}


/* ============================================================
   HEADER
   ============================================================ */

.main-header {
    background: linear-gradient(
        135deg,
        #0f766e,
        #0d9488
    );

    padding: 30px;

    border-radius: 18px;

    color: white;

    margin-bottom: 25px;

    box-shadow:
        0 5px 20px rgba(0,0,0,0.08);
}

.main-header h1 {
    margin: 0;
    font-size: 34px;
}

.main-header p {
    margin-top: 8px;
    font-size: 16px;
}


/* ============================================================
   CARDS
   ============================================================ */

.metric-card {

    background: white;

    padding: 22px;

    border-radius: 16px;

    border: 1px solid #dce7e3;

    box-shadow:
        0 3px 12px rgba(0,0,0,0.05);

    min-height: 135px;
}

.metric-title {

    font-size: 14px;

    color: #64748b;

    margin-bottom: 8px;
}

.metric-value {

    font-size: 30px;

    font-weight: 700;

    color: #0f172a;
}

.metric-sub {

    font-size: 13px;

    color: #64748b;

    margin-top: 5px;
}


/* ============================================================
   RISK CARDS
   ============================================================ */

.risk-card {

    background: white;

    padding: 25px;

    border-radius: 18px;

    border: 1px solid #dce7e3;

    box-shadow:
        0 3px 12px rgba(0,0,0,0.05);

    min-height: 180px;
}

.risk-score {

    font-size: 42px;

    font-weight: 800;

    margin-top: 10px;
}

.risk-level {

    font-size: 20px;

    font-weight: 700;

    margin-top: 5px;
}


/* ============================================================
   SECTION
   ============================================================ */

.section-title {

    font-size: 22px;

    font-weight: 700;

    color: #134e4a;

    margin-top: 28px;

    margin-bottom: 15px;
}


/* ============================================================
   INFO BOX
   ============================================================ */

.info-box {

    background: #ecfeff;

    border-left: 5px solid #0891b2;

    padding: 16px;

    border-radius: 10px;

    margin-bottom: 15px;
}


/* ============================================================
   ALERT
   ============================================================ */

.alert-box {

    background: #fff7ed;

    border-left: 5px solid #f97316;

    padding: 16px;

    border-radius: 10px;

    margin-bottom: 10px;
}


/* ============================================================
   GPS
   ============================================================ */

.gps-box {

    background: white;

    padding: 15px;

    border-radius: 12px;

    border: 1px solid #dce7e3;

    margin-bottom: 15px;
}


/* ============================================================
   ANALYZE AREA
   ============================================================ */

.analyze-box {

    background: white;

    padding: 22px;

    border-radius: 16px;

    border: 1px solid #dce7e3;

    box-shadow:
        0 3px 12px rgba(0,0,0,0.05);

    text-align: center;

    margin-top: 20px;

    margin-bottom: 25px;
}


/* ============================================================
   SIDEBAR BUTTONS
   ============================================================ */

[data-testid="stSidebar"] button {

    border-radius: 8px;

}


/* ============================================================
   FOOTER
   ============================================================ */

.footer-text {

    text-align: center;

    color: #64748b;

    font-size: 12px;

    padding-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.html("""
<div class="main-header">

    <h1>
        🌍 AI Disaster Risk Monitoring System
    </h1>

    <p>
        Global location-based monitoring for
        landslide and flood hazards
    </p>

</div>
""")


# ============================================================
# SESSION STATE
# ============================================================

if "latitude" not in st.session_state:
    st.session_state.latitude = 25.5941

if "longitude" not in st.session_state:
    st.session_state.longitude = 85.1376

if "selected_location" not in st.session_state:
    st.session_state.selected_location = "Patna, Bihar"

if "geocoded_key" not in st.session_state:
    # Tracks which lat/lon we last resolved a name for, so we
    # don't call the geocoding API again for the same spot.
    st.session_state.geocoded_key = location_key(
        st.session_state.latitude, st.session_state.longitude
    )

if "last_gps_lat" not in st.session_state:
    st.session_state.last_gps_lat = None

if "last_gps_lon" not in st.session_state:
    st.session_state.last_gps_lon = None

if "risk_data" not in st.session_state:
    st.session_state.risk_data = None

if "search_error" not in st.session_state:
    st.session_state.search_error = None

if "search_results" not in st.session_state:
    # List of candidate matches for the current search box text,
    # so the user can pick the right one before it's applied.
    st.session_state.search_results = []

if "last_searched_query" not in st.session_state:
    # Avoids re-querying the API on every rerun when the text
    # box content hasn't actually changed.
    st.session_state.last_searched_query = ""


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📍 Location"
)

st.sidebar.write(
    "Choose your location method."
)


# ============================================================
# SEARCH BY NAME / ADDRESS
# ============================================================
# NEW: primary, easiest way to set a location — type any place
# name or address (e.g. "Visakhapatnam", "Shillong, Meghalaya",
# "1600 Amphitheatre Parkway") and jump straight to it. Uses
# forward_geocode() (OpenStreetMap Nominatim) to resolve the
# text into latitude/longitude, then feeds those into the same
# session-state fields the GPS and manual inputs use.

st.sidebar.subheader(
    "🔎 Search Location"
)

search_query = st.sidebar.text_input(
    "Place name or address",
    placeholder="e.g. Visakha",
    key="search_query_input",
)

stripped_query = search_query.strip() if search_query else ""

# Live search: as soon as at least 3 characters are typed,
# look up matching places automatically (no separate "Search"
# button to click). Only re-queries the API when the text has
# actually changed since the last rerun.
if len(stripped_query) >= 3:

    if stripped_query != st.session_state.last_searched_query:

        with st.sidebar:
            with st.spinner("Searching..."):
                matches = search_locations(stripped_query)

        st.session_state.search_results = matches
        st.session_state.last_searched_query = stripped_query
        st.session_state.search_error = (
            None if matches else f"No matches found for '{stripped_query}'."
        )

elif stripped_query:
    st.sidebar.caption("Keep typing — at least 3 characters needed to search.")
    st.session_state.search_results = []
    st.session_state.last_searched_query = ""
    st.session_state.search_error = None

else:
    st.session_state.search_results = []
    st.session_state.last_searched_query = ""
    st.session_state.search_error = None

if st.session_state.search_error:
    st.sidebar.caption("⚪ " + st.session_state.search_error)

# ------------------------------------------------------------
# Show every matching city/place so the user can pick the right
# one (e.g. several towns can share the same name worldwide).
# ------------------------------------------------------------

if st.session_state.search_results:

    match_labels = [
        m["display_name"]
        for m in st.session_state.search_results
    ]

    chosen_idx = st.sidebar.radio(
        f"Found {len(match_labels)} match(es) — pick one:",
        options=range(len(match_labels)),
        format_func=lambda i: match_labels[i],
        key="search_match_choice",
    )

    if st.sidebar.button(
        "✅ Use this location",
        use_container_width=True,
        type="primary",
    ):
        chosen = st.session_state.search_results[chosen_idx]

        st.session_state.latitude = chosen["latitude"]
        st.session_state.longitude = chosen["longitude"]
        st.session_state.selected_location = chosen["display_name"]
        st.session_state.geocoded_key = location_key(
            chosen["latitude"], chosen["longitude"]
        )
        st.session_state.risk_data = None
        st.session_state.search_results = []
        st.session_state.search_error = None
        st.session_state.last_searched_query = ""

        st.rerun()

st.sidebar.markdown("---")


# ============================================================
# AUTOMATIC GPS
# ============================================================
# FIX (bug 2): the old version posted a custom "GPS_DATA"
# message from inside a components.html() iframe, but
# components.html() has no mechanism to receive that back into
# Python — nothing was ever listening for it, so the manual
# coordinate fields never updated.
#
# streamlit_geolocation() is a real Streamlit Component (built
# on the official Components JS protocol), so its return value
# comes back to Python normally, like any other widget.

st.sidebar.subheader(
    "📡 Automatic GPS"
)

gps_result = streamlit_geolocation()

st.sidebar.caption(
    "Click the location icon above and allow location "
    "permission in your browser when requested."
)

if gps_result and gps_result.get("latitude") is not None:
    new_lat = gps_result["latitude"]
    new_lon = gps_result["longitude"]

    if (
        new_lat != st.session_state.last_gps_lat
        or new_lon != st.session_state.last_gps_lon
    ):
        # A genuinely new GPS fix came in — push it into the
        # same session-state fields the manual inputs use, so
        # they pick it up automatically on this rerun.
        st.session_state.latitude = new_lat
        st.session_state.longitude = new_lon
        st.session_state.last_gps_lat = new_lat
        st.session_state.last_gps_lon = new_lon
        st.session_state.risk_data = None
        st.rerun()


# ============================================================
# MANUAL COORDINATES
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "✏️ Manual Coordinates"
)


latitude = st.sidebar.number_input(

    "Latitude",

    min_value=-90.0,

    max_value=90.0,

    value=float(
        st.session_state.latitude
    ),

    step=0.0001,

    format="%.6f"

)


longitude = st.sidebar.number_input(

    "Longitude",

    min_value=-180.0,

    max_value=180.0,

    value=float(
        st.session_state.longitude
    ),

    step=0.0001,

    format="%.6f"

)


# ============================================================
# UPDATE LOCATION + RESOLVE NAME
# ============================================================
# FIX (bug 1): whenever the effective coordinates change — from
# manual input, a preset button, GPS, or the search bar —
# resolve a real place name for them instead of leaving the old
# label in place.

st.session_state.latitude = latitude
st.session_state.longitude = longitude

current_key = location_key(latitude, longitude)

if current_key != st.session_state.geocoded_key:
    st.session_state.selected_location = reverse_geocode(latitude, longitude)
    st.session_state.geocoded_key = current_key


# ============================================================
# EXAMPLE LOCATIONS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "🌍 Example Locations"
)

st.sidebar.caption(
    "Click a location to load its coordinates."
)


# ============================================================
# PATNA
# ============================================================

if st.sidebar.button(
    "📍 Patna, Bihar",
    use_container_width=True
):

    st.session_state.latitude = 25.5941

    st.session_state.longitude = 85.1376

    st.session_state.risk_data = None

    st.rerun()


# ============================================================
# VISAKHAPATNAM
# ============================================================

if st.sidebar.button(
    "📍 Visakhapatnam, Andhra Pradesh",
    use_container_width=True
):

    st.session_state.latitude = 17.6868

    st.session_state.longitude = 83.2185

    st.session_state.risk_data = None

    st.rerun()


# ============================================================
# SHILLONG
# ============================================================

if st.sidebar.button(
    "📍 Shillong, Meghalaya",
    use_container_width=True
):

    st.session_state.latitude = 25.5788

    st.session_state.longitude = 91.8933

    st.session_state.risk_data = None

    st.rerun()


# ============================================================
# DARJEELING
# ============================================================

if st.sidebar.button(
    "📍 Darjeeling, West Bengal",
    use_container_width=True
):

    st.session_state.latitude = 27.0410

    st.session_state.longitude = 88.2663

    st.session_state.risk_data = None

    st.rerun()


# ============================================================
# GLOBAL LOCATION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.info(
    "🌍 You can enter coordinates for "
    "any location in the world."
)


# ============================================================
# API FUNCTION
# ============================================================

def get_complete_risk(
    latitude,
    longitude
):

    url = (
        f"{API_URL}/complete-risk"
    )


    # IMPORTANT:
    # FastAPI /complete-risk uses POST.

    response = requests.post(

        url,

        json={

            "latitude": latitude,

            "longitude": longitude

        },

        timeout=90

    )


    response.raise_for_status()


    return response.json()


# ============================================================
# RISK STYLE
# ============================================================

def risk_style(level):

    level = str(
        level
    ).upper()


    if level == "VERY HIGH":

        return (
            "🚨",
            "#b91c1c"
        )


    elif level == "HIGH":

        return (
            "⚠️",
            "#ea580c"
        )


    elif level == "MEDIUM":

        return (
            "🟡",
            "#ca8a04"
        )


    elif level == "LOW":

        return (
            "🟢",
            "#15803d"
        )


    else:

        return (
            "❔",
            "#64748b"
        )


# ============================================================
# MAIN PAGE LOCATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📍 Selected Location'
    '</div>',
    unsafe_allow_html=True
)


st.html(f"""
<div class="info-box">

<b>📍 Selected Location</b>

<br><br>

Latitude:
<b>{st.session_state.latitude:.6f}</b>

&nbsp;&nbsp;&nbsp;&nbsp;

Longitude:
<b>{st.session_state.longitude:.6f}</b>

<br><br>

<b>{st.session_state.selected_location}</b>

</div>
""")


# ============================================================
# ANALYZE BUTTON IN MIDDLE
# ============================================================

st.html("""
<div class="analyze-box">

    <h3 style="
        color:#134e4a;
        margin-bottom:8px;
    ">
        🔍 Analyze Selected Location
    </h3>

    <p style="
        color:#64748b;
        font-size:14px;
        margin-bottom:15px;
    ">
        The system automatically retrieves terrain,
        weather and river conditions and calculates
        landslide and flood risk.
    </p>

</div>
""")


analyze = st.button(

    "🔍 ANALYZE LOCATION",

    use_container_width=True,

    type="primary"

)


# ============================================================
# ANALYZE
# ============================================================

if analyze:

    with st.spinner(
        "Fetching live weather, terrain and flood data..."
    ):

        try:

            data = get_complete_risk(

                st.session_state.latitude,

                st.session_state.longitude

            )


            st.session_state.risk_data = data


            st.success(
                "✅ Location analyzed successfully."
            )


        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Could not connect to the FastAPI server."
            )

            st.info(
                "Make sure the FastAPI backend is running:"
            )

            st.code(
                "python app.py"
            )

            st.stop()


        except requests.exceptions.Timeout:

            st.error(
                "⏳ FastAPI request timed out."
            )

            st.stop()


        except requests.exceptions.HTTPError as e:

            st.error(
                "❌ FastAPI returned an HTTP error."
            )

            st.code(
                str(e)
            )

            st.stop()


        except Exception as e:

            st.error(
                "❌ Unexpected error."
            )

            st.code(
                str(e)
            )

            st.stop()


# ============================================================
# DATA
# ============================================================

data = st.session_state.risk_data


# ============================================================
# BEFORE ANALYSIS
# ============================================================

if data is None:

    st.info(
        "📍 Select a location and click "
        "**ANALYZE LOCATION**."
    )

    st.stop()


# ============================================================
# EXTRACT CURRENT FASTAPI RESPONSE
# ============================================================

location_raw = data.get(
    "location",
    {}
)

terrain = data.get(
    "terrain",
    {}
)

weather = data.get(
    "weather",
    {}
)

landslide_raw = data.get(
    "landslide_risk",
    {}
)

flood_raw = data.get(
    "flood_risk",
    {}
)

overall_raw = data.get(
    "overall_hazard",
    {}
)

alerts = data.get(
    "alerts",
    []
)

system = data.get(
    "system",
    {}
)


# ============================================================
# NORMALIZE LOCATION
# ============================================================

location = {

    "latitude":
        location_raw.get(
            "latitude",
            st.session_state.latitude
        ),

    "longitude":
        location_raw.get(
            "longitude",
            st.session_state.longitude
        ),

    "elevation_m":
        terrain.get(
            "elevation_m",
            0
        ),

    "slope_angle":
        terrain.get(
            "slope_angle",
            0
        ),

    "aspect":
        terrain.get(
            "aspect",
            0
        )

}


# ============================================================
# NORMALIZE LANDSLIDE
# ============================================================

landslide = {

    "score":
        landslide_raw.get(
            "risk_score",
            0
        ),

    "level":
        landslide_raw.get(
            "risk_level",
            "UNKNOWN"
        ),

    "model_score":
        landslide_raw.get(
            "model_score",
            0
        ),

    "condition_score":
        landslide_raw.get(
            "condition_score",
            0
        ),

    "model_supported":
        landslide_raw.get(
            "model_supported",
            False
        ),

    "condition_status":
        landslide_raw.get(
            "condition_status",
            ""
        )

}


# ============================================================
# NORMALIZE FLOOD
# ============================================================

flood = {

    "flood_score":
        flood_raw.get(
            "risk_score",
            0
        ),

    "flood_level":
        flood_raw.get(
            "risk_level",
            "UNKNOWN"
        ),

    "flood_condition":
        flood_raw.get(
            "flood_condition",
            ""
        ),

    "river_discharge_m3s":
        flood_raw.get(
            "current_discharge"
        ),

    "forecast_max_m3s":
        flood_raw.get(
            "forecast_max_discharge"
        ),

    "historical_mean":
        flood_raw.get(
            "historical_mean_discharge"
        ),

    "historical_p75":
        flood_raw.get(
            "historical_p75_discharge"
        ),

    "historical_p90":
        flood_raw.get(
            "historical_p90_discharge"
        ),

    "historical_p95":
        flood_raw.get(
            "historical_p95_discharge"
        ),

    "historical_max":
        flood_raw.get(
            "historical_max_discharge"
        ),

    "source":
        flood_raw.get(
            "source",
            "GloFAS / Open-Meteo"
        ),

    "data_status":
        flood_raw.get(
            "data_status",
            "UNKNOWN"
        )

}


# ============================================================
# NORMALIZE OVERALL
# ============================================================

overall = {

    "score":
        overall_raw.get(
            "risk_score",
            0
        ),

    "level":
        overall_raw.get(
            "risk_level",
            "UNKNOWN"
        )

}


# ============================================================
# DATA STATUS
# ============================================================

weather_status = weather.get(
    "data_status",
    "UNKNOWN"
)


if weather_status == "LIVE":

    st.success(
        "🟢 LIVE environmental data"
    )


elif weather_status == "CACHED_OFFLINE":

    st.warning(
        "🟠 Cached environmental data — "
        "internet connection unavailable"
    )


else:

    st.info(
        "⚪ Environmental data unavailable"
    )


# ============================================================
# LOCATION INFORMATION
# ============================================================

st.html(f"""

<div class="info-box">

<b>📍 Current Analyzed Location</b>

<br><br>

Latitude:
<b>{location["latitude"]:.6f}</b>

<br>

Longitude:
<b>{location["longitude"]:.6f}</b>

<br><br>

Elevation:
<b>{location["elevation_m"]:.1f} m</b>

<br>

Slope:
<b>{location["slope_angle"]:.2f}°</b>

</div>

""")


# ============================================================
# OVERALL HAZARD
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🚨 Overall Hazard Assessment'
    '</div>',
    unsafe_allow_html=True
)


overall_icon, overall_color = risk_style(
    overall["level"]
)


st.html(f"""

<div class="risk-card"
     style="
        border-left:7px solid {overall_color};
        text-align:center;
     ">

<div style="font-size:20px;">

{overall_icon}

<b>Overall Hazard</b>

</div>


<div class="risk-score"
     style="color:{overall_color};">

{float(overall["score"]):.1f}%

</div>


<div class="risk-level"
     style="color:{overall_color};">

{overall["level"]}

</div>


<div class="metric-sub">

Combined assessment of
landslide and flood hazards.

</div>

</div>

""")


# ============================================================
# LANDSLIDE + FLOOD
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🌋🌊 Hazard Assessment'
    '</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


# ============================================================
# LANDSLIDE
# ============================================================

with col1:

    icon, color = risk_style(
        landslide["level"]
    )


    st.html(f"""

    <div class="risk-card"
         style="
            border-left:7px solid {color};
         ">

    <div style="font-size:20px;">

    {icon}

    <b>🌋 Landslide Risk</b>

    </div>


    <div class="risk-score"
         style="color:{color};">

    {float(landslide["score"]):.1f}%

    </div>


    <div class="risk-level"
         style="color:{color};">

    {landslide["level"]}

    </div>


    <div class="metric-sub">

    Inputs:
    Rainfall + automatically calculated terrain slope

    </div>

    </div>

    """)


# ============================================================
# FLOOD
# ============================================================

with col2:

    icon, color = risk_style(
        flood["flood_level"]
    )


    st.html(f"""

    <div class="risk-card"
         style="
            border-left:7px solid {color};
         ">

    <div style="font-size:20px;">

    {icon}

    <b>🌊 Flood Risk</b>

    </div>


    <div class="risk-score"
         style="color:{color};">

    {float(flood["flood_score"]):.1f}%

    </div>


    <div class="risk-level"
         style="color:{color};">

    {flood["flood_level"]}

    </div>


    <div class="metric-sub">

    Global river-discharge indicator
    using GloFAS.

    </div>

    </div>

    """)


# ============================================================
# ENVIRONMENTAL CONDITIONS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🌦️ Environmental Conditions'
    '</div>',
    unsafe_allow_html=True
)


c1, c2, c3, c4 = st.columns(4)


# ============================================================
# CURRENT RAIN
# ============================================================

with c1:

    st.html(f"""

    <div class="metric-card">

    <div class="metric-title">

    🌧️ Current Rainfall

    </div>

    <div class="metric-value">

    {float(weather.get("rainfall_mm", 0)):.1f} mm

    </div>

    <div class="metric-sub">

    Current rainfall

    </div>

    </div>

    """)


# ============================================================
# 3 DAY
# ============================================================

with c2:

    st.html(f"""

    <div class="metric-card">

    <div class="metric-title">

    🌧️ 3-Day Rainfall

    </div>

    <div class="metric-value">

    {float(weather.get("rainfall_3day_mm", 0)):.1f} mm

    </div>

    <div class="metric-sub">

    Accumulated rainfall

    </div>

    </div>

    """)


# ============================================================
# 7 DAY
# ============================================================

with c3:

    st.html(f"""

    <div class="metric-card">

    <div class="metric-title">

    🌧️ 7-Day Rainfall

    </div>

    <div class="metric-value">

    {float(weather.get("rainfall_7day_mm", 0)):.1f} mm

    </div>

    <div class="metric-sub">

    Accumulated rainfall

    </div>

    </div>

    """)


# ============================================================
# SOIL MOISTURE
# ============================================================

with c4:

    st.html(f"""

    <div class="metric-card">

    <div class="metric-title">

    💧 Soil Moisture

    </div>

    <div class="metric-value">

    {float(weather.get("soil_moisture", 0)):.3f}

    </div>

    <div class="metric-sub">

    Volumetric soil moisture

    </div>

    </div>

    """)


# ============================================================
# TERRAIN + WEATHER
# ============================================================

st.markdown(
    '<div class="section-title">'
    '⛰️ Terrain & Weather'
    '</div>',
    unsafe_allow_html=True
)


d1, d2, d3, d4 = st.columns(4)


with d1:

    st.metric(

        "Elevation",

        f'{float(location["elevation_m"]):.1f} m'

    )


with d2:

    st.metric(

        "Slope",

        f'{float(location["slope_angle"]):.2f}°'

    )


with d3:

    st.metric(

        "Temperature",

        f'{float(weather.get("temperature_c", 0)):.1f} °C'

    )


with d4:

    st.metric(

        "Humidity",

        f'{float(weather.get("humidity_percent", 0)):.0f} %'

    )


# ============================================================
# ASPECT
# ============================================================

st.info(

    f'🧭 Terrain Aspect: '
    f'{float(location["aspect"]):.2f}°'

)


# ============================================================
# LANDSLIDE MODEL DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🤖 Landslide AI Details'
    '</div>',
    unsafe_allow_html=True
)


m1, m2, m3 = st.columns(3)


with m1:

    st.metric(

        "AI Model Score",

        f'{float(landslide["model_score"]):.2f}%'

    )


with m2:

    st.metric(

        "Condition Score",

        f'{float(landslide["condition_score"]):.2f}%'

    )


with m3:

    if landslide["model_supported"]:

        st.success(
            "AI Model Supported"
        )

    else:

        st.info(
            "Condition-Based Assessment"
        )


if landslide["condition_status"]:

    st.caption(
        landslide["condition_status"]
    )


# ============================================================
# FLOOD INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🌊 River / Flood Information'
    '</div>',
    unsafe_allow_html=True
)


f1, f2, f3 = st.columns(3)


# ============================================================
# CURRENT DISCHARGE
# ============================================================

with f1:

    discharge = flood[
        "river_discharge_m3s"
    ]


    if discharge is not None:

        st.metric(

            "Current River Discharge",

            f"{float(discharge):.2f} m³/s"

        )

    else:

        st.metric(

            "Current River Discharge",

            "N/A"

        )


# ============================================================
# FORECAST
# ============================================================

with f2:

    forecast = flood[
        "forecast_max_m3s"
    ]


    if forecast is not None:

        st.metric(

            "Maximum Forecast",

            f"{float(forecast):.2f} m³/s"

        )

    else:

        st.metric(

            "Maximum Forecast",

            "N/A"

        )


# ============================================================
# FLOOD LEVEL
# ============================================================

with f3:

    st.metric(

        "Flood Indicator",

        flood["flood_level"]

    )


# ============================================================
# FLOOD DESCRIPTION
# ============================================================

if flood["flood_condition"]:

    st.info(
        flood["flood_condition"]
    )


st.caption(
    "Flood indicator is derived from "
    "GloFAS river-discharge conditions."
)


# ============================================================
# ALERTS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🚨 Alerts'
    '</div>',
    unsafe_allow_html=True
)


if alerts:

    for alert in alerts:

        if (
            "No significant"
            in str(alert)
        ):

            st.success(
                "🟢 " + str(alert)
            )

        else:

            st.html(f"""

            <div class="alert-box">

            ⚠️ {alert}

            </div>

            """)

else:

    st.success(
        "🟢 No significant hazard detected."
    )


# ============================================================
# MAP
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🗺️ Risk Location'
    '</div>',
    unsafe_allow_html=True
)


map_df = pd.DataFrame({

    "latitude": [

        float(
            location["latitude"]
        )

    ],

    "longitude": [

        float(
            location["longitude"]
        )

    ]

})


st.map(

    map_df,

    latitude="latitude",

    longitude="longitude",

    zoom=7

)


# ============================================================
# MONITORING SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📊 Monitoring Summary'
    '</div>',
    unsafe_allow_html=True
)


summary = pd.DataFrame({

    "Parameter": [

        "Latitude",

        "Longitude",

        "Elevation",

        "Slope",

        "Current Rainfall",

        "3-Day Rainfall",

        "7-Day Rainfall",

        "Temperature",

        "Humidity",

        "Soil Moisture",

        "Landslide Risk",

        "Flood Risk",

        "Overall Hazard"

    ],


    "Value": [

        f'{float(location["latitude"]):.6f}',

        f'{float(location["longitude"]):.6f}',

        f'{float(location["elevation_m"]):.2f} m',

        f'{float(location["slope_angle"]):.2f}°',

        f'{float(weather.get("rainfall_mm", 0)):.2f} mm',

        f'{float(weather.get("rainfall_3day_mm", 0)):.2f} mm',

        f'{float(weather.get("rainfall_7day_mm", 0)):.2f} mm',

        f'{float(weather.get("temperature_c", 0)):.2f} °C',

        f'{float(weather.get("humidity_percent", 0)):.0f} %',

        f'{float(weather.get("soil_moisture", 0)):.3f}',

        f'{float(landslide["score"]):.2f} - '
        f'{landslide["level"]}',

        f'{float(flood["flood_score"]):.2f} - '
        f'{flood["flood_level"]}',

        f'{float(overall["score"]):.2f} - '
        f'{overall["level"]}'

    ]

})


st.dataframe(

    summary,

    use_container_width=True,

    hide_index=True

)


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'ℹ️ System Information'
    '</div>',
    unsafe_allow_html=True
)


st.html(f"""

<div class="info-box">

<b>🌋 Landslide Model:</b>

{system.get(
    "ml_model",
    "landslide_live_model.pkl"
)}

<br><br>

<b>🌊 Flood Model:</b>

GloFAS / Open-Meteo

<br><br>

<b>🌦️ Weather Source:</b>

{weather.get(
    "source",
    "Open-Meteo"
)}

<br><br>

<b>⛰️ Terrain Source:</b>

{terrain.get(
    "source",
    "Open-Meteo / Copernicus DEM"
)}

<br><br>

<b>🌍 System Mode:</b>

{system.get(
    "mode",
    "Global / Multi-Region"
)}

<br><br>

<b>📴 Offline Support:</b>

{
    "Enabled"
    if system.get(
        "offline_support",
        True
    )
    else
    "Disabled"
}

</div>

""")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer-text">

    AI-Based Early Warning & Disaster Risk Monitoring System |
    Global Prototype

    </div>
    """,
    unsafe_allow_html=True
)