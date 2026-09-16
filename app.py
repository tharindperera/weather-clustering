import sys
from pathlib import Path
import pandas as pd
import pydeck as pdk
import streamlit as st

# ============================================================================
# PATHS & IMPORTS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "weather-clustering") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "weather-clustering"))
if str(PROJECT_ROOT / "weather-clustering" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "weather-clustering" / "src"))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.dashboard_historical import (
    HISTORICAL_VARIABLES,
    VARIABLE_UNITS,
    aggregate_monthly,
    aggregate_yearly,
    eda_summary,
    filter_historical_data,
    load_historical_data,
    summarize_variable,
)
from src.realtime_weather import (
    get_current_weather_for_city,
)
from src.city_comparison import (
    compare_cities,
    find_most_similar_cities,
    load_city_comparison_data,
    standardize_weather_profiles,
)

# ============================================================================
# WMO WEATHER CODE TRANSLATIONS
# ============================================================================

WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

CLUSTER_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clustering"
    / "dashboard_clustering_dataset.parquet"
)
if not CLUSTER_DATA_FILE.exists():
    CLUSTER_DATA_FILE = (
        PROJECT_ROOT
        / "weather-clustering"
        / "data"
        / "processed"
        / "clustering"
        / "dashboard_clustering_dataset.parquet"
    )

EDA_FEATURE_FILE = (
    PROJECT_ROOT / "data" / "processed" / "features" / "climate_features.parquet"
)
if not EDA_FEATURE_FILE.exists():
    EDA_FEATURE_FILE = (
        PROJECT_ROOT
        / "weather-clustering"
        / "data"
        / "processed"
        / "features"
        / "climate_features.parquet"
    )

# ============================================================================
# CLUSTER DEFINITIONS
# ============================================================================

CLUSTER_NAMES = {
    0: "Hot, Wet & Humid",
    1: "Cool, Elevated / Interior",
    2: "Cooler, Windier / Mid-Latitude-Coastal",
    3: "Hot, Dry & Windy",
}

# RGBA values for PyDeck.
CLUSTER_COLORS = {
    0: [50, 150, 255, 220],
    1: [70, 200, 110, 220],
    2: [180, 110, 255, 220],
    3: [255, 85, 85, 230],
}

# Hex colors used by the legend.
CLUSTER_HEX_COLORS = {
    0: "#3296FF",
    1: "#46C86E",
    2: "#B46EFF",
    3: "#FF5555",
}

# ============================================================================
# EDA VARIABLE DEFINITIONS
# ============================================================================

EDA_DEFINITIONS = {
    "mean_temperature": (
        "Mean temperature",
        "°C",
        "10-year average daily mean temperature.",
    ),
    "temperature_std": (
        "Temperature variability",
        "°C",
        "Standard deviation of daily mean temperature.",
    ),
    "mean_diurnal_range": (
        "Mean diurnal temperature range",
        "°C",
        "Average daily difference between maximum and minimum temperature.",
    ),
    "temperature_p10": (
        "Temperature 10th percentile",
        "°C",
        "Temperature below which approximately 10% of daily values fall.",
    ),
    "temperature_p90": (
        "Temperature 90th percentile",
        "°C",
        "Temperature below which approximately 90% of daily values fall.",
    ),
    "temperature_p90_p10_range": (
        "Temperature P90-P10 range",
        "°C",
        "Difference between the 90th and 10th temperature percentiles.",
    ),
    "temperature_seasonality": (
        "Temperature seasonality",
        "°C",
        "Difference between the warmest and coldest monthly mean temperatures.",
    ),
    "annual_precipitation": (
        "Average annual precipitation",
        "mm/year",
        "Average annual precipitation over the 10-year historical period.",
    ),
    "precipitation_std": (
        "Precipitation variability",
        "mm/day",
        "Standard deviation of daily precipitation.",
    ),
    "wet_day_frequency": (
        "Wet-day frequency",
        "proportion",
        "Proportion of days classified as wet.",
    ),
    "maximum_daily_precipitation": (
        "Maximum daily precipitation",
        "mm/day",
        "Maximum observed daily precipitation.",
    ),
    "precipitation_seasonality": (
        "Precipitation seasonality",
        "mm",
        "Difference between maximum and minimum monthly precipitation.",
    ),
    "mean_humidity": (
        "Mean relative humidity",
        "%",
        "10-year average relative humidity.",
    ),
    "humidity_std": (
        "Humidity variability",
        "%",
        "Standard deviation of daily relative humidity.",
    ),
    "mean_wind": (
        "Mean wind speed",
        "km/h",
        "10-year average wind speed.",
    ),
    "wind_std": (
        "Wind-speed variability",
        "km/h",
        "Standard deviation of daily wind speed.",
    ),
}

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Weather Pattern Analytics",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# DATA LOADING
# ============================================================================


@st.cache_data
def load_cluster_data() -> pd.DataFrame:
    df = pd.read_parquet(CLUSTER_DATA_FILE)

    required_columns = [
        "location_id",
        "city",
        "country",
        "latitude",
        "longitude",
        "cluster_id",
        "temperature_c",
        "rainfall_mm_per_day",
        "humidity_percent",
        "wind_speed_kmh",
        "pressure_hpa",
    ]

    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Missing dashboard columns: {missing}")

    if len(df) != 100:
        raise ValueError(f"Expected 100 cities, found {len(df)}.")

    if df["city"].nunique() != 100:
        raise ValueError("Expected 100 unique cities.")

    return df


@st.cache_data
def load_eda_features() -> pd.DataFrame:
    df = pd.read_parquet(EDA_FEATURE_FILE)

    if len(df) != 100:
        raise ValueError(f"Expected 100 EDA city profiles, found {len(df)}.")

    if "city" not in df.columns:
        raise ValueError("EDA feature file does not contain 'city'.")

    return df


@st.cache_data
def load_comparison_data() -> pd.DataFrame:
    df = load_city_comparison_data()
    return standardize_weather_profiles(df)


df = load_cluster_data()
eda_df = load_eda_features()
historical_df = load_historical_data()
comparison_standardized_df = load_comparison_data()

# ============================================================================
# MERGE CLUSTER + EDA INFORMATION
# ============================================================================

location_details = df.merge(
    eda_df,
    on=["city"],
    how="left",
    suffixes=("", "_eda"),
    validate="one_to_one",
)

# ============================================================================
# SESSION STATE
# ============================================================================

if "selected_city" not in st.session_state:
    st.session_state.selected_city = None

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🌦️ Weather Explorer")
st.sidebar.markdown("Use the controls below to explore the 100 clustered locations.")
st.sidebar.divider()

# --------------------------------------------------------------------
# City search
# --------------------------------------------------------------------

st.sidebar.subheader("Search Location")

city_options = [f"{row.city} — {row.country}" for row in df.itertuples()]
search_options = ["None"] + city_options

current_city_label = "None"
if st.session_state.selected_city:
    selected_rows = df[df["city"] == st.session_state.selected_city]
    if not selected_rows.empty:
        row = selected_rows.iloc[0]
        current_city_label = f"{row['city']} — {row['country']}"

if current_city_label not in search_options:
    current_city_label = "None"

selected_search = st.sidebar.selectbox(
    "Select a city",
    options=search_options,
    index=search_options.index(current_city_label),
    help="Select a city to display its detailed weather and EDA profile.",
)

if selected_search != "None":
    selected_city_from_search = selected_search.split(" — ")[0]
    st.session_state.selected_city = selected_city_from_search

# --------------------------------------------------------------------
# Cluster filter
# --------------------------------------------------------------------

st.sidebar.subheader("Cluster Filter")

cluster_options = [
    f"Cluster {cluster_id} — {CLUSTER_NAMES[cluster_id]}"
    for cluster_id in sorted(CLUSTER_NAMES)
]

selected_cluster_labels = st.sidebar.multiselect(
    "Show clusters",
    options=cluster_options,
    default=cluster_options,
)

selected_cluster_ids = [
    int(label.split("—")[0].replace("Cluster", "").strip())
    for label in selected_cluster_labels
]

# ============================================================================
# FILTER DATA
# ============================================================================

filtered_df = df[df["cluster_id"].isin(selected_cluster_ids)].copy()

# ============================================================================
# HEADER
# ============================================================================

st.title("🌦️ Weather Pattern Analytics Dashboard")
st.markdown(
    """
Explore long-term weather patterns across **100 cities in 80 countries**. Locations were grouped using K-Means clustering based on five original Open-Meteo weather indicators: **temperature, rainfall, humidity, wind speed, and atmospheric pressure**.
"""
)

st.divider()

# ============================================================================
# METRICS
# ============================================================================

st.subheader("Selected Overview")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

with metric_1:
    st.metric(
        "Locations",
        len(filtered_df),
    )

with metric_2:
    st.metric(
        "Countries",
        filtered_df["country"].nunique(),
    )

with metric_3:
    st.metric(
        "Clusters Shown",
        filtered_df["cluster_id"].nunique(),
    )

with metric_4:
    if filtered_df.empty:
        st.metric(
            "Average Temperature",
            "—",
        )
    else:
        st.metric(
            "Average Temperature",
            f"{filtered_df['temperature_c'].mean():.1f} °C",
        )

# ============================================================================
# CLUSTER PROFILE TABLE
# ============================================================================

st.subheader("Weather Pattern Profiles")

if filtered_df.empty:
    st.warning("Select at least one cluster from the sidebar.")
else:
    profile = (
        filtered_df.groupby("cluster_id")
        .agg(
            locations=("city", "count"),
            temperature_c=("temperature_c", "mean"),
            rainfall_mm_per_day=(
                "rainfall_mm_per_day",
                "mean",
            ),
            humidity_percent=(
                "humidity_percent",
                "mean",
            ),
            wind_speed_kmh=(
                "wind_speed_kmh",
                "mean",
            ),
            pressure_hpa=(
                "pressure_hpa",
                "mean",
            ),
        )
        .reset_index()
    )

    profile["weather_pattern"] = profile["cluster_id"].map(CLUSTER_NAMES)

    profile = profile[
        [
            "cluster_id",
            "weather_pattern",
            "locations",
            "temperature_c",
            "rainfall_mm_per_day",
            "humidity_percent",
            "wind_speed_kmh",
            "pressure_hpa",
        ]
    ]

    profile = profile.rename(
        columns={
            "cluster_id": "Cluster",
            "weather_pattern": "Weather Pattern",
            "locations": "Locations",
            "temperature_c": "Temperature (°C)",
            "rainfall_mm_per_day": "Rainfall (mm/day)",
            "humidity_percent": "Humidity (%)",
            "wind_speed_kmh": "Wind Speed (km/h)",
            "pressure_hpa": "Pressure (hPa)",
        }
    )

    st.dataframe(
        profile.round(2),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================================
# CLUSTER LEGEND
# ============================================================================

st.subheader("Weather Pattern Legend")

legend_columns = st.columns(4)

for column, cluster_id in zip(
    legend_columns,
    sorted(CLUSTER_NAMES),
):
    color = CLUSTER_HEX_COLORS[cluster_id]

    with column:
        st.markdown(
            f"""
        <div style="
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 12px;
            padding: 12px 14px;
            background: rgba(255,255,255,0.035);
        ">
            <div style="
                display:flex;
                align-items:center;
                gap:10px;
                font-weight:600;
            ">
                <span style="
                    width:16px;
                    height:16px;
                    border-radius:50%;
                    display:inline-block;
                    background:{color};
                "></span>
                Cluster {cluster_id}
            </div>
            <div style="
                margin-top:6px;
                color:#B9C0CC;
                font-size:0.9rem;
            ">
                {CLUSTER_NAMES[cluster_id]}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ============================================================================
# INTERACTIVE MAP
# ============================================================================

st.subheader("Interactive Weather Pattern Map")

st.markdown(
    """
**Click any location marker** to select that city and view its detailed weather and EDA profile below. Hover over a marker for a quick summary.
"""
)

if filtered_df.empty:
    st.info("Select at least one cluster to display the map.")
else:
    map_df = filtered_df.copy()
    map_df["marker_color"] = map_df["cluster_id"].map(CLUSTER_COLORS)
    map_df["marker_radius"] = 30000
    map_df["cluster_name"] = map_df["cluster_id"].map(CLUSTER_NAMES)

    deck = pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=20,
            longitude=10,
            zoom=1.35,
            pitch=0,
            bearing=0,
            min_zoom=1.0,
            max_zoom=8.0,
            drag_rotate=False,
        ),
        tooltip={
            "html": """
            <div style="padding:8px;">
                <b>{city}, {country}</b><br/>
                <b>{cluster_name}</b><br/>
                Temperature: {temperature_c} °C<br/>
                Rainfall: {rainfall_mm_per_day} mm/day<br/>
                Humidity: {humidity_percent}%<br/>
                Wind: {wind_speed_kmh} km/h<br/>
                Pressure: {pressure_hpa} hPa
            </div>
            """,
            "style": {
                "backgroundColor": "#111827",
                "color": "white",
            },
        },
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                id="weather-locations",
                get_position=[
                    "longitude",
                    "latitude",
                ],
                get_radius="marker_radius",
                get_fill_color="marker_color",
                get_line_color=[
                    255,
                    255,
                    255,
                    220,
                ],
                line_width_min_pixels=1.5,
                radius_min_pixels=6,
                radius_max_pixels=24,
                stroked=True,
                filled=True,
                pickable=True,
                auto_highlight=True,
            )
        ],
    )

    map_col1, map_col2, map_col3 = st.columns([0.05, 0.90, 0.05])
    with map_col2:
        map_event = st.pydeck_chart(
            deck,
            height=585,
            selection_mode="single-object",
            on_select="rerun",
            key="weather_pattern_map",
        )

    # Process clicked marker
    try:
        selected_objects = map_event.selection.objects.get(
            "weather-locations",
            [],
        )
    except AttributeError:
        selected_objects = []

    if selected_objects:
        clicked_city = selected_objects[0].get("city")
        if clicked_city:
            st.session_state.selected_city = clicked_city

# ============================================================================
# SELECTED LOCATION DETAIL
# ============================================================================

st.divider()

st.subheader("Selected Location — Detailed Weather & EDA Profile")

selected_city = st.session_state.selected_city

if not selected_city:
    st.info("Select a city using the sidebar search or click a marker on the map.")
else:
    selected_rows = location_details[location_details["city"] == selected_city]

    if selected_rows.empty:
        st.warning(f"No detailed information was found for {selected_city}.")
    else:
        selected = selected_rows.iloc[0]
        cluster_id = int(selected["cluster_id"])

        # Location header
        title_col, cluster_col = st.columns([3, 1])

        with title_col:
            st.markdown(f"### 📍 {selected['city']}, {selected['country']}")
            st.caption(
                f"Location coordinates: "
                f"{selected['latitude']:.4f}°, "
                f"{selected['longitude']:.4f}°"
            )

        with cluster_col:
            st.markdown(
                f"""
            <div style="
                padding:12px;
                border-radius:12px;
                background:{CLUSTER_HEX_COLORS[cluster_id]}22;
                border:1px solid {CLUSTER_HEX_COLORS[cluster_id]};
            ">
                <b>Cluster {cluster_id}</b><br/>
                {CLUSTER_NAMES[cluster_id]}
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Original clustering variables
        st.markdown("#### Original Weather Indicators")

        original_metrics = st.columns(5)
        original_variables = [
            (
                "Temperature",
                selected["temperature_c"],
                "°C",
            ),
            (
                "Rainfall",
                selected["rainfall_mm_per_day"],
                "mm/day",
            ),
            (
                "Humidity",
                selected["humidity_percent"],
                "%",
            ),
            (
                "Wind speed",
                selected["wind_speed_kmh"],
                "km/h",
            ),
            (
                "Pressure",
                selected["pressure_hpa"],
                "hPa",
            ),
        ]

        for column, (
            label,
            value,
            unit,
        ) in zip(
            original_metrics,
            original_variables,
        ):
            with column:
                st.metric(
                    label,
                    f"{value:.2f}",
                    unit,
                )

        # Detailed EDA numeric profile
        st.markdown("#### Historical EDA Numerical Profile")

        eda_rows = []
        for column, (
            display_name,
            unit,
            description,
        ) in EDA_DEFINITIONS.items():
            if column not in selected.index:
                continue

            value = selected[column]

            if pd.isna(value):
                formatted_value = "N/A"
            else:
                formatted_value = f"{value:.3f}"

            eda_rows.append(
                {
                    "Variable": display_name,
                    "Value": formatted_value,
                    "Unit": unit,
                    "Meaning": description,
                }
            )

        eda_table = pd.DataFrame(eda_rows)

        st.dataframe(
            eda_table,
            use_container_width=True,
            hide_index=True,
        )

        # Direct comparison with cluster profile
        st.markdown("#### How This Location Compares With Its Cluster")

        cluster_profile = (
            df[df["cluster_id"] == cluster_id][
                [
                    "temperature_c",
                    "rainfall_mm_per_day",
                    "humidity_percent",
                    "wind_speed_kmh",
                    "pressure_hpa",
                ]
            ].mean()
        )

        comparison_rows = []
        for variable, display_name in [
            ("temperature_c", "Temperature"),
            ("rainfall_mm_per_day", "Rainfall"),
            ("humidity_percent", "Humidity"),
            ("wind_speed_kmh", "Wind speed"),
            ("pressure_hpa", "Pressure"),
        ]:
            location_value = selected[variable]
            cluster_value = cluster_profile[variable]
            difference = location_value - cluster_value

            comparison_rows.append(
                {
                    "Indicator": display_name,
                    "Location": location_value,
                    "Cluster average": cluster_value,
                    "Difference": difference,
                }
            )

        comparison_df = pd.DataFrame(comparison_rows)

        st.dataframe(
            comparison_df.round(2),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================================
# LOCATIONS TABLE
# ============================================================================

st.divider()

st.subheader("Locations in Selected Weather Patterns")

if filtered_df.empty:
    st.info("No locations selected.")
else:
    city_table = filtered_df[
        [
            "city",
            "country",
            "cluster_id",
            "temperature_c",
            "rainfall_mm_per_day",
            "humidity_percent",
            "wind_speed_kmh",
            "pressure_hpa",
        ]
    ].copy()

    city_table["weather_pattern"] = city_table["cluster_id"].map(CLUSTER_NAMES)

    city_table = city_table[
        [
            "city",
            "country",
            "weather_pattern",
            "temperature_c",
            "rainfall_mm_per_day",
            "humidity_percent",
            "wind_speed_kmh",
            "pressure_hpa",
        ]
    ]

    city_table = city_table.rename(
        columns={
            "city": "City",
            "country": "Country",
            "weather_pattern": "Weather Pattern",
            "temperature_c": "Temperature (°C)",
            "rainfall_mm_per_day": "Rainfall (mm/day)",
            "humidity_percent": "Humidity (%)",
            "wind_speed_kmh": "Wind Speed (km/h)",
            "pressure_hpa": "Pressure (hPa)",
        }
    )

    st.dataframe(
        city_table.sort_values(["Weather Pattern", "City"]).round(2),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================================
# METHODOLOGY
# ============================================================================

st.divider()

with st.expander("About the Weather Pattern Clustering Methodology"):
    st.markdown(
        """
### Clustering methodology
The project uses historical ERA5 weather data from Open-Meteo for **100 cities across 80 countries**, covering **2016–2025**.

The final K-Means model was implemented using **PySpark MLlib**. Five original Open-Meteo weather variables were used as clustering inputs:
1. Mean temperature
2. Mean daily precipitation
3. Mean relative humidity
4. Mean wind speed
5. Mean surface pressure

These five variables were standardized before K-Means clustering. The final model contains **four weather-pattern clusters (K=4)**.

The geographic coordinates are used only to visualize the resulting weather-pattern groups. They are **not** clustering variables.

The detailed EDA information shown for an individual location contains additional descriptive statistics derived from the historical dataset. Those descriptive statistics are used for interpretation and do not change the K-Means clustering assignments.
"""
    )

# ============================================================================
# HISTORICAL WEATHER EXPLORER
# ============================================================================

st.divider()

st.header("📈 Historical Weather Explorer")

st.markdown(
    """
Explore historical daily weather observations for any of the 100 cities
from **2016 to 2025**, then examine monthly patterns, yearly differences,
and statistical summaries.
"""
)

# ---------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------

historical_col1, historical_col2 = st.columns([1, 1])

with historical_col1:
    historical_cities = sorted(historical_df["city"].unique())

    default_index = 0
    if st.session_state.get("selected_city") in historical_cities:
        default_index = historical_cities.index(st.session_state.selected_city)
    elif "Colombo" in historical_cities:
        default_index = historical_cities.index("Colombo")

    historical_city = st.selectbox(
        "Select city",
        options=historical_cities,
        index=default_index,
        key="historical_city",
    )

with historical_col2:
    variable_options = list(HISTORICAL_VARIABLES.keys())
    default_var_index = 0
    if "Rainfall / Precipitation" in variable_options:
        default_var_index = variable_options.index("Rainfall / Precipitation")

    historical_variable_label = st.selectbox(
        "Select weather indicator",
        options=variable_options,
        index=default_var_index,
        key="historical_variable",
    )

selected_historical_variable = HISTORICAL_VARIABLES[historical_variable_label]
unit = VARIABLE_UNITS[selected_historical_variable]

# ---------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------

min_historical_date = historical_df["date"].min().date()
max_historical_date = historical_df["date"].max().date()

date_col1, date_col2 = st.columns([1, 1])

with date_col1:
    historical_start = st.date_input(
        "Start date",
        value=min_historical_date,
        min_value=min_historical_date,
        max_value=max_historical_date,
        key="historical_start",
    )

with date_col2:
    historical_end = st.date_input(
        "End date",
        value=max_historical_date,
        min_value=min_historical_date,
        max_value=max_historical_date,
        key="historical_end",
    )

# ---------------------------------------------------------------------
# Validate date range
# ---------------------------------------------------------------------

if historical_start > historical_end:
    st.error("Start date must be earlier than or equal to the end date.")
else:
    selected_historical = filter_historical_data(
        historical_df,
        city=historical_city,
        start_date=historical_start,
        end_date=historical_end,
    )

    if selected_historical.empty:
        st.warning("No historical observations were found for this selection.")
    else:
        # =============================================================
        # SECTION 1 — EDA SUMMARY
        # =============================================================

        st.subheader(f"EDA Summary — {historical_city}")

        summary = eda_summary(
            selected_historical,
            selected_historical_variable,
        )

        summary_columns = st.columns(4)

        with summary_columns[0]:
            st.metric(
                "Mean",
                f"{summary['mean']:.2f} {unit}",
            )

        with summary_columns[1]:
            st.metric(
                "Median",
                f"{summary['median']:.2f} {unit}",
            )

        with summary_columns[2]:
            st.metric(
                "Minimum",
                f"{summary['minimum']:.2f} {unit}",
            )

        with summary_columns[3]:
            st.metric(
                "Maximum",
                f"{summary['maximum']:.2f} {unit}",
            )

        eda_table = pd.DataFrame(
            {
                "Statistic": [
                    "Observations",
                    "Mean",
                    "Standard deviation",
                    "Minimum",
                    "25th percentile",
                    "Median",
                    "75th percentile",
                    "Maximum",
                    "Range",
                ],
                "Value": [
                    summary["count"],
                    summary["mean"],
                    summary["std"],
                    summary["minimum"],
                    summary["q1"],
                    summary["median"],
                    summary["q3"],
                    summary["maximum"],
                    summary["range"],
                ],
                "Unit": [
                    "days",
                    unit,
                    unit,
                    unit,
                    unit,
                    unit,
                    unit,
                    unit,
                    unit,
                ],
            }
        )

        with st.expander("View complete EDA statistics"):
            st.dataframe(
                eda_table.round(3),
                use_container_width=True,
                hide_index=True,
            )

        # =============================================================
        # SECTION 2 — DAILY TREND
        # =============================================================

        st.subheader(f"Daily Historical Trend — {historical_city}")

        daily_chart = selected_historical[
            ["date", selected_historical_variable]
        ].copy()

        daily_chart = daily_chart.set_index("date")

        st.line_chart(
            daily_chart,
            y=selected_historical_variable,
            x_label="Date",
            y_label=f"{historical_variable_label} ({unit})",
            height=420,
        )

        # =============================================================
        # SECTION 3 — MONTHLY AGGREGATION
        # =============================================================

        st.subheader("Monthly Aggregation")

        st.caption(
            "Rainfall is aggregated as monthly total precipitation. "
            "All other indicators are aggregated as monthly averages."
        )

        monthly_df = aggregate_monthly(
            selected_historical,
            selected_historical_variable,
        )

        if monthly_df.empty:
            st.info("Monthly aggregation is unavailable for this selection.")
        else:
            monthly_df = monthly_df.rename(
                columns={
                    "month": "Month",
                    selected_historical_variable: "Value",
                }
            )

            monthly_chart = monthly_df.set_index("Month")

            st.line_chart(
                monthly_chart,
                y="Value",
                x_label="Month",
                y_label=f"{historical_variable_label} ({'mm total' if selected_historical_variable == 'precipitation_sum' else unit})",
                height=400,
            )

            with st.expander("View monthly aggregated values"):
                st.dataframe(
                    monthly_df.round(3),
                    use_container_width=True,
                    hide_index=True,
                )

        # =============================================================
        # SECTION 4 — YEARLY COMPARISON
        # =============================================================

        st.subheader("Yearly Comparison")

        st.caption(
            "Rainfall is aggregated as annual total precipitation. "
            "All other indicators are annual averages."
        )

        yearly_df = aggregate_yearly(
            selected_historical,
            selected_historical_variable,
        )

        if yearly_df.empty:
            st.info("Yearly aggregation is unavailable for this selection.")
        else:
            yearly_df = yearly_df.rename(
                columns={
                    "year": "Year",
                    selected_historical_variable: "Value",
                }
            )

            yearly_chart = yearly_df.set_index("Year")

            st.bar_chart(
                yearly_chart,
                y="Value",
                x_label="Year",
                y_label=(
                    f"{historical_variable_label} "
                    f"({'mm total' if selected_historical_variable == 'precipitation_sum' else unit})"
                ),
                height=400,
            )

            with st.expander("View yearly values"):
                st.dataframe(
                    yearly_df.round(3),
                    use_container_width=True,
                    hide_index=True,
                )

            # Year-over-year comparison
            if len(yearly_df) >= 2:
                yearly_comparison = yearly_df.copy()

                yearly_comparison["Change from previous year"] = (
                    yearly_comparison["Value"].diff()
                )

                yearly_comparison["Percentage change"] = (
                    yearly_comparison["Value"].pct_change() * 100
                )

                with st.expander("Year-over-year changes"):
                    yoy_table = yearly_comparison.rename(
                        columns={
                            "Value": historical_variable_label,
                            "Change from previous year": "Absolute change",
                            "Percentage change": "Percentage change (%)",
                        }
                    )

                    st.dataframe(
                        yoy_table.round(3),
                        use_container_width=True,
                        hide_index=True,
                    )

        # =============================================================
        # SECTION 5 — DAILY DATA
        # =============================================================

        with st.expander("View daily historical observations"):
            display_columns = [
                "date",
                "temperature_mean",
                "temperature_max",
                "temperature_min",
                "precipitation_sum",
                "relative_humidity_mean",
                "wind_speed_mean",
                "surface_pressure_mean",
            ]

            historical_table = selected_historical[display_columns].copy()

            historical_table = historical_table.rename(
                columns={
                    "date": "Date",
                    "temperature_mean": "Mean Temperature (°C)",
                    "temperature_max": "Maximum Temperature (°C)",
                    "temperature_min": "Minimum Temperature (°C)",
                    "precipitation_sum": "Precipitation (mm)",
                    "relative_humidity_mean": "Humidity (%)",
                    "wind_speed_mean": "Wind Speed (km/h)",
                    "surface_pressure_mean": "Surface Pressure (hPa)",
                }
            )

            st.dataframe(
                historical_table.round(2),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            f"{len(selected_historical):,} daily observations "
            f"for {historical_city}."
        )

# ============================================================================
# CITY COMPARISON & WEATHER SIMILARITY
# ============================================================================

st.divider()

st.header("🔎 City Comparison & Weather Similarity")

st.markdown(
    """
Compare two project cities using the same five original weather variables
used by the K-Means clustering model. The similarity distance is calculated
in the standardized five-dimensional weather space.
"""
)

comparison_col1, comparison_col2 = st.columns(2)

project_cities = sorted(
    comparison_standardized_df["city"].unique()
)

with comparison_col1:
    city_a = st.selectbox(
        "Location A",
        options=project_cities,
        index=(
            project_cities.index("Colombo")
            if "Colombo" in project_cities
            else 0
        ),
        key="comparison_city_a",
    )

with comparison_col2:
    city_b = st.selectbox(
        "Location B",
        options=project_cities,
        index=(
            project_cities.index("Singapore")
            if "Singapore" in project_cities
            else 1
        ),
        key="comparison_city_b",
    )

if city_a == city_b:
    st.warning(
        "Please select two different cities."
    )
else:
    comparison = compare_cities(
        comparison_standardized_df,
        city_a,
        city_b,
    )

    # -----------------------------------------------------------------
    # Cluster relationship
    # -----------------------------------------------------------------

    relationship_col1, relationship_col2 = st.columns(2)

    with relationship_col1:
        cluster_a = comparison["cluster_a"]
        st.metric(
            f"{city_a} Cluster",
            f"Cluster {cluster_a}",
        )

    with relationship_col2:
        cluster_b = comparison["cluster_b"]
        st.metric(
            f"{city_b} Cluster",
            f"Cluster {cluster_b}",
        )

    if comparison["same_cluster"]:
        st.success(
            f"Both cities belong to the same weather-pattern cluster: "
            f"Cluster {cluster_a} — "
            f"{CLUSTER_NAMES[cluster_a]}"
        )
    else:
        st.info(
            f"The cities belong to different weather-pattern clusters: "
            f"{CLUSTER_NAMES[cluster_a]} vs "
            f"{CLUSTER_NAMES[cluster_b]}."
        )

    # -----------------------------------------------------------------
    # Weather-profile distance
    # -----------------------------------------------------------------

    distance_col1, distance_col2 = st.columns([1, 2])

    with distance_col1:
        st.metric(
            "Standardized Weather Distance",
            f"{comparison['standardized_distance']:.4f}",
        )

    with distance_col2:
        st.caption(
            """
            Smaller values indicate more similar five-variable
            weather profiles. The distance uses standardized
            temperature, rainfall, humidity, wind speed, and
            surface pressure.
            """
        )

    # -----------------------------------------------------------------
    # Direct comparison table
    # -----------------------------------------------------------------

    st.subheader("Weather Indicator Comparison")

    comparison_table = (
        comparison["variable_comparison"]
        .copy()
    )

    display_names = {
        "temperature_c": "Temperature",
        "rainfall_mm_per_day": "Rainfall",
        "humidity_percent": "Humidity",
        "wind_speed_kmh": "Wind Speed",
        "pressure_hpa": "Surface Pressure",
    }

    units = {
        "temperature_c": "°C",
        "rainfall_mm_per_day": "mm/day",
        "humidity_percent": "%",
        "wind_speed_kmh": "km/h",
        "pressure_hpa": "hPa",
    }

    comparison_table["Indicator"] = (
        comparison_table["variable"]
        .map(display_names)
    )

    comparison_table["Unit"] = (
        comparison_table["variable"]
        .map(units)
    )

    comparison_table = comparison_table[
        [
            "Indicator",
            "Unit",
            "city_a",
            "city_b",
            "difference",
            "absolute_difference",
        ]
    ]

    comparison_table.columns = [
        "Indicator",
        "Unit",
        city_a,
        city_b,
        "Difference",
        "Absolute Difference",
    ]

    st.dataframe(
        comparison_table.round(3),
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------------------
    # Difference drivers
    # -----------------------------------------------------------------

    st.subheader(
        "What Drives the Difference?"
    )

    contribution_df = (
        comparison["distance_contributions"]
        .copy()
    )

    contribution_df["Indicator"] = (
        contribution_df["variable"]
        .map(display_names)
    )

    contribution_df["Contribution (%)"] = (
        contribution_df["distance_contribution"]
        * 100
    )

    contribution_df = contribution_df[
        [
            "Indicator",
            "standardized_difference",
            "Contribution (%)",
        ]
    ]

    contribution_df.columns = [
        "Indicator",
        "Standardized Difference",
        "Contribution (%)",
    ]

    st.bar_chart(
        contribution_df.set_index(
            "Indicator"
        )[
            "Contribution (%)"
        ],
        height=320,
    )

    st.dataframe(
        contribution_df.round(3),
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------------------
    # Most similar cities to City A
    # -----------------------------------------------------------------

    st.subheader(
        f"Cities Most Similar to {city_a}"
    )

    similar_cities = find_most_similar_cities(
        comparison_standardized_df,
        city_a,
        top_n=5,
    )

    similar_cities["Weather Pattern"] = (
        similar_cities["cluster_id"]
        .map(CLUSTER_NAMES)
    )

    similar_cities = similar_cities[
        [
            "city",
            "country",
            "cluster_id",
            "Weather Pattern",
            "distance",
        ]
    ]

    similar_cities.columns = [
        "City",
        "Country",
        "Cluster",
        "Weather Pattern",
        "Distance",
    ]

    st.dataframe(
        similar_cities.round(4),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================================
# REAL-TIME WEATHER
# ============================================================================

st.divider()

st.header("🌤️ Real-Time Weather")

st.markdown(
    """
Retrieve the latest available weather conditions from the Open-Meteo API
for any city worldwide or select from the 100 project locations.
"""
)

search_mode = st.radio(
    "City Selection Mode",
    options=["100 Project Cities", "Custom Global City Search"],
    horizontal=True,
    key="realtime_search_mode",
)

realtime_col1, realtime_col2 = st.columns([3, 1])

with realtime_col1:
    if search_mode == "100 Project Cities":
        realtime_cities = sorted(historical_df["city"].unique())

        default_rt_index = 0
        if st.session_state.get("selected_city") in realtime_cities:
            default_rt_index = realtime_cities.index(st.session_state.selected_city)
        elif "Colombo" in realtime_cities:
            default_rt_index = realtime_cities.index("Colombo")

        realtime_city = st.selectbox(
            "Select city",
            options=realtime_cities,
            index=default_rt_index,
            key="realtime_city_dropdown",
        )
    else:
        realtime_city = st.text_input(
            "Enter any city name worldwide",
            value="Paris",
            placeholder="e.g. Paris, Sydney, Kandy, Galle, Chicago, Tokyo",
            key="realtime_city_custom",
        )

with realtime_col2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    realtime_search = st.button(
        "Get Current Weather",
        type="primary",
        use_container_width=True,
    )

if realtime_search:
    if not realtime_city.strip():
        st.warning("Please enter or select a city.")
    else:
        with st.spinner(
            f"Retrieving current weather for {realtime_city.strip()}..."
        ):
            try:
                realtime_result = get_current_weather_for_city(
                    realtime_city.strip()
                )

                location = realtime_result["location"]
                weather = realtime_result["weather"]

                weather_code = weather.get("weather_code")
                weather_description = WMO_WEATHER_CODES.get(
                    weather_code,
                    "Unknown condition",
                )

                # -----------------------------------------------------
                # Location heading
                # -----------------------------------------------------

                st.subheader(
                    f"{location['name']}, {location['country']}"
                )

                location_details = []

                if location.get("admin1"):
                    location_details.append(location["admin1"])

                location_details.append(
                    f"{location['latitude']:.4f}°, {location['longitude']:.4f}°"
                )

                location_details.append(location["timezone"])

                st.caption(" • ".join(location_details))

                # -----------------------------------------------------
                # Current weather cards
                # -----------------------------------------------------

                rt_col1, rt_col2, rt_col3, rt_col4, rt_col5 = st.columns(5)

                with rt_col1:
                    temperature = weather["temperature_c"]
                    st.metric(
                        "Temperature",
                        (
                            f"{temperature:.1f} °C"
                            if temperature is not None
                            else "N/A"
                        ),
                    )

                with rt_col2:
                    humidity = weather["humidity_percent"]
                    st.metric(
                        "Humidity",
                        (
                            f"{humidity:.0f}%"
                            if humidity is not None
                            else "N/A"
                        ),
                    )

                with rt_col3:
                    precipitation = weather["precipitation_mm"]
                    st.metric(
                        "Precipitation",
                        (
                            f"{precipitation:.1f} mm"
                            if precipitation is not None
                            else "N/A"
                        ),
                    )

                with rt_col4:
                    wind = weather["wind_speed_kmh"]
                    st.metric(
                        "Wind Speed",
                        (
                            f"{wind:.1f} km/h"
                            if wind is not None
                            else "N/A"
                        ),
                    )

                with rt_col5:
                    pressure = weather["pressure_hpa"]
                    st.metric(
                        "Surface Pressure",
                        (
                            f"{pressure:.1f} hPa"
                            if pressure is not None
                            else "N/A"
                        ),
                    )

                # -----------------------------------------------------
                # Current condition details
                # -----------------------------------------------------

                st.markdown("#### Current Conditions")

                condition_col1, condition_col2 = st.columns(2)

                with condition_col1:
                    st.metric(
                        "Condition",
                        weather_description,
                    )

                with condition_col2:
                    day_status = weather.get("is_day")
                    if day_status == 1:
                        day_text = "Day"
                    elif day_status == 0:
                        day_text = "Night"
                    else:
                        day_text = "N/A"

                    st.metric(
                        "Day / Night",
                        day_text,
                    )

                # -----------------------------------------------------
                # Observation time
                # -----------------------------------------------------

                st.caption(
                    "Open-Meteo local observation/model time: "
                    f"{weather.get('time', 'N/A')} "
                    f"({weather.get('timezone', location.get('timezone', 'local time'))})"
                )

                # -----------------------------------------------------
                # Raw real-time details
                # -----------------------------------------------------

                with st.expander("View API weather details"):
                    realtime_table = pd.DataFrame(
                        {
                            "Indicator": [
                                "Temperature",
                                "Relative humidity",
                                "Precipitation",
                                "Wind speed",
                                "Surface pressure",
                                "Weather condition",
                                "WMO code",
                                "Day / night",
                            ],
                            "Value": [
                                (
                                    f"{weather.get('temperature_c')} °C"
                                    if weather.get("temperature_c") is not None
                                    else "N/A"
                                ),
                                (
                                    f"{weather.get('humidity_percent')}%"
                                    if weather.get("humidity_percent") is not None
                                    else "N/A"
                                ),
                                (
                                    f"{weather.get('precipitation_mm')} mm"
                                    if weather.get("precipitation_mm") is not None
                                    else "N/A"
                                ),
                                (
                                    f"{weather.get('wind_speed_kmh')} km/h"
                                    if weather.get("wind_speed_kmh") is not None
                                    else "N/A"
                                ),
                                (
                                    f"{weather.get('pressure_hpa')} hPa"
                                    if weather.get("pressure_hpa") is not None
                                    else "N/A"
                                ),
                                str(weather_description),
                                str(weather.get("weather_code", "N/A")),
                                str(day_text),
                            ],
                        }
                    )

                    st.dataframe(
                        realtime_table,
                        use_container_width=True,
                        hide_index=True,
                    )

            except Exception as exc:
                st.error(f"Unable to retrieve weather data: {exc}")
                st.info(
                    "Check the city name and your internet connection, "
                    "then try again."
                )



