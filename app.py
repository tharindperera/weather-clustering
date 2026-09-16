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

from dashboard_historical import (
    HISTORICAL_VARIABLES,
    VARIABLE_UNITS,
    aggregate_monthly,
    aggregate_yearly,
    eda_summary,
    filter_historical_data,
    load_historical_data,
    summarize_variable,
)

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


df = load_cluster_data()
eda_df = load_eda_features()
historical_df = load_historical_data()

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
    historical_variable_label = st.selectbox(
        "Select weather indicator",
        options=list(HISTORICAL_VARIABLES.keys()),
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
