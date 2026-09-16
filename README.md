# Weather Pattern Analytics — Open-Meteo Big Data Project

> **DS 4004 — Big Data Analytics | Group C | Option 6: Weather Pattern Clustering and Grouping**

A reproducible, cloud-deployed weather analytics system combining a frozen ERA5 climate baseline, automatically refreshed ECMWF IFS recent weather, real-time Open-Meteo data, Spark MLlib clustering, city similarity analysis, Hugging Face storage, GitHub Actions automation, and a public Streamlit dashboard.

## Live Project Links

- **Public Streamlit dashboard:** https://weather-clustering-and-pattern-analytics.streamlit.app/
- **GitHub repository:** https://github.com/tharindperera/weather-clustering
- **Hugging Face dataset:** https://huggingface.co/datasets/tharinduperera/weather-clustering-data

## Table of Contents

1. [Project Overview](#project-overview)
2. [Assignment Scope](#assignment-scope)
3. [Final Architecture](#final-architecture)
4. [Why the Project Uses Three Weather Layers](#why-the-project-uses-three-weather-layers)
5. [Location Coverage](#location-coverage)
6. [Weather Variables](#weather-variables)
7. [Technology Stack](#technology-stack)
8. [Historical ERA5 Ingestion](#historical-era5-ingestion)
9. [Recent ECMWF IFS Layer](#recent-ecmwf-ifs-layer)
10. [Real-Time Weather Layer](#real-time-weather-layer)
11. [Storage Design](#storage-design)
12. [Exploratory Data Analysis](#exploratory-data-analysis)
13. [Derived Climate Features](#derived-climate-features)
14. [Clustering Methodology](#clustering-methodology)
15. [Why Derived Variables Were Not Used for K-Means](#why-derived-variables-were-not-used-for-k-means)
16. [K-Means Model Selection](#k-means-model-selection)
17. [Final K=4 Weather Pattern Groups](#final-k4-weather-pattern-groups)
18. [PCA Visualization](#pca-visualization)
19. [City Comparison and Weather Similarity](#city-comparison-and-weather-similarity)
20. [Streamlit Dashboard](#streamlit-dashboard)
21. [Online Automation](#online-automation)
22. [Hugging Face Data Publishing](#hugging-face-data-publishing)
23. [Streamlit Community Cloud Deployment](#streamlit-community-cloud-deployment)
24. [Project Directory Structure](#project-directory-structure)
25. [Installation](#installation)
26. [Running the Dashboard Locally](#running-the-dashboard-locally)
27. [Reproducing the Data Pipelines](#reproducing-the-data-pipelines)
28. [Quality Assurance and Validation](#quality-assurance-and-validation)
29. [Failure Recovery and Idempotency](#failure-recovery-and-idempotency)
30. [Security and Secrets](#security-and-secrets)
31. [Important Methodological Decisions](#important-methodological-decisions)
32. [Known Limitations](#known-limitations)
33. [Dynamic Data Notes](#dynamic-data-notes)
34. [Reproducibility Summary](#reproducibility-summary)
35. [Final Project Status](#final-project-status)

# Project Overview

The project analyses historical, recent, and real-time weather for a globally distributed set of cities and identifies groups of locations with similar long-term weather behaviour.

Rather than a single notebook, the final system is a complete data-engineering and analytics pipeline. It includes weather-data acquisition from Open-Meteo, restart-safe API ingestion, raw JSON persistence, analytical Parquet storage, DuckDB/pandas EDA, Spark MLlib K-Means clustering, city-level similarity analysis, interactive Streamlit/PyDeck visualization, public Hugging Face storage, GitHub Actions automation, Streamlit Community Cloud deployment, and dedicated reproducibility, provenance, data-quality, and failure-recovery QA.

The final design deliberately separates **long-term climate modelling**, **recent historical/operational weather**, and **real-time conditions** so that each layer has a clear methodological role.

# Assignment Scope

## General Requirements Covered

The system connects to Open-Meteo APIs, retrieves historical and current weather data, performs EDA on temperature, rainfall, humidity, wind speed, and atmospheric pressure, stores data efficiently, exposes an interactive web dashboard, and accepts user inputs for weather insights.

## Group C — Option 6

The team-specific objective is to retrieve historical weather for multiple locations, group locations using similar weather patterns, compare locations, and visualize the resulting groups.

The final solution clusters 100 globally distributed cities and exposes both cluster membership and pairwise weather similarity in the dashboard.

# Final Architecture

```text
                                Open-Meteo
                                     |
                 +-------------------+-------------------+
                 |                   |                   |
                 v                   v                   v
          ERA5 Historical      ECMWF IFS Recent     Current Forecast
            2016–2025             2026+              Real-Time
                 |                   |                   |
                 v                   v                   |
             Raw JSON          GitHub Actions            |
                 |                   |                   |
                 v                   v                   |
         Partitioned Parquet   Incremental Refresh       |
                 |                   |                   |
                 +-----------> Hugging Face <------------+
                                     |
                                     v
                          Streamlit Community Cloud
                                     |
                                     v
                            Public Dashboard
```

Official clustering remains frozen:

```text
ERA5 2016–2025
      ↓
daily observations
      ↓
10-year city means
      ↓
5 original weather indicators
      ↓
Spark StandardScaler
      ↓
Spark MLlib K-Means
      ↓
Final K = 4
```

The 2026+ IFS layer is not automatically added to or used to retrain the clustering model.

# Why the Project Uses Three Weather Layers

## ERA5 Climate Baseline — 2016–2025

Purpose: long-term climate analysis, historical EDA, and the official reproducible K=4 clustering baseline.

Properties: fixed date range, frozen after validation, used for K-Means, not modified by daily automation.

## ECMWF IFS Recent Layer — 2026 to Latest Complete Day

Purpose: extend the dashboard beyond the frozen climate baseline and provide recent historical/operational weather.

Properties: accessed through the Open-Meteo Historical Forecast API, refreshed incrementally, persisted as Parquet, published to Hugging Face, updated daily by GitHub Actions, and not used to automatically retrain K-Means.

## Real-Time Weather Layer

Purpose: show current weather for the 100 project cities and arbitrary user-searched global cities.

Properties: requested live through Open-Meteo and not permanently stored in the analytical climate datasets.

# Location Coverage

The final project uses **100 cities across 80 countries**.

Location catalogue:

```text
data/locations/locations.csv
```

Schema:

```text
location_id, city, city_ascii, country, iso2, iso3, admin_name,
capital, latitude, longitude, population
```

Country distribution:

```text
67 countries represented by one city
6 countries represented by two cities
7 countries represented by three cities
```

Validated catalogue SHA-256:

```text
5CBA155270694BB743E8ED4E95D3F6A95F135D4AA61CAF5B705CF13566F8BD19
```

## Why 100 Cities Instead of 500

An earlier design considered 500 cities. The production architecture was intentionally reduced to 100 because the assignment focuses on weather-pattern grouping and comparative analytics rather than maximum geographic volume.

The 100-city design provides global coverage, manageable API usage, faster Spark experimentation, easier validation, practical dashboard interaction, lower cloud-storage overhead, and enough observations for interpretable city-level clustering.

The 500-city design is obsolete and is not part of the final production pipeline.

# Weather Variables

Seven daily variables are collected for the historical layers.

| Open-Meteo Variable | Project Column | Role |
|---|---|---|
| `temperature_2m_mean` | `temperature_mean` | Clustering + EDA |
| `temperature_2m_max` | `temperature_max` | EDA |
| `temperature_2m_min` | `temperature_min` | EDA |
| `precipitation_sum` | `precipitation_sum` | Clustering + EDA |
| `relative_humidity_2m_mean` | `relative_humidity_mean` | Clustering + EDA |
| `wind_speed_10m_mean` | `wind_speed_mean` | Clustering + EDA |
| `surface_pressure_mean` | `surface_pressure_mean` | Clustering + EDA |

The official K-Means model uses only five original weather indicators: mean temperature, precipitation, relative humidity, wind speed, and surface pressure.

# Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Weather provider | Open-Meteo | Historical, recent, and live weather access |
| Long-term climate source | ERA5 via Open-Meteo | 2016–2025 baseline |
| Recent source | ECMWF IFS via Open-Meteo | 2026+ recent layer |
| Raw storage | JSON | Preserve API responses |
| Analytical storage | Apache Parquet | Efficient columnar analytics |
| SQL / EDA | DuckDB | Fast analytical queries |
| Dataframes | pandas | Transformation and dashboard preparation |
| Big-data clustering | Apache Spark MLlib | StandardScaler + K-Means |
| PCA/supporting analysis | scikit-learn | Visualization only |
| Visualization | Streamlit, PyDeck, Matplotlib | Interactive analysis |
| Cloud dataset store | Hugging Face Datasets | Persistent public data |
| Automation | GitHub Actions | Daily recent-data refresh |
| Deployment | Streamlit Community Cloud | Public web hosting |
| Source control | Git / GitHub | Versioning and workflow source |

# Historical ERA5 Ingestion

Experiment:

```text
era5_100cities_2016-01-01_2025-12-31
```

Date range:

```text
2016-01-01 → 2025-12-31
```

Validated totals:

```text
Cities:                 100
Days per city:          3,653
Total observations:     365,300
Location batches:       5
Cities per request:     20
Window length:          14 days
Date windows:           261
Total work units:       1,305
Successful work units:  1,305
Failed work units:      0
Request delay:          5 seconds
Maximum retries:        5
Request timeout:        120 seconds
Approx. raw JSON size:  66.87 MB
```

Open-Meteo supports multiple locations in one request, so one 20-city batch counts as one HTTP request, not 20 independent requests.

## Restart Safety

A work unit is marked successful only after the response is received, schema validation passes, raw JSON is persisted successfully, and the checkpoint is updated. This prevents a failed write from being incorrectly marked complete.

# Recent ECMWF IFS Layer

The recent layer begins at `2026-01-01`.

Endpoint:

```text
https://historical-forecast-api.open-meteo.com/v1/forecast
```

Model:

```text
ecmwf_ifs025
```

The same seven daily variables used in the historical ERA5 pipeline are requested.

Initial validated backfill:

```text
Period:          2026-01-01 → 2026-09-15
Days per city:   258
Cities:          100
Rows:            25,800
Work units:      95
Failed units:    0
```

## Incremental Refresh

The production refresh does not re-download the entire 2026 dataset.

```text
existing maximum date
        ↓
latest complete UTC day
        ↓
calculate missing range
        ↓
fetch only missing dates
        ↓
validate
        ↓
upsert
        ↓
atomic Parquet replacement
```

If no new complete date exists, the refresh exits with **zero Open-Meteo requests**.

If multiple days were missed, the next successful execution catches up all missing dates.

# Real-Time Weather Layer

Geocoding endpoint:

```text
https://geocoding-api.open-meteo.com/v1/search
```

Forecast/current endpoint:

```text
https://api.open-meteo.com/v1/forecast
```

Current indicators include:

```text
temperature_2m
relative_humidity_2m
precipitation
wind_speed_10m
surface_pressure
weather_code
is_day
```

The dashboard supports project-city selection and arbitrary global city search. It displays five core metrics, readable WMO weather condition, day/night state, timestamp, timezone, and location metadata.

Real-time data is intentionally not permanently appended to the ERA5 or IFS analytical datasets.

# Storage Design

## Raw JSON

Raw historical responses are retained under local ingestion directories to support reproducibility, debugging, schema inspection, and reprocessing without repeating API calls.

## ERA5 Analytical Parquet

Production analytical storage:

```text
data/processed/parquet/weather/
```

Validated properties:

```text
Rows:       365,300
Files:      1,350
Partitions: year=2016 ... year=2025
```

Leap-year partitions contain 36,600 rows; non-leap years contain 36,500.

## Dashboard-Optimized ERA5

A consolidated public dashboard-serving copy is published to Hugging Face:

```text
processed/dashboard/weather_era5_2016_2025.parquet
```

Validated:

```text
Rows:        365,300
Cities:      100
Date range:  2016-01-01 → 2025-12-31
Size:        ~2.89 MB
Compression: ZSTD
```

The original 1,350-file partitioned dataset remains untouched.

## Recent IFS Parquet

Published path:

```text
processed/parquet/weather_ifs_2026-present/year=2026/weather.parquet
```

This file is updated by the automated online workflow.

# Exploratory Data Analysis

DuckDB was used over the validated ERA5 Parquet dataset.

Selected global daily statistics:

| Variable | Mean | Std. Dev. | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Mean temperature | 20.74 °C | 8.57 | -27.7 | 43.6 |
| Max temperature | 25.30 °C | — | -24.9 | 51.3 |
| Min temperature | 16.80 °C | — | -30.0 | 38.1 |
| Precipitation | 2.91 mm/day | 7.25 | 0 | 301.3 |
| Relative humidity | 70.47% | 17.00 | 5 | 100 |
| Wind speed | 9.79 km/h | 5.14 | 1.2 | 55.2 |
| Surface pressure | 973.22 hPa | 67.90 | 662.8 | 1051.8 |

Examples from long-term city summaries:

```text
Coldest mean temperature: Saint Petersburg, Moscow, Toronto
Hottest:                  Khartoum, Dubai, Ouagadougou
Wettest annual average:   Bandung, Douala, Kuala Lumpur, Singapore, Colombo
Driest annual average:    Cairo, Khartoum, Riyadh, Alexandria, Dubai
```

# Derived Climate Features

File:

```text
data/processed/features/climate_features.parquet
```

It contains 100 city rows and 16 derived numeric features:

```text
mean_temperature
temperature_std
mean_diurnal_range
temperature_p10
temperature_p90
temperature_p90_p10_range
temperature_seasonality
annual_precipitation
precipitation_std
wet_day_frequency
maximum_daily_precipitation
precipitation_seasonality
mean_humidity
humidity_std
mean_wind
wind_std
```

These features are used for EDA and descriptive city profiles, not K-Means.

Near-redundancies found during diagnostics included:

```text
temperature_std ↔ temperature_p90_p10_range    0.999
temperature_p90_p10_range ↔ seasonality        0.997
temperature_std ↔ seasonality                  0.996
```

# Clustering Methodology

The model clusters city-level climate profiles, not daily rows.

```text
3,653 daily observations per city
            ↓
10-year mean aggregation
            ↓
one row per city
            ↓
100 rows × 5 variables
```

City-level clustering input summary:

| Indicator | Mean | Std. Dev. | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Temperature | 20.739 °C | 6.245 | 6.384 | 30.392 |
| Precipitation | 2.909 mm/day | 2.061 | 0.063 | 10.125 |
| Humidity | 70.471% | 12.217 | 26.946 | 88.274 |
| Wind speed | 9.786 km/h | 3.404 | 3.522 | 20.521 |
| Surface pressure | 973.222 hPa | 68.039 | 668.550 | 1015.786 |

Spark MLlib `StandardScaler` is applied before K-Means so that different physical units do not dominate Euclidean distance.

Because K-Means is unsupervised, the project does not use a conventional supervised train/test split. All 100 cities receive final assignments.

# Why Derived Variables Were Not Used for K-Means

This was deliberate. Using the five original indicators keeps the model directly aligned with the assignment, reduces redundancy, avoids overweighting temperature through many correlated derivatives, and preserves interpretability in physical units.

Only one temperature measure—mean temperature—is included in clustering.

# K-Means Model Selection

Apache Spark MLlib evaluated `K=2` through `K=8`.

| K | WSSSE / Training Cost | Silhouette | Cluster Sizes |
|---:|---:|---:|---|
| 2 | 351.709268 | 0.391259 | 67, 33 |
| 3 | 265.413684 | 0.454907 | 33, 20, 47 |
| 4 | 202.844504 | **0.517870** | 36, 20, 34, 10 |
| 5 | 169.512565 | 0.473226 | 31, 10, 23, 26, 10 |
| 6 | 154.865950 | 0.389312 | — |
| 7 | 143.741603 | 0.380157 | minimum cluster size 3 |
| 8 | 134.529176 | 0.388802 | minimum cluster size 3 |

`K=4` was selected because it produced the highest silhouette score, a sensible training-cost reduction, interpretable groups, and no extremely small clusters.

Seed-stability checks:

| Seed | Silhouette | WSSSE |
|---:|---:|---:|
| 42 | **0.517870** | **202.8445** |
| 100 | 0.496368 | 207.3142 |
| 2024 | 0.515486 | 203.5302 |
| 777 | 0.501684 | — |
| 999 | 0.501684 | — |

Seed `42` is used in the final model.

# Final K=4 Weather Pattern Groups

These are descriptive weather-pattern groups, not formal climate classifications.

## Cluster 0 — Hot, Wet & Humid

```text
Cities:           36
Temperature:      26.09 °C
Precipitation:    4.81 mm/day
Humidity:         79.12%
Wind:             8.05 km/h
Pressure:         1000.11 hPa
```

Representative cities: Colombo, Singapore, Jakarta, Bangkok, Mumbai.

## Cluster 1 — Cool / Elevated / Interior

```text
Cities:           20
Temperature:      17.19 °C
Precipitation:    2.00 mm/day
Humidity:         61.21%
Wind:             7.54 km/h
Pressure:         863.01 hPa
```

Representative cities: La Paz, Bogotá, Mexico City, Addis Ababa, Nairobi, Kabul.

The pressure profile is strongly influenced by high-altitude locations.

## Cluster 2 — Cooler, Windier / Mid-Latitude-Coastal

```text
Cities:           34
Temperature:      15.18 °C
Precipitation:    2.09 mm/day
Humidity:         72.87%
Wind:             11.76 km/h
Pressure:         1004.20 hPa
```

Representative cities: London, Paris, Berlin, New York, Tokyo, Sydney, Moscow.

## Cluster 3 — Hot, Dry & Windy / Hot Arid

```text
Cities:           10
Temperature:      27.50 °C
Precipitation:    0.69 mm/day
Humidity:         49.70%
Wind:             13.81 km/h
Pressure:         991.50 hPa
```

Representative cities: Dubai, Riyadh, Cairo, Baghdad, Khartoum.

Final cluster sizes:

```text
Cluster 0: 36
Cluster 1: 20
Cluster 2: 34
Cluster 3: 10
Total:    100
```

# PCA Visualization

PCA is used only for visualization after clustering.

```text
PC1: 39.54%
PC2: 28.00%
PC1 + PC2: 67.54%
```

PCA is not used as K-Means input.

Latitude and longitude are also not clustering features; the geographic map is visualization-only.

# City Comparison and Weather Similarity

The dashboard compares cities in the same five standardized dimensions used by K-Means.

Standardized Weather Distance:

```text
D(A,B) = sqrt( Σ (z_Aj - z_Bj)^2 )
```

Standardization uses population standard deviation (`ddof=0`) across the 100-city comparison table.

The dashboard shows raw values, raw and absolute differences, same/different cluster status, standardized distance, contribution of each weather dimension, and the five nearest weather profiles.

Validated example:

```text
Colombo vs Singapore
Distance:     0.3998
Same cluster: Yes
```

Approximate driver contributions:

```text
Rainfall:     51.95%
Humidity:     42.74%
Wind:          4.34%
Temperature:   0.96%
Pressure:      0.01%
```

The metric is intentionally called **distance**, not a percentage similarity.

# Streamlit Dashboard

Public URL:

```text
https://weather-clustering-and-pattern-analytics.streamlit.app/
```

## Real-Time Weather

Project-city selector, custom global search, five weather metrics, readable condition, day/night state, timezone, and metadata.

## Recent Weather

Reads ECMWF IFS Parquet from Hugging Face, shows latest complete daily values, five independent time-series charts, and a daily table.

The button `🔄 Reload Latest Published Data` clears the Streamlit cache and reloads the current published Hugging Face dataset. It does not update the source data itself.

## Climate Baseline & Weather Pattern Clusters

Cluster filters, profile table, legend, PyDeck map, selected-city detail, derived EDA features, and city table.

## Historical Weather Explorer

Reads the consolidated ERA5 dashboard file from Hugging Face. Supports date range, indicator selection, EDA summary, daily trends, monthly aggregation, yearly comparison, year-over-year changes, and daily records.

Aggregation rule:

```text
Precipitation → SUM
Temperature   → MEAN
Humidity      → MEAN
Wind          → MEAN
Pressure      → MEAN
```

## City Comparison & Weather Similarity

Pairwise selectors, cluster relationship, standardized distance, indicator comparison, driver contribution chart, and five nearest weather profiles.

# Online Automation

Workflow:

```text
.github/workflows/refresh_recent_ifs.yml
```

Schedule:

```text
cron: 47 1 * * *
```

Equivalent:

```text
01:47 UTC
07:17 Asia/Colombo
```

Manual `workflow_dispatch` is also enabled.

Workflow sequence:

```text
checkout repository
      ↓
set up Python 3.11
      ↓
install requirements
      ↓
download latest IFS Parquet from Hugging Face
      ↓
record SHA-256
      ↓
run incremental refresh
      ↓
record new SHA-256
      ↓
unchanged → skip upload
changed   → upload to Hugging Face
```

Two manual cloud workflow runs were successfully verified during QA.

# Hugging Face Data Publishing

Dataset repository:

```text
tharinduperera/weather-clustering-data
```

Important published files:

```text
processed/dashboard/weather_era5_2016_2025.parquet

processed/parquet/weather_ifs_2026-present/year=2026/weather.parquet

processed/parquet/weather/year=2016/...
...
processed/parquet/weather/year=2025/...
```

Hugging Face acts as persistent cloud storage because GitHub Actions runners are temporary and Streamlit Community Cloud is not used as a database.

# Streamlit Community Cloud Deployment

Configuration:

```text
Repository:     tharindperera/weather-clustering
Branch:         main
Main file:      app.py
Python:         3.11
```

The app reads small frozen files from Git, ERA5 and IFS from Hugging Face, and current weather from Open-Meteo.

No development PC needs to remain powered on.

# Project Directory Structure

```text
weather-clustering/
│
├── .github/
│   └── workflows/
│       └── refresh_recent_ifs.yml
├── .streamlit/
│   └── config.toml
├── data/
│   ├── locations/
│   │   └── locations.csv
│   └── processed/
│       ├── clustering/
│       │   └── dashboard_clustering_dataset.parquet
│       └── features/
│           └── climate_features.parquet
├── reports/
│   └── qa/
│       ├── phase_11c_data_quality_provenance.json
│       └── phase_11d_automation_operational.json
├── scripts/
│   └── run_daily_refresh.ps1
├── src/
│   ├── analyze_climate_features.py
│   ├── analyze_clustering_input.py
│   ├── analyze_final_clusters.py
│   ├── build_climate_features.py
│   ├── build_clustering_input.py
│   ├── build_dashboard_clustering_dataset.py
│   ├── build_dashboard_era5.py
│   ├── city_comparison.py
│   ├── dashboard_historical.py
│   ├── dashboard_recent_ifs.py
│   ├── duckdb_eda.py
│   ├── finalize_kmeans_clustering.py
│   ├── ingest_historical.py
│   ├── ingest_recent_ifs.py
│   ├── json_to_parquet.py
│   ├── kmeans_model_selection.py
│   ├── publish_dashboard_era5_hf.py
│   ├── qa_automation_operational.py
│   ├── qa_data_provenance.py
│   ├── realtime_weather.py
│   ├── recent_ifs_to_parquet.py
│   ├── refresh_recent_ifs.py
│   ├── standardize_clustering_input.py
│   └── sync_recent_ifs_hf.py
├── app.py
├── requirements.txt
└── README.md
```

# Installation

Validated Python:

```text
Python 3.11.4
```

Pinned direct dependencies:

```text
streamlit==1.64.0
pydeck==0.9.3
pandas==3.0.5
pyarrow==25.0.1
pyspark==4.2.0
duckdb==1.5.5
matplotlib==3.11.2
scikit-learn==1.9.1
huggingface_hub==1.31.0
requests==2.34.2
numpy==2.4.6
```

Clone:

```bash
git clone https://github.com/tharindperera/weather-clustering.git
cd weather-clustering
```

Windows environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Expected:

```text
No broken requirements found.
```

# Running the Dashboard Locally

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Default URL:

```text
http://localhost:8501
```

# Reproducing the Data Pipelines

Main sequence:

```text
locations.csv
    ↓
ingest_historical.py
    ↓
json_to_parquet.py
    ↓
duckdb_eda.py
    ↓
build_climate_features.py
    ↓
build_clustering_input.py
    ↓
analyze_clustering_input.py
    ↓
kmeans_model_selection.py
    ↓
finalize_kmeans_clustering.py
    ↓
analyze_final_clusters.py
    ↓
build_dashboard_clustering_dataset.py
```

Recent pipeline:

```text
test_recent_ifs_api.py
    ↓
ingest_recent_ifs.py
    ↓
recent_ifs_to_parquet.py
    ↓
refresh_recent_ifs.py
    ↓
sync_recent_ifs_hf.py
```

Dashboard publishing:

```text
build_dashboard_era5.py
    ↓
publish_dashboard_era5_hf.py
```

The full historical ingestion should not normally be rerun because the official ERA5 baseline has already been frozen and validated.

# Quality Assurance and Validation

## Phase 11A — Production Smoke Test

Public endpoint returned HTTP 200 and all five dashboard sections were manually verified in a private/incognito browser session.

## Phase 11B — Fresh-Clone Reproducibility

A completely fresh clone was tested with a new Python 3.11.4 virtual environment.

Results:

```text
pip check:                     PASS
compileall:                    PASS
Git static datasets:           PASS
anonymous ERA5 download:       PASS
anonymous IFS download:        PASS
cloud loaders:                 PASS
fresh-clone Streamlit health:  200 ok
```

## Phase 11C — Data Quality & Provenance

Script:

```text
src/qa_data_provenance.py
```

Report:

```text
reports/qa/phase_11c_data_quality_provenance.json
```

Validated:

```text
Locations:                       100
Countries:                       80
Duplicate location IDs:          0

K:                               4
Cluster sizes:                   36 / 20 / 34 / 10

Derived features:                16
Used for K-Means:                No

ERA5 rows:                       365,300
ERA5 period:                     2016-01-01 → 2025-12-31
ERA5 observations/city:          3,653
ERA5 duplicates:                 0
ERA5 missing weather values:     0
ERA5 temperature violations:     0

IFS QA snapshot rows:             25,800
IFS QA snapshot period:           2026-01-01 → 2026-09-15
IFS observations/city snapshot:   258
IFS duplicates:                   0
IFS missing weather values:       0
IFS temperature violations:       0
```

## Phase 11D — Operational / Failure-Recovery QA

Script:

```text
src/qa_automation_operational.py
```

Report:

```text
reports/qa/phase_11d_automation_operational.json
```

Verified workflow configuration, missed-schedule catch-up, idempotency, atomic update safety, and checkpoint safety.

Three missed days were simulated against a temporary dataset:

```text
3 days × 100 cities = 300 new rows
5 synthetic multi-location batch fetches
0 duplicates
```

A second run at the same latest date made zero fetches and left the Parquet SHA-256 unchanged.

A synthetic API failure left the Parquet byte-for-byte unchanged and did not advance the successful-date checkpoint.

# Failure Recovery and Idempotency

Already current:

```text
stored date == latest complete day
              ↓
        zero API requests
              ↓
             exit
```

Missing dates:

```text
stored date < latest complete day
              ↓
          find date gap
              ↓
        fetch only gap
              ↓
           validate
              ↓
            upsert
```

Failed refresh:

```text
fetch/validation failure
              ↓
        exception propagated
              ↓
   persisted Parquet unchanged
              ↓
 checkpoint remains at last success
```

# Security and Secrets

No write credentials are committed to Git.

GitHub Actions uses a repository secret:

```text
HF_TOKEN
```

for Hugging Face write access.

The Hugging Face dataset is public, so Streamlit only requires anonymous read access.

Least-privilege design:

```text
GitHub Actions → authenticated writer
Streamlit      → anonymous reader
```

# Important Methodological Decisions

## ERA5 Is Frozen

The clustering baseline remains fixed at 2016–2025 to preserve reproducibility.

## IFS Does Not Automatically Retrain K-Means

The recent layer is operational/recent data, not part of the official 10-year climate reference.

## No Coordinates in K-Means

Latitude and longitude are visualization metadata only.

## PCA Is Visualization Only

PCA does not replace the five standardized clustering inputs.

## Derived Features Are EDA Only

The 16 derived climate variables are descriptive and excluded from K-Means.

## No Conventional Train/Test Split

The task is unsupervised and has no target label.

## Surface Pressure Is Retained

Low surface pressure in high-altitude cities is physically meaningful and contributes to weather-profile differences.

## No Log Transformation

All five original clustering indicators were kept in interpretable form and standardized before K-Means.

# Known Limitations

1. A city coordinate represents one point and does not capture every local microclimate.
2. The 100-city catalogue is globally distributed but not exhaustive.
3. K-Means assumes Euclidean geometry and relatively compact groups.
4. Cluster labels are descriptive interpretations, not formal climate classifications.
5. The IFS operational/historical-forecast layer is methodologically separate from the ERA5 reanalysis baseline.
6. Surface pressure strongly reflects elevation.
7. Real-time features depend on Open-Meteo availability.
8. The public system also depends on GitHub, Hugging Face, and Streamlit Community Cloud availability.

# Dynamic Data Notes

Frozen values:

```text
Cities:                  100
Countries:               80
ERA5 rows:               365,300
ERA5 period:             2016–2025
ERA5 observations/city:  3,653
Final K:                 4
Cluster sizes:           36 / 20 / 34 / 10
```

Dynamic values:

```text
latest IFS date
IFS total rows
IFS observations per city
```

Phase 11C validated snapshot:

```text
Latest IFS date:        2026-09-15
IFS rows:               25,800
IFS observations/city:  258
```

The deployed dashboard and Hugging Face dataset should be used for the current recent-data date after automation advances.

# Reproducibility Summary

The verified reproducibility chain is:

```text
fresh GitHub clone
      ↓
Python 3.11.4
      ↓
pinned requirements
      ↓
public Hugging Face datasets
      ↓
no development-machine-only data dependency
      ↓
Streamlit starts successfully
      ↓
HTTP 200 health response
```

# Final Project Status

| Component | Status |
|---|---|
| 100-city / 80-country catalogue | ✅ |
| ERA5 2016–2025 ingestion | ✅ |
| Raw JSON persistence | ✅ |
| Partitioned Parquet | ✅ |
| DuckDB EDA | ✅ |
| 16 derived EDA features | ✅ |
| 5-variable clustering input | ✅ |
| Spark K=2..8 model selection | ✅ |
| Final K=4 model | ✅ |
| Stability checks | ✅ |
| PCA visualization | ✅ |
| Geographic visualization | ✅ |
| City similarity engine | ✅ |
| Real-time weather | ✅ |
| ECMWF IFS recent layer | ✅ |
| Incremental refresh | ✅ |
| Hugging Face publication | ✅ |
| GitHub Actions automation | ✅ |
| Streamlit deployment | ✅ |
| Production smoke test | ✅ |
| Fresh-clone reproducibility | ✅ |
| Data quality/provenance QA | ✅ |
| Failure-recovery QA | ✅ |

# Final Production Flow

```text
                         Open-Meteo
                             |
            +----------------+----------------+
            |                |                |
            v                v                v
          ERA5           ECMWF IFS         Current
       2016–2025          2026+            Weather
            |                |                |
            |                v                |
            |         GitHub Actions           |
            |                |                |
            v                v                |
       Hugging Face <--- Updated Parquet       |
            |                                 |
            +----------------+----------------+
                             |
                             v
                  Streamlit Community Cloud
                             |
                             v
                    Public Analytics App
```

---

## Public Dashboard

https://weather-clustering-and-pattern-analytics.streamlit.app/

## Source Code

https://github.com/tharindperera/weather-clustering

## Published Dataset

https://huggingface.co/datasets/tharinduperera/weather-clustering-data

---

**Project state:** Production pipeline implemented, online automation verified, public dashboard deployed, fresh-clone reproducibility proven, and final data-quality and operational QA passed.
