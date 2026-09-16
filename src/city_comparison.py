from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
if not CLUSTER_DATA_FILE.exists():
    CLUSTER_DATA_FILE = Path(
        "data/processed/clustering/dashboard_clustering_dataset.parquet"
    )

WEATHER_COLUMNS = [
    "temperature_c",
    "rainfall_mm_per_day",
    "humidity_percent",
    "wind_speed_kmh",
    "pressure_hpa",
]


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_city_comparison_data() -> pd.DataFrame:
    """
    Load the 100-city dashboard clustering dataset.
    """

    df = pd.read_parquet(
        CLUSTER_DATA_FILE
    )

    required_columns = [
        "location_id",
        "city",
        "country",
        "cluster_id",
        *WEATHER_COLUMNS,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing comparison columns: {missing}"
        )

    if len(df) != 100:
        raise ValueError(
            f"Expected 100 cities, found {len(df)}."
        )

    if df["city"].nunique() != 100:
        raise ValueError(
            "Expected 100 unique cities."
        )

    return df


# ---------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------

def standardize_weather_profiles(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize the five weather variables across the 100 cities.

    Population standard deviation (ddof=0) is used to match the
    mathematical convention used by Spark StandardScaler.
    """

    result = df.copy()

    means = result[
        WEATHER_COLUMNS
    ].mean()

    stds = result[
        WEATHER_COLUMNS
    ].std(
        ddof=0
    )

    if (stds == 0).any():
        zero_variance = stds[
            stds == 0
        ].index.tolist()

        raise ValueError(
            "Zero-variance weather variables: "
            f"{zero_variance}"
        )

    for column in WEATHER_COLUMNS:
        result[
            f"{column}_z"
        ] = (
            result[column] - means[column]
        ) / stds[column]

    return result


# ---------------------------------------------------------------------
# City lookup
# ---------------------------------------------------------------------

def get_city_record(
    standardized_df: pd.DataFrame,
    city: str,
) -> pd.Series:
    """
    Return one city's record.
    """

    matches = standardized_df[
        standardized_df["city"] == city
    ]

    if matches.empty:
        raise ValueError(
            f"City not found: {city}"
        )

    if len(matches) != 1:
        raise ValueError(
            f"Expected one record for {city}, "
            f"found {len(matches)}."
        )

    return matches.iloc[0]


# ---------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------

def compare_cities(
    standardized_df: pd.DataFrame,
    city_a: str,
    city_b: str,
) -> dict:
    """
    Compare two cities using the original five weather variables.

    Returns both original-unit differences and standardized
    Euclidean distance.
    """

    if city_a == city_b:
        raise ValueError(
            "Please select two different cities."
        )

    row_a = get_city_record(
        standardized_df,
        city_a,
    )

    row_b = get_city_record(
        standardized_df,
        city_b,
    )

    z_columns = [
        f"{column}_z"
        for column in WEATHER_COLUMNS
    ]

    vector_a = row_a[
        z_columns
    ].to_numpy(
        dtype=float
    )

    vector_b = row_b[
        z_columns
    ].to_numpy(
        dtype=float
    )

    standardized_distance = float(
        np.linalg.norm(
            vector_a - vector_b
        )
    )

    variable_comparison = []

    for column in WEATHER_COLUMNS:

        value_a = float(
            row_a[column]
        )

        value_b = float(
            row_b[column]
        )

        difference = value_a - value_b

        variable_comparison.append(
            {
                "variable": column,
                "city_a": value_a,
                "city_b": value_b,
                "difference": difference,
                "absolute_difference": abs(
                    difference
                ),
            }
        )

    # The standardized per-variable differences show which dimensions
    # contribute most to the pairwise distance.
    contribution_rows = []

    squared_differences = (
        vector_a - vector_b
    ) ** 2

    total_squared_distance = (
        squared_differences.sum()
    )

    for column, squared_difference in zip(
        WEATHER_COLUMNS,
        squared_differences,
    ):

        contribution = (
            float(squared_difference)
            / float(total_squared_distance)
            if total_squared_distance > 0
            else 0.0
        )

        contribution_rows.append(
            {
                "variable": column,
                "standardized_difference": float(
                    abs(
                        row_a[
                            f"{column}_z"
                        ]
                        - row_b[
                            f"{column}_z"
                        ]
                    )
                ),
                "distance_contribution": contribution,
            }
        )

    return {
        "city_a": city_a,
        "city_b": city_b,
        "country_a": row_a["country"],
        "country_b": row_b["country"],
        "cluster_a": int(
            row_a["cluster_id"]
        ),
        "cluster_b": int(
            row_b["cluster_id"]
        ),
        "same_cluster": (
            int(row_a["cluster_id"])
            == int(row_b["cluster_id"])
        ),
        "standardized_distance": standardized_distance,
        "variable_comparison": pd.DataFrame(
            variable_comparison
        ),
        "distance_contributions": pd.DataFrame(
            contribution_rows
        ).sort_values(
            "distance_contribution",
            ascending=False,
        ).reset_index(
            drop=True
        ),
    }


# ---------------------------------------------------------------------
# Nearest cities
# ---------------------------------------------------------------------

def find_most_similar_cities(
    standardized_df: pd.DataFrame,
    city: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Find the nearest weather-profile cities using standardized
    Euclidean distance.
    """

    target = get_city_record(
        standardized_df,
        city,
    )

    z_columns = [
        f"{column}_z"
        for column in WEATHER_COLUMNS
    ]

    target_vector = target[
        z_columns
    ].to_numpy(
        dtype=float
    )

    rows = []

    for _, row in standardized_df.iterrows():

        if row["city"] == city:
            continue

        vector = row[
            z_columns
        ].to_numpy(
            dtype=float
        )

        distance = float(
            np.linalg.norm(
                target_vector - vector
            )
        )

        rows.append(
            {
                "city": row["city"],
                "country": row["country"],
                "cluster_id": int(
                    row["cluster_id"]
                ),
                "distance": distance,
            }
        )

    result = (
        pd.DataFrame(rows)
        .sort_values("distance")
        .head(top_n)
        .reset_index(drop=True)
    )

    return result
