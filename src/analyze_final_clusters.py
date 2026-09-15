from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import PROCESSED_DIR, LOCATIONS_FILE

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

INPUT_FILE = PROCESSED_DIR / "clustering" / "final_clusters_k4.parquet"
OUTPUT_DIR = PROCESSED_DIR / "clustering" / "final_analysis"

PROFILE_FILE = OUTPUT_DIR / "cluster_profiles_original_units.csv"
STANDARDIZED_PROFILE_FILE = OUTPUT_DIR / "cluster_profiles_standardized.csv"
PCA_FILE = OUTPUT_DIR / "cluster_pca_coordinates.csv"

PROFILE_PLOT = OUTPUT_DIR / "cluster_profiles_original_units.png"
HEATMAP_PLOT = OUTPUT_DIR / "cluster_profiles_standardized.png"
PCA_PLOT = OUTPUT_DIR / "cluster_pca.png"
GEOGRAPHIC_PLOT = OUTPUT_DIR / "cluster_geographic_distribution.png"


# ---------------------------------------------------------------------
# Original weather variables used for clustering
# ---------------------------------------------------------------------

WEATHER_COLUMNS = [
    "temperature_mean",
    "precipitation_sum",
    "relative_humidity_mean",
    "wind_speed_mean",
    "surface_pressure_mean",
]

DISPLAY_NAMES = {
    "temperature_mean": "Temperature (°C)",
    "precipitation_sum": "Rainfall (mm/day)",
    "relative_humidity_mean": "Humidity (%)",
    "wind_speed_mean": "Wind speed (km/h)",
    "surface_pressure_mean": "Surface pressure (hPa)",
}


def main() -> None:
    print("=" * 80)
    print("FINAL K=4 CLUSTER ANALYSIS AND VISUALIZATION")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # 1. Load final cluster assignments
    # -----------------------------------------------------------------

    print()
    print("1. LOADING FINAL CLUSTER ASSIGNMENTS")
    print("-" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    # Merge latitude & longitude from location catalogue if missing
    if ("latitude" not in df.columns or "longitude" not in df.columns) and LOCATIONS_FILE.exists():
        loc_df = pd.read_csv(LOCATIONS_FILE)[["location_id", "latitude", "longitude"]]
        df = df.merge(loc_df, on="location_id", how="left")

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    required_columns = [
        "location_id",
        "city",
        "country",
        "cluster_id",
        *WEATHER_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if len(df) != 100:
        raise ValueError(
            f"Expected 100 cities, found {len(df)}."
        )

    if df["city"].nunique() != 100:
        raise ValueError(
            "Expected 100 unique cities."
        )

    if df["cluster_id"].nunique() != 4:
        raise ValueError(
            f"Expected 4 clusters, found {df['cluster_id'].nunique()}."
        )

    if df[WEATHER_COLUMNS].isna().any().any():
        raise ValueError(
            "Missing weather values detected."
        )

    print("Input validation: PASSED")

    # -----------------------------------------------------------------
    # 2. Cluster sizes
    # -----------------------------------------------------------------

    print()
    print("2. FINAL CLUSTER SIZES")
    print("-" * 80)

    cluster_sizes = (
        df.groupby("cluster_id")
        .size()
        .sort_index()
    )

    print(cluster_sizes.to_string())

    # -----------------------------------------------------------------
    # 3. Cluster profiles in original units
    # -----------------------------------------------------------------

    print()
    print("3. CLUSTER PROFILES — ORIGINAL UNITS")
    print("-" * 80)

    profile = (
        df.groupby("cluster_id")[WEATHER_COLUMNS]
        .mean()
        .sort_index()
    )

    profile["size"] = cluster_sizes

    profile = profile[
        ["size"] + WEATHER_COLUMNS
    ]

    print(
        profile.round(3).to_string()
    )

    profile.to_csv(
        PROFILE_FILE,
        index=True,
    )

    # -----------------------------------------------------------------
    # 4. Standardized cluster profiles
    # -----------------------------------------------------------------

    print()
    print("4. STANDARDIZED CLUSTER PROFILES")
    print("-" * 80)

    global_means = df[WEATHER_COLUMNS].mean()
    global_stds = df[WEATHER_COLUMNS].std(ddof=0)

    standardized_values = (
        profile[WEATHER_COLUMNS] - global_means
    ) / global_stds

    print(
        standardized_values.round(3).to_string()
    )

    standardized_values.to_csv(
        STANDARDIZED_PROFILE_FILE,
        index=True,
    )

    # -----------------------------------------------------------------
    # 5. PCA — visualization only
    # -----------------------------------------------------------------

    print()
    print("5. PCA VISUALIZATION")
    print("-" * 80)

    X = df[WEATHER_COLUMNS].to_numpy(
        dtype=float
    )

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)

    X_standardized = (
        X - X_mean
    ) / X_std

    pca = PCA(
        n_components=2
    )

    pca_coordinates = pca.fit_transform(
        X_standardized
    )

    pca_df = pd.DataFrame(
        {
            "city": df["city"].values,
            "country": df["country"].values,
            "cluster_id": df["cluster_id"].values,
            "PC1": pca_coordinates[:, 0],
            "PC2": pca_coordinates[:, 1],
        }
    )

    pca_df.to_csv(
        PCA_FILE,
        index=False,
    )

    explained = pca.explained_variance_ratio_

    print(
        f"PC1 explained variance: "
        f"{explained[0] * 100:.2f}%"
    )

    print(
        f"PC2 explained variance: "
        f"{explained[1] * 100:.2f}%"
    )

    print(
        f"Total shown in 2D: "
        f"{explained.sum() * 100:.2f}%"
    )

    # -----------------------------------------------------------------
    # 6. Plot — original-unit cluster profiles
    # -----------------------------------------------------------------

    print()
    print("6. CREATING PROFILE VISUALIZATION")
    print("-" * 80)

    fig, axes = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(10, 18),
    )

    clusters = profile.index.tolist()

    for axis, feature in zip(
        axes,
        WEATHER_COLUMNS,
    ):
        values = profile.loc[
            clusters,
            feature,
        ]

        axis.bar(
            [str(cluster) for cluster in clusters],
            values,
        )

        axis.set_title(
            DISPLAY_NAMES[feature]
        )

        axis.set_xlabel(
            "Cluster"
        )

        axis.set_ylabel(
            DISPLAY_NAMES[feature]
        )

    fig.suptitle(
        "Final K=4 Weather Cluster Profiles",
        fontsize=16,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    fig.savefig(
        PROFILE_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -----------------------------------------------------------------
    # 7. Plot — standardized profile heatmap
    # -----------------------------------------------------------------

    print("Creating standardized profile heatmap...")

    heatmap_data = standardized_values[
        WEATHER_COLUMNS
    ].copy()

    heatmap_data.columns = [
        DISPLAY_NAMES[column]
        for column in WEATHER_COLUMNS
    ]

    fig, axis = plt.subplots(
        figsize=(11, 5)
    )

    image = axis.imshow(
        heatmap_data.values,
        aspect="auto",
    )

    axis.set_xticks(
        range(len(heatmap_data.columns))
    )

    axis.set_xticklabels(
        heatmap_data.columns,
        rotation=30,
        ha="right",
    )

    axis.set_yticks(
        range(len(heatmap_data.index))
    )

    axis.set_yticklabels(
        [
            f"Cluster {cluster}"
            for cluster in heatmap_data.index
        ]
    )

    axis.set_title(
        "Final K=4 Cluster Profiles — Standardized Weather Variables"
    )

    for row in range(
        heatmap_data.shape[0]
    ):
        for column in range(
            heatmap_data.shape[1]
        ):
            axis.text(
                column,
                row,
                f"{heatmap_data.iloc[row, column]:.2f}",
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=axis,
        label="Standardized value (z-score)",
    )

    fig.tight_layout()

    fig.savefig(
        HEATMAP_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -----------------------------------------------------------------
    # 8. Plot — PCA
    # -----------------------------------------------------------------

    print("Creating PCA visualization...")

    fig, axis = plt.subplots(
        figsize=(10, 7)
    )

    for cluster in sorted(
        pca_df["cluster_id"].unique()
    ):
        subset = pca_df[
            pca_df["cluster_id"] == cluster
        ]

        axis.scatter(
            subset["PC1"],
            subset["PC2"],
            label=f"Cluster {cluster}",
            s=50,
            alpha=0.8,
        )

        for _, row in subset.iterrows():
            axis.annotate(
                row["city"],
                (
                    row["PC1"],
                    row["PC2"],
                ),
                fontsize=6,
                alpha=0.7,
            )

    axis.set_xlabel(
        f"PC1 ({explained[0] * 100:.1f}% variance)"
    )

    axis.set_ylabel(
        f"PC2 ({explained[1] * 100:.1f}% variance)"
    )

    axis.set_title(
        "Weather Pattern Clusters — PCA Projection"
    )

    axis.legend()

    fig.tight_layout()

    fig.savefig(
        PCA_PLOT,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -----------------------------------------------------------------
    # 9. Geographic distribution
    # -----------------------------------------------------------------

    coordinate_columns = [
        "latitude",
        "longitude",
    ]

    if all(
        column in df.columns
        for column in coordinate_columns
    ):
        print(
            "Creating geographic cluster visualization..."
        )

        fig, axis = plt.subplots(
            figsize=(14, 8)
        )

        for cluster in sorted(
            df["cluster_id"].unique()
        ):
            subset = df[
                df["cluster_id"] == cluster
            ]

            axis.scatter(
                subset["longitude"],
                subset["latitude"],
                label=f"Cluster {cluster}",
                s=45,
                alpha=0.8,
            )

        axis.set_xlabel(
            "Longitude"
        )

        axis.set_ylabel(
            "Latitude"
        )

        axis.set_title(
            "Geographic Distribution of Weather Pattern Clusters"
        )

        axis.legend()

        axis.axhline(
            0,
            linewidth=0.5,
        )

        axis.axvline(
            0,
            linewidth=0.5,
        )

        fig.tight_layout()

        fig.savefig(
            GEOGRAPHIC_PLOT,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

    else:
        print(
            "Latitude/longitude not present in final clustering file."
        )
        print(
            "Geographic visualization will be created later "
            "from the location catalogue."
        )

    # -----------------------------------------------------------------
    # 10. Final report
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL ANALYSIS OUTPUTS")
    print("=" * 80)

    print(f"Cluster profile:")
    print(f"  {PROFILE_FILE}")

    print()
    print(f"Standardized cluster profile:")
    print(f"  {STANDARDIZED_PROFILE_FILE}")

    print()
    print(f"PCA coordinates:")
    print(f"  {PCA_FILE}")

    print()
    print(f"Profile visualization:")
    print(f"  {PROFILE_PLOT}")

    print()
    print(f"Standardized profile heatmap:")
    print(f"  {HEATMAP_PLOT}")

    print()
    print(f"PCA visualization:")
    print(f"  {PCA_PLOT}")

    if GEOGRAPHIC_PLOT.exists():
        print()
        print(f"Geographic visualization:")
        print(f"  {GEOGRAPHIC_PLOT}")

    print()
    print("=" * 80)
    print("FINAL CLUSTER ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
