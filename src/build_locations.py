import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SOURCE_FILE = Path("data/locations/source/worldcities.csv")
OUTPUT_FILE = Path("data/locations/locations.csv")

TARGET_LOCATIONS = 100

# Minimum population used to identify reasonably prominent
# cities in the source catalogue.
MIN_POPULATION = 100_000

# First select one representative city from each of the
# strongest countries, then use the remaining slots to add
# additional cities while keeping country concentration low.
PRIMARY_COUNTRIES = 80

# No country may contribute more than this many cities.
MAX_PER_COUNTRY = 3


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------


def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with valid geographic coordinates."""

    return df[
        df["lat"].between(-90, 90)
        & df["lng"].between(-180, 180)
    ].copy()


# ---------------------------------------------------------
# Main selection logic
# ---------------------------------------------------------


def main() -> None:

    print("=" * 70)
    print("BUILDING 100-CITY LOCATION CATALOGUE")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. Load source data
    # -----------------------------------------------------

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source file not found: {SOURCE_FILE}"
        )

    df = pd.read_csv(SOURCE_FILE)

    print(f"\nSource rows: {len(df):,}")

    # -----------------------------------------------------
    # 2. Validate required source columns
    # -----------------------------------------------------

    required = [
        "city",
        "city_ascii",
        "lat",
        "lng",
        "country",
        "iso2",
        "iso3",
        "admin_name",
        "capital",
        "population",
        "id",
    ]

    missing_columns = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # -----------------------------------------------------
    # 3. Remove invalid records
    # -----------------------------------------------------

    df = df.dropna(
        subset=[
            "city",
            "city_ascii",
            "country",
            "iso3",
            "lat",
            "lng",
            "id",
        ]
    ).copy()

    df = validate_coordinates(df)

    print(
        f"After geographic validation: {len(df):,}"
    )

    # -----------------------------------------------------
    # 4. Build deterministic city identity
    # -----------------------------------------------------

    df["city_key"] = (
        df["iso3"]
        .astype(str)
        .str.upper()
        .str.strip()
        + "::"
        + df["city_ascii"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Prefer larger populations, then smaller source ID.
    df = df.sort_values(
        by=["population", "id"],
        ascending=[False, True],
        na_position="last",
    )

    df = df.drop_duplicates(
        subset=["city_key"],
        keep="first",
    ).copy()

    print(
        f"After city deduplication: {len(df):,}"
    )

    # -----------------------------------------------------
    # 5. Select population-qualified candidates
    # -----------------------------------------------------

    candidates = df[
        df["population"].notna()
        & (df["population"] >= MIN_POPULATION)
    ].copy()

    print(
        f"Cities with population >= "
        f"{MIN_POPULATION:,}: {len(candidates):,}"
    )

    if len(candidates) < TARGET_LOCATIONS:
        raise RuntimeError(
            f"Only {len(candidates)} cities satisfy the "
            f"minimum population requirement, but "
            f"{TARGET_LOCATIONS} are required."
        )

    # -----------------------------------------------------
    # 6. Select the strongest representative per country
    # -----------------------------------------------------

    representatives = (
        candidates
        .sort_values(
            by=["iso3", "population", "id"],
            ascending=[True, False, True],
        )
        .groupby(
            "iso3",
            as_index=False,
            sort=False,
        )
        .first()
    )

    country_count = len(representatives)

    print(
        f"Countries with qualifying cities: "
        f"{country_count:,}"
    )

    if country_count < PRIMARY_COUNTRIES:
        raise RuntimeError(
            f"Only {country_count} qualifying countries "
            f"are available, but "
            f"{PRIMARY_COUNTRIES} are required."
        )

    # -----------------------------------------------------
    # 7. Choose the primary countries
    # -----------------------------------------------------

    primary_countries = (
        representatives
        .sort_values(
            by=["population", "id"],
            ascending=[False, True],
        )
        .head(PRIMARY_COUNTRIES)
    )

    selected = primary_countries.copy()

    selected_counts = (
        selected["iso3"]
        .value_counts()
        .to_dict()
    )

    print(
        f"Primary countries selected: "
        f"{selected['iso3'].nunique():,}"
    )

    # -----------------------------------------------------
    # 8. Fill remaining slots with additional cities
    # -----------------------------------------------------
    #
    # Only cities from the selected primary countries can
    # receive extra slots. This keeps the final catalogue
    # globally diverse while allowing some within-country
    # climate variation.
    # -----------------------------------------------------

    remaining_slots = (
        TARGET_LOCATIONS - len(selected)
    )

    additional_candidates = candidates[
        candidates["iso3"].isin(
            selected["iso3"]
        )
        & ~candidates["id"].isin(
            selected["id"]
        )
    ].copy()

    additional_candidates = additional_candidates.sort_values(
        by=["population", "id"],
        ascending=[False, True],
    )

    additional_rows = []

    for _, row in additional_candidates.iterrows():

        if len(additional_rows) >= remaining_slots:
            break

        country = row["iso3"]

        current_count = selected_counts.get(
            country,
            0,
        )

        if current_count >= MAX_PER_COUNTRY:
            continue

        additional_rows.append(row)

        selected_counts[country] = (
            current_count + 1
        )

    if len(additional_rows) < remaining_slots:
        raise RuntimeError(
            "Unable to fill the requested 100-city "
            "catalogue under the country cap."
        )

    if additional_rows:
        selected = pd.concat(
            [
                selected,
                pd.DataFrame(additional_rows),
            ],
            ignore_index=True,
        )

    # -----------------------------------------------------
    # 9. Final deterministic ordering
    # -----------------------------------------------------

    selected = (
        selected
        .sort_values(
            by=[
                "country",
                "population",
                "city_ascii",
                "id",
            ],
            ascending=[
                True,
                False,
                True,
                True,
            ],
            na_position="last",
        )
        .head(TARGET_LOCATIONS)
        .copy()
    )

    # -----------------------------------------------------
    # 10. Build final project schema
    # -----------------------------------------------------

    final = selected[
        [
            "id",
            "city",
            "city_ascii",
            "country",
            "iso2",
            "iso3",
            "admin_name",
            "capital",
            "lat",
            "lng",
            "population",
        ]
    ].copy()

    final = final.rename(
        columns={
            "id": "location_id",
            "lat": "latitude",
            "lng": "longitude",
        }
    )

    # -----------------------------------------------------
    # 11. Final validation
    # -----------------------------------------------------

    if len(final) != TARGET_LOCATIONS:
        raise RuntimeError(
            f"Expected {TARGET_LOCATIONS} locations, "
            f"but selected {len(final)}."
        )

    if final["location_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate location_id values found."
        )

    if final["iso3"].isna().any():
        raise RuntimeError(
            "Missing country codes found."
        )

    if final[
        ["latitude", "longitude"]
    ].isna().any().any():
        raise RuntimeError(
            "Missing coordinates found."
        )

    if not final["latitude"].between(
        -90,
        90,
    ).all():
        raise RuntimeError(
            "Invalid latitude detected."
        )

    if not final["longitude"].between(
        -180,
        180,
    ).all():
        raise RuntimeError(
            "Invalid longitude detected."
        )

    country_counts = (
        final["iso3"]
        .value_counts()
    )

    if country_counts.max() > MAX_PER_COUNTRY:
        raise RuntimeError(
            "Country concentration exceeds the "
            f"maximum of {MAX_PER_COUNTRY} cities."
        )

    if final["iso3"].nunique() < PRIMARY_COUNTRIES:
        raise RuntimeError(
            "Final catalogue contains fewer than "
            f"{PRIMARY_COUNTRIES} countries."
        )

    # -----------------------------------------------------
    # 12. Save
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # 13. Print summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL 100-CITY LOCATION CATALOGUE")
    print("=" * 70)

    print(
        f"Locations selected     : "
        f"{len(final):,}"
    )

    print(
        f"Countries represented  : "
        f"{final['iso3'].nunique():,}"
    )

    print(
        f"Maximum cities/country : "
        f"{country_counts.max()}"
    )

    print("\nColumns:")

    for column in final.columns:
        print(f"  - {column}")

    print("\nFirst 10 locations:")

    print(
        final.head(10).to_string(
            index=False
        )
    )

    print("\nPopulation statistics:")

    print(
        final["population"].describe()
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )

    print(
        "\n100-city location catalogue "
        "created successfully."
    )


if __name__ == "__main__":
    main()