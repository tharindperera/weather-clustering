from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SOURCE_FILE = Path("data/locations/source/worldcities.csv")
OUTPUT_FILE = Path("data/locations/locations.csv")

TARGET_LOCATIONS = 500

# We want a reasonable minimum prominence.
# Lower ranking number in the source means more prominent,
# but the current CSV does not contain the ranking field,
# so population will be our main prominence signal.
MIN_POPULATION = 100_000


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
    print("BUILDING FINAL LOCATION CATALOGUE")
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
    # 2. Basic validation
    # -----------------------------------------------------

    required = [
        "city",
        "city_ascii",
        "lat",
        "lng",
        "country",
        "iso3",
        "population",
        "id",
    ]

    missing_columns = [
        column for column in required
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # -----------------------------------------------------
    # 3. Remove invalid geographic records
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

    print(f"After basic validation: {len(df):,}")

    # -----------------------------------------------------
    # 4. Remove duplicate city-country identities
    # -----------------------------------------------------

    df["city_key"] = (
        df["iso3"].astype(str).str.upper()
        + "::"
        + df["city_ascii"].astype(str).str.lower().str.strip()
    )

    df = df.sort_values(
        by=["population", "id"],
        ascending=[False, True],
        na_position="last",
    )

    df = df.drop_duplicates(
        subset=["city_key"],
        keep="first",
    ).copy()

    print(f"After city deduplication: {len(df):,}")

    # -----------------------------------------------------
    # 5. Prefer cities with known population
    # -----------------------------------------------------

    df["has_population"] = df["population"].notna()

    # Candidates with useful population information.
    populated = df[
        df["population"].notna()
        & (df["population"] >= MIN_POPULATION)
    ].copy()

    print(
        f"Cities with population >= {MIN_POPULATION:,}: "
        f"{len(populated):,}"
    )

    # If fewer than TARGET_LOCATIONS somehow survive,
    # fall back to all valid cities.
    if len(populated) < TARGET_LOCATIONS:
        print(
            "Warning: not enough populated cities. "
            "Falling back to all valid cities."
        )
        candidates = df.copy()
    else:
        candidates = populated.copy()

    # -----------------------------------------------------
    # 6. Country-balanced deterministic selection
    # -----------------------------------------------------
    #
    # Strategy:
    #   - First choose the most populous qualifying city
    #     from each country.
    #   - Then fill remaining slots by global population,
    #     while applying a maximum-per-country cap.
    #
    # This prevents one country from dominating the sample.

    candidates = candidates.sort_values(
        by=["population", "id"],
        ascending=[False, True],
        na_position="last",
    )

    # First: one strong representative per country.
    first_per_country = (
        candidates
        .sort_values(
            by=["population", "id"],
            ascending=[False, True],
            na_position="last",
        )
        .groupby("iso3", as_index=False)
        .first()
    )

    selected = first_per_country.copy()

    # Maximum number of locations per country.
    max_per_country = 15

    selected_counts = selected["iso3"].value_counts().to_dict()

    remaining = candidates[
        ~candidates["id"].isin(selected["id"])
    ].copy()

    for _, row in remaining.iterrows():

        if len(selected) >= TARGET_LOCATIONS:
            break

        country = row["iso3"]
        count = selected_counts.get(country, 0)

        if count >= max_per_country:
            continue

        selected = pd.concat(
            [selected, pd.DataFrame([row])],
            ignore_index=True,
        )

        selected_counts[country] = count + 1

    # -----------------------------------------------------
    # 7. Final deterministic ordering
    # -----------------------------------------------------

    selected = selected.sort_values(
        by=["country", "population", "city_ascii"],
        ascending=[True, False, True],
        na_position="last",
    ).head(TARGET_LOCATIONS)

    # -----------------------------------------------------
    # 8. Build final schema
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
    # 9. Validate final result
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

    if final[
        ["latitude", "longitude"]
    ].isna().any().any():
        raise RuntimeError(
            "Missing coordinates found."
        )

    # -----------------------------------------------------
    # 10. Save
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
    # 11. Print summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL LOCATION CATALOGUE")
    print("=" * 70)

    print(f"Locations selected : {len(final):,}")
    print(
        f"Countries represented: "
        f"{final['iso3'].nunique():,}"
    )

    print("\nColumns:")
    for column in final.columns:
        print(f"  - {column}")

    print("\nFirst 10 locations:")
    print(
        final.head(10).to_string(index=False)
    )

    print("\nPopulation statistics:")
    print(
        final["population"].describe()
    )

    print(f"\nSaved to: {OUTPUT_FILE}")

    print("\nLocation catalogue created successfully.")


if __name__ == "__main__":
    main()