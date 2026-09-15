from pathlib import Path

import pandas as pd


SOURCE_FILE = Path("data/locations/source/worldcities.csv")


def main() -> None:
    print("=" * 70)
    print("SIMPLEMAPS WORLD CITIES DATASET INSPECTION")
    print("=" * 70)

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source dataset not found:\n{SOURCE_FILE}"
        )

    df = pd.read_csv(SOURCE_FILE)

    print("\nDataset shape:")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nMissing values in important fields:")

    important_columns = [
        "city",
        "lat",
        "lng",
        "country",
        "iso2",
        "iso3",
        "population",
        "timezone",
        "ranking",
        "id",
    ]

    for column in important_columns:
        if column in df.columns:
            missing = df[column].isna().sum()
            print(f"  {column:15s}: {missing:,}")


if __name__ == "__main__":
    main()