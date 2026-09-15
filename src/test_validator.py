import json
from pathlib import Path

from config import (
    HISTORICAL_END_DATE,
    HISTORICAL_START_DATE,
)
from validator import validate_response


RAW_FILE = (
    Path("data")
    / "raw"
    / "historical"
    / "test_batch_001.json"
)


def main() -> None:

    print("=" * 70)
    print("VALIDATOR MODULE TEST")
    print("=" * 70)

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw test file not found: {RAW_FILE}"
        )

    with RAW_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    results = validate_response(
        data=data,
        expected_locations=20,
        expected_days=14,
        expected_start=HISTORICAL_START_DATE,
        expected_end=HISTORICAL_END_DATE,
    )

    print(
        f"\nValidated locations: {len(results)}"
    )

    print(
        "\nVALIDATOR TEST PASSED"
    )


if __name__ == "__main__":
    main()
    