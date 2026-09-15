from __future__ import annotations

from typing import Any


REQUIRED_DAILY_VARIABLES = [
    "time",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "surface_pressure_mean",
]


class ValidationError(Exception):
    """Raised when an Open-Meteo response fails validation."""


def normalize_results(
    data: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Open-Meteo returns a dictionary for a single location
    and a list for multiple locations.

    Normalize both cases into a list.
    """

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        return [data]

    raise ValidationError(
        "Unexpected Open-Meteo response type: "
        f"{type(data).__name__}"
    )


def validate_location_count(
    results: list[dict[str, Any]],
    expected_count: int,
) -> None:
    """Verify that the API returned every requested location."""

    actual_count = len(results)

    if actual_count != expected_count:
        raise ValidationError(
            f"Expected {expected_count} locations, "
            f"but received {actual_count}."
        )


def validate_location_metadata(
    location: dict[str, Any],
    index: int,
) -> None:
    """Verify basic location metadata exists."""

    required_metadata = [
        "latitude",
        "longitude",
        "timezone",
    ]

    for field in required_metadata:

        if field not in location:
            raise ValidationError(
                f"Location {index} is missing "
                f"metadata field '{field}'."
            )


def validate_daily_structure(
    location: dict[str, Any],
    index: int,
    expected_days: int,
) -> None:
    """Validate daily weather arrays and their lengths."""

    if "daily" not in location:
        raise ValidationError(
            f"Location {index} has no 'daily' section."
        )

    daily = location["daily"]

    missing = [
        variable
        for variable in REQUIRED_DAILY_VARIABLES
        if variable not in daily
    ]

    if missing:
        raise ValidationError(
            f"Location {index} is missing daily "
            f"variables: {missing}"
        )

    date_count = len(daily["time"])

    if date_count != expected_days:
        raise ValidationError(
            f"Location {index} has {date_count} dates; "
            f"expected {expected_days}."
        )

    for variable in REQUIRED_DAILY_VARIABLES[1:]:

        value_count = len(daily[variable])

        if value_count != date_count:
            raise ValidationError(
                f"Location {index}: '{variable}' has "
                f"{value_count} values but 'time' has "
                f"{date_count} dates."
            )


def validate_no_duplicate_dates(
    location: dict[str, Any],
    index: int,
) -> None:
    """Ensure the response does not contain duplicate dates."""

    dates = location["daily"]["time"]

    if len(dates) != len(set(dates)):
        raise ValidationError(
            f"Location {index} contains duplicate dates."
        )


def validate_date_range(
    location: dict[str, Any],
    index: int,
    expected_start: str,
    expected_end: str,
) -> None:
    """Verify the returned first and last dates."""

    dates = location["daily"]["time"]

    if not dates:
        raise ValidationError(
            f"Location {index} contains no dates."
        )

    actual_start = dates[0]
    actual_end = dates[-1]

    if actual_start != expected_start:
        raise ValidationError(
            f"Location {index}: expected first date "
            f"{expected_start}, got {actual_start}."
        )

    if actual_end != expected_end:
        raise ValidationError(
            f"Location {index}: expected last date "
            f"{expected_end}, got {actual_end}."
        )


def validate_response(
    data: dict[str, Any] | list[dict[str, Any]],
    expected_locations: int,
    expected_days: int,
    expected_start: str,
    expected_end: str,
) -> list[dict[str, Any]]:
    """
    Run the complete validation suite.

    Returns the normalized list of location results
    if validation succeeds.
    """

    results = normalize_results(data)

    validate_location_count(
        results,
        expected_locations,
    )

    for index, location in enumerate(
        results,
        start=1,
    ):

        validate_location_metadata(
            location,
            index,
        )

        validate_daily_structure(
            location,
            index,
            expected_days,
        )

        validate_no_duplicate_dates(
            location,
            index,
        )

        validate_date_range(
            location,
            index,
            expected_start,
            expected_end,
        )

    return results