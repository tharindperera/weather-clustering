from __future__ import annotations

import time
from typing import Any

import requests

from config import (
    BACKOFF_BASE_SECONDS,
    HISTORICAL_API_URL,
    MAX_RETRIES,
    REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)


class OpenMeteoClient:
    """
    Safe client for communicating with the Open-Meteo Historical API.

    Responsibilities:
        - maintain a reusable HTTP session
        - enforce a delay between requests
        - handle HTTP 429 responses
        - retry transient request failures
        - use exponential backoff
        - fail clearly after all retries are exhausted
    """

    def __init__(
        self,
        session: requests.Session | None = None,
    ) -> None:

        self.session = session or requests.Session()

        self._last_request_time: float | None = None

    # -----------------------------------------------------
    # Rate limiting
    # -----------------------------------------------------

    def _wait_before_request(self) -> None:
        """
        Ensure at least REQUEST_DELAY_SECONDS passes between
        requests made by this client.
        """

        if self._last_request_time is None:
            return

        elapsed = time.monotonic() - self._last_request_time

        remaining = (
            REQUEST_DELAY_SECONDS - elapsed
        )

        if remaining > 0:
            print(
                f"Rate limiter: waiting "
                f"{remaining:.2f} seconds..."
            )

            time.sleep(remaining)

    # -----------------------------------------------------
    # HTTP request
    # -----------------------------------------------------

    def _perform_request(
        self,
        params: dict[str, Any],
    ) -> requests.Response:

        self._wait_before_request()

        print("Sending request to Open-Meteo...")

        self._last_request_time = time.monotonic()

        response = self.session.get(
            HISTORICAL_API_URL,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        print(
            f"HTTP {response.status_code}"
        )

        return response

    # -----------------------------------------------------
    # Retry logic
    # -----------------------------------------------------

    def fetch_historical(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:

        last_error: Exception | None = None

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):

            try:

                response = self._perform_request(
                    params
                )

                # -----------------------------------------
                # Success
                # -----------------------------------------

                if response.status_code == 200:

                    print(
                        f"Request successful "
                        f"on attempt {attempt}."
                    )

                    return response.json()

                # -----------------------------------------
                # Rate limited
                # -----------------------------------------

                if response.status_code == 429:
                    retry_after_header = response.headers.get(
                        "Retry-After"
                    )

                    try:
                        response_reason = response.json()
                    except ValueError:
                        response_reason = response.text[:500]

                    reason_text = str(response_reason).lower()

                    print(
                        "HTTP 429: Open-Meteo rate limit response."
                    )

                    print(
                        f"429 response: {response_reason}"
                    )

                    # Open-Meteo explicitly tells us when the
                    # hourly limit has been reached. In that case,
                    # do not keep retrying inside this run.
                    if (
                        "hourly api request limit exceeded"
                        in reason_text
                    ):
                        raise RuntimeError(
                            "Open-Meteo hourly API limit reached. "
                            "Stop the ingestion run and resume "
                            "after the provider's hourly window resets."
                        )

                    # For other 429 responses, honor Retry-After
                    # when provided; otherwise use conservative
                    # exponential backoff.
                    if retry_after_header is not None:
                        try:
                            wait_seconds = max(
                                60,
                                int(float(retry_after_header)),
                            )
                        except ValueError:
                            wait_seconds = (
                                60 * (2 ** (attempt - 1))
                            )
                    else:
                        wait_seconds = (
                            60 * (2 ** (attempt - 1))
                        )

                    wait_seconds = min(
                        wait_seconds,
                        600,
                    )

                    print(
                        f"Waiting {wait_seconds} seconds "
                        f"before retry..."
                    )

                    time.sleep(wait_seconds)

                    continue

                # -----------------------------------------
                # Server-side temporary errors
                # -----------------------------------------

                if response.status_code in {
                    500,
                    502,
                    503,
                    504,
                }:

                    wait_seconds = (
                        BACKOFF_BASE_SECONDS
                        * (2 ** (attempt - 1))
                    )

                    print(
                        f"Temporary server error: "
                        f"HTTP {response.status_code}"
                    )

                    print(
                        f"Waiting "
                        f"{wait_seconds} seconds "
                        f"before retry..."
                    )

                    time.sleep(
                        wait_seconds
                    )

                    continue

                # -----------------------------------------
                # Other HTTP errors
                # -----------------------------------------

                response.raise_for_status()

            except requests.Timeout as exc:

                last_error = exc

                wait_seconds = (
                    BACKOFF_BASE_SECONDS
                    * (2 ** (attempt - 1))
                )

                print(
                    f"Request timed out "
                    f"on attempt {attempt}."
                )

                if attempt < MAX_RETRIES:
                    print(
                        f"Waiting "
                        f"{wait_seconds} seconds "
                        f"before retry..."
                    )

                    time.sleep(
                        wait_seconds
                    )

            except requests.ConnectionError as exc:

                last_error = exc

                wait_seconds = (
                    BACKOFF_BASE_SECONDS
                    * (2 ** (attempt - 1))
                )

                print(
                    f"Connection error "
                    f"on attempt {attempt}."
                )

                if attempt < MAX_RETRIES:
                    print(
                        f"Waiting "
                        f"{wait_seconds} seconds "
                        f"before retry..."
                    )

                    time.sleep(
                        wait_seconds
                    )

            except requests.RequestException as exc:

                last_error = exc

                # Other request-level errors are not
                # automatically retried unless they are
                # explicitly classified above.

                raise RuntimeError(
                    "Open-Meteo request failed."
                ) from exc

        # -------------------------------------------------
        # All retries exhausted
        # -------------------------------------------------

        if last_error is not None:

            raise RuntimeError(
                "Open-Meteo request failed after "
                f"{MAX_RETRIES} attempts."
            ) from last_error

        raise RuntimeError(
            "Open-Meteo request failed after "
            f"{MAX_RETRIES} attempts."
        )

    # -----------------------------------------------------
    # Close session
    # -----------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP session."""

        self.session.close()

    # -----------------------------------------------------
    # Context manager
    # -----------------------------------------------------

    def __enter__(self) -> "OpenMeteoClient":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:

        self.close()