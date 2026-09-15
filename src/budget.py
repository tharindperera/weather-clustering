from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class BudgetExceededError(Exception):
    """Raised when the configured project API budget is exhausted."""


class ApiBudget:
    """
    Tracks the project's estimated Open-Meteo API usage.

    This is deliberately a project safety budget, not a claim about
    the provider's internal billing/accounting system.
    """

    def __init__(
        self,
        budget_file: Path,
        daily_limit: int,
    ) -> None:

        self.budget_file = budget_file
        self.daily_limit = daily_limit

        self.budget_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.data = self._load()

        self._reset_if_new_day()

    # ---------------------------------------------------------
    # Current UTC date
    # ---------------------------------------------------------

    @staticmethod
    def _today() -> str:
        return datetime.now(
            timezone.utc
        ).date().isoformat()

    # ---------------------------------------------------------
    # Current timestamp
    # ---------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    def _load(self) -> dict:
        if not self.budget_file.exists():

            return {
                "date": self._today(),
                "estimated_calls": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "updated_at": self._now(),
            }

        with self.budget_file.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    # ---------------------------------------------------------
    # Reset at beginning of a new UTC day
    # ---------------------------------------------------------

    def _reset_if_new_day(self) -> None:

        today = self._today()

        if self.data.get("date") != today:

            self.data = {
                "date": today,
                "estimated_calls": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "updated_at": self._now(),
            }

            self._save()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    def _save(self) -> None:

        self.data["updated_at"] = self._now()

        temporary_file = (
            self.budget_file.with_suffix(".tmp")
        )

        with temporary_file.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.data,
                file,
                indent=2,
            )

        temporary_file.replace(
            self.budget_file
        )

    # ---------------------------------------------------------
    # Check whether another request is allowed
    # ---------------------------------------------------------

    def can_consume(
        self,
        estimated_calls: int = 1,
    ) -> bool:

        current = self.data["estimated_calls"]

        return (
            current + estimated_calls
            <= self.daily_limit
        )

    # ---------------------------------------------------------
    # Reserve usage
    # ---------------------------------------------------------

    def reserve(
        self,
        estimated_calls: int = 1,
    ) -> None:

        if not self.can_consume(
            estimated_calls
        ):

            raise BudgetExceededError(
                "API safety budget exceeded. "
                f"Current usage: "
                f"{self.data['estimated_calls']}, "
                f"requested: {estimated_calls}, "
                f"daily limit: {self.daily_limit}."
            )

        self.data["estimated_calls"] += (
            estimated_calls
        )

        self._save()

    # ---------------------------------------------------------
    # Record successful request
    # ---------------------------------------------------------

    def record_success(self) -> None:

        self.data["successful_requests"] += 1

        self._save()

    # ---------------------------------------------------------
    # Record failed request
    # ---------------------------------------------------------

    def record_failure(self) -> None:

        self.data["failed_requests"] += 1

        self._save()

    # ---------------------------------------------------------
    # Remaining budget
    # ---------------------------------------------------------

    def remaining(self) -> int:

        return (
            self.daily_limit
            - self.data["estimated_calls"]
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self) -> dict:

        return {
            **self.data,
            "daily_limit": self.daily_limit,
            "remaining": self.remaining(),
        }