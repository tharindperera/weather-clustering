from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class CheckpointManager:
    """
    Manages persistent progress for the historical ingestion pipeline.

    Each completed work unit is recorded in a JSON manifest so that
    interrupted runs can resume without re-downloading successful data.
    """

    def __init__(self, checkpoint_file: Path) -> None:
        self.checkpoint_file = checkpoint_file

        self.checkpoint_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.data = self._load()

    # ---------------------------------------------------------
    # Load existing checkpoint
    # ---------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if not self.checkpoint_file.exists():
            return {
                "version": 1,
                "created_at": self._now(),
                "updated_at": self._now(),
                "items": {},
            }

        with self.checkpoint_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    # ---------------------------------------------------------
    # Save checkpoint
    # ---------------------------------------------------------

    def _save(self) -> None:
        self.data["updated_at"] = self._now()

        temporary_file = self.checkpoint_file.with_suffix(
            ".tmp"
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

        # Atomic replacement:
        # write temporary file first, then replace the real one.
        temporary_file.replace(
            self.checkpoint_file
        )

    # ---------------------------------------------------------
    # Timestamp
    # ---------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()

    # ---------------------------------------------------------
    # Status
    # ---------------------------------------------------------

    def is_completed(
        self,
        work_id: str,
    ) -> bool:
        item = self.data["items"].get(work_id)

        return (
            item is not None
            and item.get("status") == "success"
        )

    # ---------------------------------------------------------
    # Mark success
    # ---------------------------------------------------------

    def mark_success(
        self,
        work_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:

        self.data["items"][work_id] = {
            "status": "success",
            "completed_at": self._now(),
            "metadata": metadata or {},
        }

        self._save()

    # ---------------------------------------------------------
    # Mark failure
    # ---------------------------------------------------------

    def mark_failure(
        self,
        work_id: str,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:

        self.data["items"][work_id] = {
            "status": "failed",
            "failed_at": self._now(),
            "error": error,
            "metadata": metadata or {},
        }

        self._save()

    # ---------------------------------------------------------
    # Get item
    # ---------------------------------------------------------

    def get(
        self,
        work_id: str,
    ) -> dict[str, Any] | None:

        return self.data["items"].get(
            work_id
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self) -> dict[str, int]:

        items = self.data["items"]

        success = sum(
            1
            for item in items.values()
            if item.get("status") == "success"
        )

        failed = sum(
            1
            for item in items.values()
            if item.get("status") == "failed"
        )

        return {
            "total": len(items),
            "success": success,
            "failed": failed,
        }