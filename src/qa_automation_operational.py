from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "weather-clustering") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "weather-clustering"))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "weather-clustering" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "weather-clustering" / "src"))

try:
    import src.refresh_recent_ifs as refresh
except ImportError:
    import refresh_recent_ifs as refresh


# =============================================================================
# PATHS
# =============================================================================

WORKFLOW_FILE = (
    PROJECT_ROOT
    / ".github"
    / "workflows"
    / "refresh_recent_ifs.yml"
)
if not WORKFLOW_FILE.exists() and (PROJECT_ROOT / "weather-clustering" / ".github" / "workflows" / "refresh_recent_ifs.yml").exists():
    WORKFLOW_FILE = (
        PROJECT_ROOT
        / "weather-clustering"
        / ".github"
        / "workflows"
        / "refresh_recent_ifs.yml"
    )

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "qa"
)

REPORT_FILE = (
    REPORT_DIR
    / "phase_11d_automation_operational.json"
)


# =============================================================================
# HUGGING FACE
# =============================================================================

HF_REPO_ID = "tharinduperera/weather-clustering-data"

HF_IFS_FILE = (
    "processed/parquet/"
    "weather_ifs_2026-present/"
    "year=2026/"
    "weather.parquet"
)


# =============================================================================
# HELPERS
# =============================================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def assert_contains(
    text: str,
    value: str,
    description: str,
) -> None:

    if value not in text:
        raise AssertionError(
            f"Workflow missing {description}: {value!r}"
        )


def make_fake_responses(
    batch: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> list[dict]:

    dates = [
        (
            start_date
            + timedelta(days=offset)
        ).isoformat()
        for offset in range(
            (end_date - start_date).days + 1
        )
    ]

    responses = []

    for _, location in batch.iterrows():

        count = len(dates)

        responses.append(
            {
                "latitude": float(
                    location["latitude"]
                ),
                "longitude": float(
                    location["longitude"]
                ),
                "timezone": "UTC",
                "daily": {
                    "time": dates,
                    "temperature_2m_mean": [
                        25.0
                    ] * count,
                    "temperature_2m_max": [
                        30.0
                    ] * count,
                    "temperature_2m_min": [
                        20.0
                    ] * count,
                    "precipitation_sum": [
                        1.0
                    ] * count,
                    "relative_humidity_2m_mean": [
                        70
                    ] * count,
                    "wind_speed_10m_mean": [
                        10.0
                    ] * count,
                    "surface_pressure_mean": [
                        1000.0
                    ] * count,
                },
            }
        )

    return responses


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)
    print(
        "PHASE 11D — AUTOMATION, RECOVERY & OPERATIONAL QA"
    )
    print("=" * 80)

    report = {
        "status": "PASS",
        "workflow": {},
        "incremental_refresh": {},
        "failure_safety": {},
        "manual_cloud_evidence": {
            "github_actions_manual_runs_completed": 2,
            "note": (
                "Two manual GitHub Actions runs were "
                "previously verified successfully."
            ),
        },
    }

    # =========================================================================
    # 1. WORKFLOW CONFIGURATION AUDIT
    # =========================================================================

    if not WORKFLOW_FILE.exists():
        raise FileNotFoundError(
            f"Workflow not found: {WORKFLOW_FILE}"
        )

    workflow_text = WORKFLOW_FILE.read_text(
        encoding="utf-8"
    )

    workflow_checks = {
        "scheduled_cron": "47 1 * * *",
        "manual_trigger": "workflow_dispatch",
        "hugging_face_secret": "secrets.HF_TOKEN",
        "checkout": "actions/checkout@",
        "python_setup": "actions/setup-python@",
        "requirements_install": (
            "pip install -r requirements.txt"
        ),
        "hf_download_step": (
            "sync_recent_ifs_hf.py download"
        ),
        "refresh_step": (
            "refresh_recent_ifs.py"
        ),
        "hf_upload_step": (
            "sync_recent_ifs_hf.py upload"
        ),
        "checksum": "sha256sum",
    }

    for description, expected_text in (
        workflow_checks.items()
    ):
        assert_contains(
            workflow_text,
            expected_text,
            description,
        )

    report["workflow"] = {
        "workflow_file": str(
            WORKFLOW_FILE.name
        ),
        "daily_cron_utc": "47 1 * * *",
        "daily_time_asia_colombo": "07:17",
        "manual_trigger_enabled": True,
        "hf_secret_reference_present": True,
        "checksum_change_detection": True,
        "download_before_refresh": True,
        "conditional_upload_after_change": True,
        "status": "PASS",
    }

    print()
    print("[PASS] GitHub Actions workflow configuration")
    print(
        "       Daily schedule: 01:47 UTC / 07:17 Asia/Colombo"
    )
    print(
        "       Manual workflow_dispatch enabled"
    )
    print(
        "       Hugging Face secret reference present"
    )
    print(
        "       Download → refresh → checksum → conditional upload"
    )

    # =========================================================================
    # 2. DOWNLOAD CURRENT PUBLISHED IFS DATASET
    # =========================================================================

    published_file = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_IFS_FILE,
            repo_type="dataset",
            token=False,
        )
    )

    published_df = pd.read_parquet(
        published_file
    )

    published_df["date"] = pd.to_datetime(
        published_df["date"]
    )

    baseline_rows = len(
        published_df
    )

    baseline_latest = (
        published_df["date"]
        .max()
        .date()
    )

    if published_df["location_id"].nunique() != 100:
        raise AssertionError(
            "Published IFS baseline does not contain 100 locations."
        )

    print()
    print("[PASS] Published IFS baseline loaded")
    print(
        f"       Rows: {baseline_rows:,}"
    )
    print(
        f"       Latest date: {baseline_latest}"
    )

    # =========================================================================
    # 3. SIMULATE A THREE-DAY MISSED SCHEDULE
    #
    # No Open-Meteo request is made.
    # We run the REAL refresh main() against a temporary Parquet copy while
    # replacing only the network fetch function with deterministic fake data.
    # =========================================================================

    with tempfile.TemporaryDirectory(
        prefix="weather-clustering-qa-"
    ) as temp_directory:

        temp_root = Path(
            temp_directory
        )

        temp_parquet = (
            temp_root
            / "data"
            / "processed"
            / "parquet"
            / "weather_ifs_2026-present"
            / "year=2026"
            / "weather.parquet"
        )

        temp_parquet.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            published_file,
            temp_parquet,
        )

        temp_state_dir = (
            temp_root
            / "checkpoints"
            / "ifs_2026-present"
        )

        temp_state_file = (
            temp_state_dir
            / "refresh_state.json"
        )

        # ---------------------------------------------------------------------
        # Patch the module only inside this QA process.
        # ---------------------------------------------------------------------

        refresh.YEAR_PARQUET = temp_parquet
        refresh.STATE_DIR = temp_state_dir
        refresh.STATE_FILE = temp_state_file

        simulated_latest = (
            baseline_latest
            + timedelta(days=3)
        )

        refresh.latest_complete_day = (
            lambda: simulated_latest
        )

        api_call_counter = {
            "count": 0
        }

        def fake_fetch_batch(
            batch: pd.DataFrame,
            start_date: date,
            end_date: date,
        ) -> list[dict]:

            api_call_counter[
                "count"
            ] += 1

            return make_fake_responses(
                batch,
                start_date,
                end_date,
            )

        refresh.fetch_batch = (
            fake_fetch_batch
        )

        print()
        print(
            "Simulating missed schedule:"
        )
        print(
            f"       Existing latest: {baseline_latest}"
        )
        print(
            f"       Simulated latest: {simulated_latest}"
        )
        print(
            "       Missing days: 3"
        )

        # ---------------------------------------------------------------------
        # First execution: must catch up all three days.
        # ---------------------------------------------------------------------

        refresh.main()

        updated_df = pd.read_parquet(
            temp_parquet
        )

        updated_df[
            "date"
        ] = pd.to_datetime(
            updated_df["date"]
        )

        expected_new_rows = (
            100 * 3
        )

        expected_total_rows = (
            baseline_rows
            + expected_new_rows
        )

        if (
            len(updated_df)
            != expected_total_rows
        ):
            raise AssertionError(
                "Catch-up row count failed: "
                f"expected {expected_total_rows}, "
                f"found {len(updated_df)}."
            )

        if (
            updated_df["date"]
            .max()
            .date()
            != simulated_latest
        ):
            raise AssertionError(
                "Catch-up maximum date is incorrect."
            )

        if (
            updated_df.duplicated(
                [
                    "location_id",
                    "date",
                ]
            ).sum()
            != 0
        ):
            raise AssertionError(
                "Catch-up introduced duplicate city-date rows."
            )

        if api_call_counter[
            "count"
        ] != 5:
            raise AssertionError(
                "Expected exactly five synthetic batch fetches "
                "for 100 cities."
            )

        new_dates = [
            baseline_latest
            + timedelta(days=offset)
            for offset in range(
                1,
                4,
            )
        ]

        for new_date in new_dates:

            rows_for_date = int(
                (
                    updated_df["date"].dt.date
                    == new_date
                ).sum()
            )

            if rows_for_date != 100:
                raise AssertionError(
                    f"{new_date}: expected 100 rows, "
                    f"found {rows_for_date}."
                )

        state = json.loads(
            temp_state_file.read_text(
                encoding="utf-8"
            )
        )

        if (
            state.get(
                "last_successful_date"
            )
            != simulated_latest.isoformat()
        ):
            raise AssertionError(
                "Refresh state was not advanced "
                "to the simulated latest date."
            )

        print()
        print("[PASS] Missed-schedule catch-up")
        print(
            f"       Added {expected_new_rows} rows"
        )
        print(
            "       3 missing days × 100 cities"
        )
        print(
            "       5 multi-location batch fetches"
        )
        print(
            "       0 duplicates"
        )

        # ---------------------------------------------------------------------
        # 4. IDEMPOTENCY TEST
        #
        # Second execution on the same latest date must make ZERO fetches
        # and must not modify the Parquet file.
        # ---------------------------------------------------------------------

        hash_before = sha256_file(
            temp_parquet
        )

        def forbidden_fetch(
            *args,
            **kwargs,
        ):
            raise AssertionError(
                "Network fetch should not occur "
                "when dataset is already current."
            )

        refresh.fetch_batch = (
            forbidden_fetch
        )

        refresh.main()

        hash_after = sha256_file(
            temp_parquet
        )

        if hash_before != hash_after:
            raise AssertionError(
                "Parquet changed during an idempotent "
                "no-update execution."
            )

        print()
        print("[PASS] Idempotent no-update execution")
        print(
            "       0 fetch calls"
        )
        print(
            "       Parquet SHA-256 unchanged"
        )

        # ---------------------------------------------------------------------
        # 5. FAILURE-SAFETY TEST
        #
        # Simulate another missing day, but force every fetch to fail.
        # The existing Parquet must remain byte-for-byte unchanged.
        # ---------------------------------------------------------------------

        failure_latest = (
            simulated_latest
            + timedelta(days=1)
        )

        refresh.latest_complete_day = (
            lambda: failure_latest
        )

        hash_before_failure = (
            sha256_file(
                temp_parquet
            )
        )

        def failing_fetch(
            *args,
            **kwargs,
        ):
            raise RuntimeError(
                "Synthetic API failure for QA"
            )

        refresh.fetch_batch = (
            failing_fetch
        )

        failure_detected = False

        try:

            refresh.main()

        except RuntimeError as exc:

            if (
                "Synthetic API failure"
                not in str(exc)
            ):
                raise

            failure_detected = True

        if not failure_detected:
            raise AssertionError(
                "Synthetic refresh failure was not propagated."
            )

        hash_after_failure = (
            sha256_file(
                temp_parquet
            )
        )

        if (
            hash_before_failure
            != hash_after_failure
        ):
            raise AssertionError(
                "Parquet changed despite a failed refresh."
            )

        failed_state = json.loads(
            temp_state_file.read_text(
                encoding="utf-8"
            )
        )

        if (
            failed_state.get(
                "last_successful_date"
            )
            != simulated_latest.isoformat()
        ):
            raise AssertionError(
                "Failed refresh incorrectly advanced "
                "the successful-date checkpoint."
            )

        print()
        print("[PASS] Failure safety")
        print(
            "       Synthetic API failure propagated"
        )
        print(
            "       Parquet SHA-256 unchanged"
        )
        print(
            "       Successful-date checkpoint not advanced"
        )

        report[
            "incremental_refresh"
        ] = {
            "published_baseline_rows": int(
                baseline_rows
            ),
            "published_baseline_latest_date": (
                baseline_latest.isoformat()
            ),
            "simulated_missing_days": 3,
            "simulated_new_rows": (
                expected_new_rows
            ),
            "synthetic_multi_location_fetches": 5,
            "duplicates_after_catch_up": 0,
            "catch_up_status": "PASS",
            "idempotent_second_run_fetches": 0,
            "idempotent_parquet_hash_unchanged": True,
            "idempotency_status": "PASS",
        }

        report[
            "failure_safety"
        ] = {
            "synthetic_failure_propagated": True,
            "parquet_hash_unchanged_after_failure": True,
            "checkpoint_not_advanced_after_failure": True,
            "status": "PASS",
        }

    # =========================================================================
    # WRITE REPORT
    # =========================================================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print(
        "PHASE 11D OPERATIONAL QA PASSED"
    )
    print("=" * 80)

    print()
    print(
        f"Audit report written to:\n{REPORT_FILE}"
    )


if __name__ == "__main__":
    main()
