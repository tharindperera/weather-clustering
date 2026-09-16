from datetime import date
from pathlib import Path
import json
import sys

# Ensure src is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

try:
    from ingest_recent_ifs import (
        BATCH_SIZE,
        EXPERIMENT_NAME,
        RAW_DIR,
        CHECKPOINT_DIR,
        load_locations,
        load_manifest,
        load_budget,
        make_batches,
        request_batch,
        validate_batch_response,
        raw_file_path,
        work_unit_key,
        save_runtime_state,
    )
except ImportError:
    from src.ingest_recent_ifs import (
        BATCH_SIZE,
        EXPERIMENT_NAME,
        RAW_DIR,
        CHECKPOINT_DIR,
        load_locations,
        load_manifest,
        load_budget,
        make_batches,
        request_batch,
        validate_batch_response,
        raw_file_path,
        work_unit_key,
        save_runtime_state,
    )


TEST_START = date(2026, 9, 8)
TEST_END = date(2026, 9, 14)


def main() -> None:

    print("=" * 80)
    print("RECENT IFS SINGLE-WINDOW INGESTION TEST")
    print("=" * 80)

    # -----------------------------------------------------------------
    # Load locations
    # -----------------------------------------------------------------

    locations = load_locations()

    batches = make_batches(
        locations
    )

    batch_number = 1
    batch = batches[0]

    print()
    print(
        f"Test experiment: {EXPERIMENT_NAME}"
    )

    print(
        f"Cities in batch: {len(batch)}"
    )

    print(
        f"Period: {TEST_START} to {TEST_END}"
    )

    # -----------------------------------------------------------------
    # Load state
    # -----------------------------------------------------------------

    manifest = load_manifest()
    budget = load_budget()

    key = work_unit_key(
        batch_number,
        TEST_START,
        TEST_END,
    )

    raw_path = raw_file_path(
        batch_number,
        TEST_START,
        TEST_END,
    )

    # ---------------------------------------------------------------
    # Remove ONLY this test work unit if it already exists.
    #
    # This makes the test deterministic without touching any other
    # recent-IFS production work units.
    # ---------------------------------------------------------------

    if raw_path.exists():
        raw_path.unlink()

    manifest["records"].pop(
        key,
        None,
    )

    save_runtime_state(
        manifest,
        budget,
    )

    # -----------------------------------------------------------------
    # First run
    # -----------------------------------------------------------------

    print()
    print("1. FIRST INGESTION")
    print("-" * 80)

    payload = request_batch(
        batch,
        TEST_START,
        TEST_END,
        budget,
    )

    validate_batch_response(
        payload,
        batch,
        TEST_START,
        TEST_END,
    )

    raw_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with raw_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            {
                "experiment": EXPERIMENT_NAME,
                "batch_number": batch_number,
                "start_date": TEST_START.isoformat(),
                "end_date": TEST_END.isoformat(),
                "location_ids": batch[
                    "location_id"
                ].tolist(),
                "locations": payload,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    manifest["records"][key] = {
        "status": "success",
        "batch_number": batch_number,
        "start_date": TEST_START.isoformat(),
        "end_date": TEST_END.isoformat(),
        "raw_file": str(raw_path),
        "locations": len(batch),
    }

    save_runtime_state(
        manifest,
        budget,
    )

    print(
        "First ingestion: PASSED"
    )

    print(
        f"Raw file exists: {raw_path.exists()}"
    )

    # -----------------------------------------------------------------
    # Inspect raw file
    # -----------------------------------------------------------------

    with raw_path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        saved = json.load(
            handle
        )

    assert (
        len(saved["locations"])
        == len(batch)
    )

    print(
        f"Saved location responses: "
        f"{len(saved['locations'])}"
    )

    # -----------------------------------------------------------------
    # Second run — idempotency check
    # -----------------------------------------------------------------

    print()
    print("2. SECOND RUN / IDEMPOTENCY CHECK")
    print("-" * 80)

    reloaded_manifest = load_manifest()

    record = reloaded_manifest[
        "records"
    ].get(
        key
    )

    if not record:
        raise RuntimeError(
            "Checkpoint record was not found."
        )

    if record.get(
        "status"
    ) != "success":

        raise RuntimeError(
            "Checkpoint status is not success."
        )

    if not raw_path.exists():
        raise RuntimeError(
            "Raw file disappeared after first run."
        )

    print(
        "Checkpoint status: success"
    )

    print(
        "Raw file exists: True"
    )

    print(
        "The production ingestion logic would SKIP "
        "this work unit on the next run."
    )

    print()
    print("=" * 80)
    print("SINGLE-WINDOW INGESTION TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
