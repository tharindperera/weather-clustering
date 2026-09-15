from pathlib import Path

from checkpoint import CheckpointManager


CHECKPOINT_FILE = (
    Path("data")
    / "checkpoints"
    / "test_manifest.json"
)


def main() -> None:

    print("=" * 70)
    print("CHECKPOINT MANAGER TEST")
    print("=" * 70)

    # Remove any previous test checkpoint so this test
    # always starts clean.
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    manager = CheckpointManager(
        CHECKPOINT_FILE
    )

    work_id = "batch_000001_window_000001"

    # -----------------------------------------------------
    # 1. Initially not complete
    # -----------------------------------------------------

    print(
        "\nInitially completed:",
        manager.is_completed(work_id),
    )

    assert not manager.is_completed(
        work_id
    )

    # -----------------------------------------------------
    # 2. Mark success
    # -----------------------------------------------------

    manager.mark_success(
        work_id,
        metadata={
            "locations": 20,
            "start_date": "2024-01-01",
            "end_date": "2024-01-14",
        },
    )

    print(
        "After success:",
        manager.is_completed(work_id),
    )

    assert manager.is_completed(
        work_id
    )

    # -----------------------------------------------------
    # 3. Reload from disk
    # -----------------------------------------------------

    manager2 = CheckpointManager(
        CHECKPOINT_FILE
    )

    print(
        "After reload:",
        manager2.is_completed(work_id),
    )

    assert manager2.is_completed(
        work_id
    )

    # -----------------------------------------------------
    # 4. Test summary
    # -----------------------------------------------------

    summary = manager2.summary()

    print("\nSummary:")
    print(summary)

    assert summary["total"] == 1
    assert summary["success"] == 1
    assert summary["failed"] == 0

    print(
        "\nCHECKPOINT TEST PASSED"
    )


if __name__ == "__main__":
    main()
    