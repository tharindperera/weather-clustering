from pathlib import Path

from budget import (
    ApiBudget,
    BudgetExceededError,
)


BUDGET_FILE = (
    Path("data")
    / "checkpoints"
    / "test_budget.json"
)


def main() -> None:

    print("=" * 70)
    print("API BUDGET TEST")
    print("=" * 70)

    if BUDGET_FILE.exists():
        BUDGET_FILE.unlink()

    budget = ApiBudget(
        budget_file=BUDGET_FILE,
        daily_limit=10,
    )

    print("\nInitial summary:")
    print(budget.summary())

    # -----------------------------------------------------
    # Reserve 3 calls
    # -----------------------------------------------------

    budget.reserve(3)

    print("\nAfter reserving 3:")
    print(budget.summary())

    assert budget.remaining() == 7

    # -----------------------------------------------------
    # Record success
    # -----------------------------------------------------

    budget.record_success()

    print("\nAfter success:")
    print(budget.summary())

    # -----------------------------------------------------
    # Reserve another 7
    # -----------------------------------------------------

    budget.reserve(7)

    print("\nAfter reserving another 7:")
    print(budget.summary())

    assert budget.remaining() == 0

    # -----------------------------------------------------
    # Confirm budget protection
    # -----------------------------------------------------

    try:

        budget.reserve(1)

        raise AssertionError(
            "Budget should have been exceeded."
        )

    except BudgetExceededError:

        print(
            "\nBudget protection worked correctly."
        )

    # -----------------------------------------------------
    # Final
    # -----------------------------------------------------

    print(
        "\nAPI BUDGET TEST PASSED"
    )


if __name__ == "__main__":
    main()